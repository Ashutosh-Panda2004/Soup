"""CUDA decode backend safety, static-input handling, and generation compatibility."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from soup_cli.utils import cuda_graphs


def _model(device: str = "cuda:0") -> object:
    model_type = type(
        "Qwen2ForCausalLM", (), {"__module__": "transformers.models.qwen2.modeling_qwen2"}
    )
    model = model_type()
    model.training = False
    model.device = device
    model.config = SimpleNamespace(model_type="qwen2", is_encoder_decoder=False)
    model.generation_config = SimpleNamespace(cache_implementation=None, temperature=0.7)
    model.hf_device_map = {"": 0}
    model.parameters = lambda: iter([SimpleNamespace(device=device)])
    model.buffers = lambda: iter([])
    model.modules = lambda: iter([model])
    model._supports_default_dynamic_cache = lambda: True
    return model


def test_generation_kwargs_preserve_defaults_and_register_once(monkeypatch):
    model = _model()
    before = vars(model.generation_config).copy()
    register = Mock()
    monkeypatch.setattr(cuda_graphs, "_register_backend", register)
    result = cuda_graphs.cuda_graph_generation_kwargs(model)
    assert vars(model.generation_config) == before
    assert result["cache_implementation"] == "static"
    assert result["disable_compile"] is False
    config = result["compile_config"]
    assert config.backend == "soup_cudagraphs"
    assert config.mode is None
    assert config.fullgraph is True
    assert config.dynamic is False
    register.assert_called_once()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda model: setattr(model, "training", True), "eval"),
        (lambda model: setattr(model, "is_quantized", True), "quantiz"),
        (lambda model: setattr(model, "hf_quantizer", object()), "quantiz"),
        (lambda model: setattr(model, "hf_device_map", {"model": 0, "head": "cpu"}), "offload"),
        (lambda model: setattr(model, "hf_device_map", {"model": 0, "head": "disk"}), "offload"),
        (lambda model: setattr(model, "_supports_default_dynamic_cache", lambda: False), "cache"),
        (lambda model: setattr(model.config, "model_type", "gpt2"), "Qwen2.*Llama"),
        (lambda model: setattr(model, "_hf_hook", SimpleNamespace(offload=True)), "offload"),
    ],
)
def test_rejects_unsupported_models_before_registering(monkeypatch, change, message):
    model = _model()
    change(model)
    register = Mock()
    monkeypatch.setattr(cuda_graphs, "_register_backend", register)
    with pytest.raises(RuntimeError, match=message):
        cuda_graphs.cuda_graph_generation_kwargs(model)
    register.assert_not_called()


@pytest.mark.parametrize("device", ["cpu", "meta", "mps"])
def test_rejects_non_cuda_parameters(monkeypatch, device):
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    with pytest.raises(RuntimeError, match="CUDA"):
        cuda_graphs.cuda_graph_generation_kwargs(_model(device))


def test_rejects_parameters_spread_across_gpus(monkeypatch):
    model = _model()
    model.parameters = lambda: iter(
        [SimpleNamespace(device="cuda:0"), SimpleNamespace(device="cuda:1")]
    )
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    with pytest.raises(RuntimeError, match="single CUDA"):
        cuda_graphs.cuda_graph_generation_kwargs(model)


def test_rejects_cpu_buffers_and_nested_offload_hooks(monkeypatch):
    model = _model()
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    model.buffers = lambda: iter([SimpleNamespace(device="cpu")])
    with pytest.raises(RuntimeError, match="CUDA"):
        cuda_graphs.cuda_graph_generation_kwargs(model)
    model.buffers = lambda: iter([])
    model._hf_hook = SimpleNamespace(hooks=[SimpleNamespace(offload=True)])
    with pytest.raises(RuntimeError, match="offload"):
        cuda_graphs.cuda_graph_generation_kwargs(model)


def test_rejects_streaming_wrappers(monkeypatch):
    model = _model()
    streamed = type(
        "StreamedDecoderLayer", (), {"__module__": "soup_cli.utils.layer_stream_runtime"}
    )()
    model.modules = lambda: iter([model, streamed])
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    with pytest.raises(RuntimeError, match="stream"):
        cuda_graphs.cuda_graph_generation_kwargs(model)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("use_cache", False, "use_cache"),
        ("num_beams", 2, "beam"),
        ("prompt_lookup_num_tokens", 3, "assisted"),
        ("assistant_early_exit", 1, "assisted"),
        ("prefill_chunk_size", 32, "prefill"),
        ("cache_implementation", "offloaded", "cache"),
        ("cache_implementation", "quantized", "cache"),
    ],
)
def test_rejects_generation_defaults_that_bypass_or_conflict_with_graphs(
    monkeypatch, field, value, message
):
    model = _model()
    setattr(model.generation_config, field, value)
    register = Mock()
    monkeypatch.setattr(cuda_graphs, "_register_backend", register)
    with pytest.raises(RuntimeError, match=message):
        cuda_graphs.cuda_graph_generation_kwargs(model)
    register.assert_not_called()


def test_cache_disabled_in_model_config_and_explicit_generation_override(monkeypatch):
    model = _model()
    model.config.use_cache = False
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    with pytest.raises(RuntimeError, match="use_cache"):
        cuda_graphs.cuda_graph_generation_kwargs(model)
    model.generation_config.use_cache = True
    assert cuda_graphs.cuda_graph_generation_kwargs(model)["cache_implementation"] == "static"


def test_rejects_training_submodule_in_eval_model(monkeypatch):
    model = _model()
    model.modules = lambda: iter([model, SimpleNamespace(training=True)])
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    with pytest.raises(RuntimeError, match="eval"):
        cuda_graphs.cuda_graph_generation_kwargs(model)


@pytest.mark.parametrize("implementation", ["flash_attention_2", "flash_attention_3"])
def test_rejects_flash_attention_partial_graph_override(monkeypatch, implementation):
    model = _model()
    model.config._attn_implementation = implementation
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    with pytest.raises(RuntimeError, match="attention"):
        cuda_graphs.cuda_graph_generation_kwargs(model)


@pytest.mark.parametrize(
    "metadata", [None, SimpleNamespace(), SimpleNamespace(static_input_indices=[])]
)
def test_missing_static_metadata_fails_closed(metadata):
    context = SimpleNamespace(fw_metadata=metadata)
    with pytest.raises(RuntimeError, match="static input"):
        cuda_graphs._static_input_indices(context, 4)


@pytest.mark.parametrize("indices", [[-1], [4], [True], [1, 1], ["1"]])
def test_invalid_static_metadata_fails_closed(indices):
    context = SimpleNamespace(fw_metadata=SimpleNamespace(static_input_indices=indices))
    with pytest.raises(RuntimeError, match="static input"):
        cuda_graphs._static_input_indices(context, 4)


def test_static_indices_use_aot_coordinates_including_inlined_weights():
    context = SimpleNamespace(fw_metadata=SimpleNamespace(static_input_indices=[0, 2, 3]))
    assert cuda_graphs._static_input_indices(context, 4) == [0, 2, 3]


def test_missing_private_api_has_actionable_error(monkeypatch):
    def missing(name):
        raise ImportError("removed private module")

    monkeypatch.setattr(cuda_graphs, "import_module", missing)
    with pytest.raises(RuntimeError, match="PyTorch.*CUDA graph"):
        cuda_graphs._load_backend_api()


def test_backend_registration_is_idempotent_and_rejects_name_collision(monkeypatch):
    registry = {}

    def register(backend, *, name):
        assert name not in registry
        registry[name] = backend

    register_mock = Mock(side_effect=register)
    monkeypatch.setattr(
        cuda_graphs,
        "_load_backend_api",
        lambda: SimpleNamespace(registry=registry, register_backend=register_mock),
    )
    cuda_graphs._register_backend()
    cuda_graphs._register_backend()
    register_mock.assert_called_once_with(cuda_graphs._soup_cudagraphs, name="soup_cudagraphs")
    registry["soup_cudagraphs"] = object()
    with pytest.raises(RuntimeError, match="already registered"):
        cuda_graphs._register_backend()


def test_backend_refuses_training_before_graph_capture(monkeypatch):
    import torch

    monkeypatch.setattr(cuda_graphs, "_load_backend_api", lambda: SimpleNamespace(torch=torch))
    with torch.enable_grad(), pytest.raises(RuntimeError, match="inference"):
        cuda_graphs._soup_cudagraphs(object(), [])


def test_backend_uses_static_metadata_and_preserves_functionalization(monkeypatch):
    import torch

    context = SimpleNamespace(fw_metadata=SimpleNamespace(static_input_indices=[0, 2]))
    fake_graph = SimpleNamespace(graph=object())
    captured = {}

    def replay(inputs):
        return inputs

    def aot_autograd(**kwargs):
        captured.update(kwargs)
        return lambda graph, inputs: kwargs["inference_compiler"](graph, inputs)

    api = SimpleNamespace(
        torch=torch,
        tracing_context=SimpleNamespace(try_get=lambda: context),
        aot_autograd=aot_autograd,
        boxed_nop=lambda graph, inputs: replay,
        find_input_mutations=lambda graph: set(),
        get_device_node_mapping=lambda graph: {torch.device("cuda:0"): object()},
        check_devices=lambda devices: None,
        incompatible_node=lambda graph: None,
        get_stack_traces=lambda graph: [],
        get_placeholder_info=lambda graph: [],
        cudagraphify=Mock(return_value=replay),
    )
    monkeypatch.setattr(cuda_graphs, "_load_backend_api", lambda: api)
    with torch.no_grad():
        result = cuda_graphs._soup_cudagraphs(fake_graph, [1, 2, 3])
    assert captured["keep_inference_input_mutations"] is False
    assert api.cudagraphify.call_args.args[2] == [0, 2]
    assert api.cudagraphify.call_args.kwargs["is_inference"] is True
    assert result is replay
    with pytest.raises(RuntimeError, match="inference"):
        captured["bw_compiler"]()
    with pytest.raises(RuntimeError, match="inference"):
        captured["fw_compiler"]()

    # If metadata disappears or mutation targets an ordinary input, no capture
    # (and therefore no graph-pool weight allocation) may be attempted.
    api.cudagraphify.reset_mock()
    context.fw_metadata = None
    with pytest.raises(RuntimeError, match="static input"):
        captured["inference_compiler"](fake_graph, [1, 2, 3])
    api.cudagraphify.assert_not_called()
    context.fw_metadata = SimpleNamespace(static_input_indices=[0, 2])
    api.find_input_mutations = lambda graph: {1}
    with pytest.raises(RuntimeError, match="mutation"):
        captured["inference_compiler"](fake_graph, [1, 2, 3])
    api.cudagraphify.assert_not_called()


@pytest.mark.gpu
def test_tiny_llama_repeated_generation_keeps_tokens_and_default_config():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(17)
    model = (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
        )
        .to(device="cuda", dtype=torch.float16)
        .eval()
    )
    before = model.generation_config.to_dict()
    prompts = ([1, 5, 9], [1, 7, 8, 4], [1, 5, 9])
    kwargs = cuda_graphs.cuda_graph_generation_kwargs(model)
    with torch.no_grad():
        for prompt in prompts:
            ids = torch.tensor([prompt], device="cuda")
            mask = torch.ones_like(ids)
            normal = model.generate(ids, attention_mask=mask, max_new_tokens=8, do_sample=False)
            graphed = model.generate(
                ids, attention_mask=mask, max_new_tokens=8, do_sample=False, **kwargs
            )
            torch.testing.assert_close(graphed, normal, rtol=0, atol=0)
    assert model.generation_config.to_dict() == before
