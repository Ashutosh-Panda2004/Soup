"""Opt-in graph decoding reaches the real command generation wrappers."""

import re
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output: Rich colours AND wraps."""
    return " ".join(_ANSI_RE.sub("", text).split())


@pytest.mark.parametrize("args", [["infer", "--help"], ["bench", "infer", "--help"]])
def test_cuda_graph_option_is_documented(args):
    from soup_cli.cli import app

    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "--cuda-graphs" in _plain(result.output)


def test_chat_does_not_offer_cuda_graphs():
    """Scope pin. A chat history grows every turn and transformers sizes a static
    cache as max(this request, every earlier one), so each turn would recompile.
    Re-adding the flag to chat needs a capacity reservation first."""
    from soup_cli.cli import app

    result = CliRunner().invoke(app, ["chat", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "--cuda-graphs" not in _plain(result.output)


@pytest.mark.parametrize("enabled", [False, True])
def test_generation_uses_graph_kwargs_only_when_requested(monkeypatch, enabled):
    import torch

    from soup_cli.commands import infer
    from soup_cli.utils import cuda_graphs

    inputs = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)}
    monkeypatch.setattr("soup_cli.utils.vllm.encode_chat_prompt", lambda *a, **kw: inputs)
    config = object()
    prepare = MagicMock(return_value={"cache_implementation": "static", "compile_config": config})
    monkeypatch.setattr(cuda_graphs, "cuda_graph_generation_kwargs", prepare)
    model = MagicMock(device=torch.device("cpu"))
    model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
    tokenizer = MagicMock(pad_token_id=0)
    tokenizer.decode.return_value = "answer"
    response = infer._generate(
        model,
        tokenizer,
        [{"role": "user", "content": "question"}],
        max_tokens=2,
        temperature=0.0,
        cuda_graphs=enabled,
    )
    assert response == ("answer", 2)
    actual = model.generate.call_args.kwargs
    assert actual["max_new_tokens"] == 2
    assert actual["do_sample"] is False
    assert torch.equal(actual["input_ids"], inputs["input_ids"])
    if enabled:
        prepare.assert_called_once_with(model)
        assert actual["cache_implementation"] == "static"
        assert actual["compile_config"] is config
    else:
        prepare.assert_not_called()
        assert "compile_config" not in actual
        assert "cache_implementation" not in actual


@pytest.mark.parametrize("command", ["infer", "bench"])
def test_cli_forwards_opt_in_to_every_generation(monkeypatch, tmp_path, command):
    from soup_cli.cli import app
    from soup_cli.commands import infer

    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (tmp_path / "prompts.txt").write_text("question\n")
    monkeypatch.setattr("soup_cli.utils.gpu.detect_device", lambda: ("cpu", None))
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("soup_cli.utils.cuda_graphs.cuda_graph_generation_kwargs", lambda model: {})
    monkeypatch.setattr(infer, "_load_model", MagicMock(return_value=(object(), object())))
    generate = MagicMock(return_value=("answer", 2))
    monkeypatch.setattr(infer, "_generate", generate)
    if command == "bench":
        args = ["bench", "infer", str(model_dir), "--num-prompts", "1", "--max-tokens", "64"]
    else:
        args = ["infer", "--model", str(model_dir), "--input", "prompts.txt",
                "--output", "output.jsonl"]
    result = CliRunner().invoke(app, [*args, "--cuda-graphs"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert generate.call_count == (2 if command == "bench" else 1)
    assert all(call.kwargs["cuda_graphs"] is True for call in generate.call_args_list)
    if command == "bench":
        assert all(call.kwargs["max_tokens"] == 64 for call in generate.call_args_list)


def test_cuda_graphs_reject_asr_before_loading(monkeypatch, tmp_path):
    from soup_cli.cli import app
    from soup_cli.commands import infer

    monkeypatch.chdir(tmp_path)
    (tmp_path / "audio.jsonl").write_text('{"audio": "speech.wav"}\n')
    load = MagicMock()
    monkeypatch.setattr(infer, "_infer_asr", load)
    result = CliRunner().invoke(
        app,
        ["infer", "--model", ".", "--task", "asr", "--input", "audio.jsonl",
         "--output", "output.jsonl", "--cuda-graphs"],
    )
    assert result.exit_code == 2
    assert "text generation only" in _plain(result.output)
    load.assert_not_called()


def test_unsupported_graph_model_preserves_existing_output(monkeypatch, tmp_path):
    from soup_cli.cli import app
    from soup_cli.commands import infer

    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (tmp_path / "prompts.txt").write_text("question\n")
    output = tmp_path / "output.jsonl"
    output.write_text("previous results\n")
    monkeypatch.setattr("soup_cli.utils.gpu.detect_device", lambda: ("cpu", None))
    monkeypatch.setattr(infer, "_load_model", MagicMock(return_value=(object(), object())))
    reject = MagicMock(side_effect=RuntimeError("Unsupported model for CUDA graphs"))
    monkeypatch.setattr("soup_cli.utils.cuda_graphs.cuda_graph_generation_kwargs", reject)
    generate = MagicMock(side_effect=RuntimeError("Should fail before generation"))
    monkeypatch.setattr(infer, "_generate", generate)
    result = CliRunner().invoke(
        app,
        ["infer", "--model", str(model_dir), "--input", "prompts.txt",
         "--output", str(output), "--cuda-graphs"],
    )
    assert result.exit_code == 1
    assert "Unsupported model for CUDA graphs" in _plain(result.output)
    assert output.read_text() == "previous results\n"
    generate.assert_not_called()
