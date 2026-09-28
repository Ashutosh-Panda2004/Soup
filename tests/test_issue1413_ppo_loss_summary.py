import ast
from pathlib import Path

from soup_cli.trainer.loss_summary import summarize_training_loss


def _ppo_history():
    """Entries shaped like trl 0.29 PPOTrainer's ``state.log_history``.

    Per-step losses live under ``loss/policy_avg``; there is no ``loss``
    key and no final ``train_loss`` mean, which is exactly why a completed
    PPO run ended with "Loss: unavailable" (#1413).
    """
    return [
        {
            "loss/policy_avg": 1.42,
            "loss/policy": 1.42,
            "loss/kl": 0.013,
            "loss/clipfrac": 0.02,
            "objective/kl": 12.1,
            "ppo/learning_rate": 1e-05,
            "step": 10,
        },
        {
            "loss/policy_avg": 0.87,
            "loss/policy": 0.87,
            "loss/kl": 0.009,
            "loss/clipfrac": 0.01,
            "objective/kl": 9.4,
            "ppo/learning_rate": 1e-05,
            "step": 20,
        },
    ]


def test_ppo_policy_avg_history_yields_a_delta_not_unavailable():
    summary = summarize_training_loss(_ppo_history(), loss_key="loss/policy_avg")

    assert summary["initial_loss"] == 1.42
    assert summary["final_loss"] == 0.87
    assert summary["loss_summary_kind"] == "delta"
    assert summary["loss_key"] == "loss/policy_avg"


def test_default_key_still_reports_unavailable_for_ppo_history():
    # Without the declared key the PPO-shaped history has no usable loss,
    # which is the bug: the key must come from the trainer, not be guessed.
    summary = summarize_training_loss(_ppo_history())

    assert summary["loss_summary_kind"] == "unavailable"


def test_single_ppo_step_is_tagged_with_its_key():
    summary = summarize_training_loss([_ppo_history()[0]], loss_key="loss/policy_avg")

    assert summary["loss_summary_kind"] == "single"
    assert summary["initial_loss"] == summary["final_loss"] == 1.42
    assert summary["loss_key"] == "loss/policy_avg"


def test_default_key_is_not_recorded_for_sft_style_summaries():
    summary = summarize_training_loss([{"loss": 2.0}, {"loss": 1.0}])

    assert summary["loss_summary_kind"] == "delta"
    assert "loss_key" not in summary


def test_ppo_trainer_declares_the_policy_avg_key():
    # Guard: the trl-path call in ppo.py must keep declaring the key instead
    # of the summary guessing it (the manual loop normalizes to "loss").
    source = (Path(__file__).parents[1] / "src" / "soup_cli" / "trainer" / "ppo.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    declaring_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "summarize_training_loss"
        and any(
            kw.arg == "loss_key"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value == "loss/policy_avg"
            for kw in node.keywords
        )
    ]
    assert len(declaring_calls) == 1
