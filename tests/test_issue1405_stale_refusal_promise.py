"""Regression guard for #1405: user-facing refusals must not promise past releases.

``test_issue823_stale_feature_descriptions.py`` guards schema descriptions and
CLI help text, but a refusal raised from ``soup_cli.utils`` (re-raised by the
schema cross-validator as ``ValueError(str(exc))``) slips through it. This
module scans every ``raise`` under ``src/soup_cli/utils/`` for versioned
release promises that are already in the past, and pins the wording of the
``task='unlearn'`` + ``backend='mlx'`` refusal.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from soup_cli import __version__
from soup_cli.utils.unlearning import validate_unlearn_compat

ROOT = Path(__file__).parents[1]
UTILS_DIR = ROOT / "src" / "soup_cli" / "utils"

VERSIONED_RELEASE_PROMISE = re.compile(
    r"(?:deferred\s+to|lands\s+in|ships\s+in)\s+v"
    r"(?P<version>\d+(?:\.\d+){1,2})",
    re.IGNORECASE,
)


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in value.split(".")]
    padded = parts + [0, 0]
    return padded[0], padded[1], padded[2]


CURRENT_RELEASE = _version_tuple(__version__)


def _has_expired_version_promise(text: str) -> bool:
    return any(
        _version_tuple(match.group("version")) <= CURRENT_RELEASE
        for match in VERSIONED_RELEASE_PROMISE.finditer(text)
    )


def _raise_string_literals(path: Path) -> list[tuple[int, str]]:
    """Collect (lineno, literal) pairs from every ``raise`` in a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    literals: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        for sub in ast.walk(node.exc):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                literals.append((node.lineno, sub.value))
    return literals


def test_no_raise_under_utils_promises_a_past_release() -> None:
    offenders = [
        f"{path.relative_to(ROOT)}:{lineno}: {text[:100]}"
        for path in sorted(UTILS_DIR.glob("*.py"))
        for lineno, text in _raise_string_literals(path)
        if _has_expired_version_promise(text)
    ]
    assert offenders == []


def test_mlx_unlearn_refusal_names_supported_backends() -> None:
    with pytest.raises(ValueError, match="not supported") as exc_info:
        validate_unlearn_compat(task="unlearn", backend="mlx")
    message = str(exc_info.value)
    assert not _has_expired_version_promise(message)
    assert "transformers" in message
    assert "unsloth" in message
