"""A pull-request push runs a quick CI subset; the full matrix runs at approval.

Measured 2026-09-28 07:30Z: the Actions account sat at its free-plan cap of 20
concurrent jobs (8 ubuntu, 8 windows, 4 macos) with 301 jobs queued, and every
push to a pull request cost 15 jobs of ``ci.yml``. So a push to a pull request
now runs ``lint`` plus ONE test cell, ``test (ubuntu-latest, 3.12)``. The other
eight cells and the five smoke / contract jobs run on pushes to ``main`` and
``release/**``, and on a pull request a maintainer has labelled ``ci:full``.

The merge gate survives that because of one shape, and these tests pin it: the
``test`` job's matrix is whatever the ``plan`` job outputs, so a quick run never
CREATES the other eight required ``test (...)`` contexts. Branch protection
passes a skipped job and waits on a missing one, so it is the missing cells --
not the skipped smokes -- that keep the merge button locked until the full
matrix has run.

Nothing here talks to GitHub. The workflow is read as YAML, the ``plan`` step's
shell runs under a local bash, and the expressions GitHub would evaluate are
evaluated by a small interpreter for exactly the operators they use.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"

LABEL = "ci:full"
GATE = "needs.plan.outputs.full == 'true'"
GATED_JOBS = (
    "type-check",
    "mlx-smoke",
    "torchao-contract",
    "pytorch-smoke",
    "transformers-floor",
)
SUPPORT_MATRIX = {
    "os": ["ubuntu-latest", "windows-latest", "macos-latest"],
    "python-version": ["3.10", "3.11", "3.12"],
}
QUICK_MATRIX = {"os": ["ubuntu-latest"], "python-version": ["3.12"]}
#: The nine ``test`` contexts branch protection on ``main`` requires, read on
#: 2026-09-28 with ``gh api repos/MakazhanAlpamys/Soup/branches/main/protection/
#: required_status_checks`` (the other four are lint, mlx-smoke, pytorch-smoke and
#: transformers-floor). A cell named any other way is a context nobody waits on.
REQUIRED_TEST_CONTEXTS = frozenset(
    {
        "test (ubuntu-latest, 3.10)",
        "test (ubuntu-latest, 3.11)",
        "test (ubuntu-latest, 3.12)",
        "test (windows-latest, 3.10)",
        "test (windows-latest, 3.11)",
        "test (windows-latest, 3.12)",
        "test (macos-latest, 3.10)",
        "test (macos-latest, 3.11)",
        "test (macos-latest, 3.12)",
    }
)


# --- Reading the workflow ----------------------------------------------------


def _workflow() -> dict[str, Any]:
    data = yaml.safe_load(CI.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "ci.yml did not parse to a mapping"
    return data


def _triggers() -> Any:
    data = _workflow()
    # PyYAML (YAML 1.1) reads the bare key `on` as the boolean True.
    return data.get("on", data.get(True))


def _jobs() -> dict[str, Any]:
    return _workflow()["jobs"]


def _plan() -> dict[str, Any]:
    plan = _jobs().get("plan")
    assert isinstance(plan, dict), "ci.yml has no `plan` job"
    return plan


def _needs(job: dict[str, Any]) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _decide_step() -> dict[str, Any]:
    steps = [step for step in _plan().get("steps", []) if step.get("id") == "decide"]
    assert len(steps) == 1, "the plan job must have exactly one step with `id: decide`"
    return steps[0]


def _full_literal() -> str:
    literal = (_plan().get("env") or {}).get("FULL_MATRIX")
    assert isinstance(literal, str), (
        "the plan job must declare the support matrix once, as the JSON literal "
        "env.FULL_MATRIX"
    )
    return literal


def _quick_literal() -> str:
    literal = (_decide_step().get("env") or {}).get("QUICK_MATRIX")
    assert isinstance(literal, str), "the decide step must declare env.QUICK_MATRIX"
    return literal


def _context_names(matrix: dict[str, list[str]]) -> set[str]:
    """What GitHub titles each cell: the values in key order, os first."""
    return {
        f"test ({runner}, {python})"
        for runner in matrix["os"]
        for python in matrix["python-version"]
    }


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


# --- A small evaluator for the GitHub expressions these tests need -----------
#
# It covers exactly what ci.yml's concurrency key and the plan decision use:
# string literals, true/false/null, property paths (with one `*` object
# filter), parentheses, ! == != && || and the functions format() and
# contains(). The semantics are GitHub's: && and || return an OPERAND, not a
# boolean; ! binds tighter than == and !=, which bind tighter than &&, then ||;
# strings compare case-insensitively; a missing property is null. Anything else
# fails loudly instead of being guessed at.

_TOKEN = re.compile(
    r"(?P<string>'(?:[^']|'')*')"
    r"|(?P<op>==|!=|&&|\|\||!|\(|\)|,)"
    r"|(?P<name>[A-Za-z_][\w-]*(?:\.(?:\*|[A-Za-z_][\w-]*))*)"
)
_TEMPLATE = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)
_KEYWORDS = {"true": True, "false": False, "null": None}


def _tokenize(source: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(source):
        if source[position].isspace():
            position += 1
            continue
        match = _TOKEN.match(source, position)
        assert match, f"expression syntax this evaluator does not know: {source[position:]!r}"
        kind = match.lastgroup
        assert kind is not None
        tokens.append((kind, match.group(kind)))
        position = match.end()
    return tokens


def _truthy(value: Any) -> bool:
    return value not in (None, False, 0, "")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return left.casefold() == right.casefold()
    return left == right


def _lookup(context: dict[str, Any], path: str) -> Any:
    value: Any = context
    spread = False
    for part in path.split("."):
        if part == "*":
            assert not spread, "nested object filters are not modelled"
            if isinstance(value, dict):
                value = list(value.values())
            elif not isinstance(value, list):
                value = []
            spread = True
        elif spread:
            value = [item.get(part) for item in value if isinstance(item, dict)]
        else:
            value = value.get(part) if isinstance(value, dict) else None
    return value


def _call(name: str, args: list[Any]) -> Any:
    function = name.lower()  # GitHub function names are case-insensitive
    if function == "format" and args:
        template, *values = args
        return re.sub(
            r"\{(\d+)\}", lambda match: _text(values[int(match.group(1))]), _text(template)
        )
    if function == "contains" and len(args) == 2:
        haystack, needle = args
        if isinstance(haystack, list):
            return any(_equal(item, needle) for item in haystack)
        return _text(needle).casefold() in _text(haystack).casefold()
    raise AssertionError(f"{name}() is not modelled by this test's evaluator; add it to _call()")


class _Expression:
    """Recursive descent over GitHub's precedence: || < && < == != < ! < primary."""

    def __init__(self, source: str, context: dict[str, Any]) -> None:
        self._tokens = _tokenize(source)
        self._position = 0
        self._context = context

    def value(self) -> Any:
        result = self._either()
        assert self._position == len(self._tokens), f"unparsed: {self._tokens[self._position:]}"
        return result

    def _take(self, operator: str) -> bool:
        if self._position < len(self._tokens) and self._tokens[self._position] == (
            "op",
            operator,
        ):
            self._position += 1
            return True
        return False

    def _either(self) -> Any:
        result = self._both()
        while self._take("||"):
            right = self._both()
            result = result if _truthy(result) else right
        return result

    def _both(self) -> Any:
        result = self._compare()
        while self._take("&&"):
            right = self._compare()
            result = right if _truthy(result) else result
        return result

    def _compare(self) -> Any:
        result = self._unary()
        while True:
            if self._take("=="):
                result = _equal(result, self._unary())
            elif self._take("!="):
                result = not _equal(result, self._unary())
            else:
                return result

    def _unary(self) -> Any:
        if self._take("!"):
            return not _truthy(self._unary())
        return self._primary()

    def _primary(self) -> Any:
        assert self._position < len(self._tokens), "the expression ends early"
        kind, text = self._tokens[self._position]
        self._position += 1
        if kind == "string":
            return text[1:-1].replace("''", "'")
        if (kind, text) == ("op", "("):
            result = self._either()
            assert self._take(")"), "unbalanced parenthesis"
            return result
        assert kind == "name", f"unexpected {text!r}"
        if self._take("("):
            args: list[Any] = []
            if not self._take(")"):
                args.append(self._either())
                while self._take(","):
                    args.append(self._either())
                assert self._take(")"), f"unterminated call to {text}()"
            return _call(text, args)
        if text in _KEYWORDS:
            return _KEYWORDS[text]
        return _lookup(self._context, text)


def _render(template: str, context: dict[str, Any]) -> str:
    """Substitute every ``${{ ... }}`` in *template* the way the runner would."""
    return _TEMPLATE.sub(
        lambda match: _text(_Expression(match.group(1), context).value()), template
    )


def _push(ref: str = "refs/heads/main", *, run_id: str = "900") -> dict[str, Any]:
    return {
        "github": {
            "workflow": "CI",
            "event_name": "push",
            "ref": ref,
            "sha": "a" * 40,
            "run_id": run_id,
            "event": {"ref": ref},
        }
    }


def _pull_request(
    action: str,
    *,
    labels: tuple[str, ...] = (),
    label: str | None = None,
    run_id: str = "100",
) -> dict[str, Any]:
    """A pull_request context shaped like GitHub's payload.

    On ``labeled``, ``event.label`` is the label just added and
    ``event.pull_request.labels`` already includes it.
    """
    names = list(labels)
    event: dict[str, Any] = {"action": action, "number": 7}
    if label is not None:
        event["label"] = {"name": label}
        if label not in names:
            names.append(label)
    event["pull_request"] = {"labels": [{"name": name} for name in names]}
    return {
        "github": {
            "workflow": "CI",
            "event_name": "pull_request",
            "ref": "refs/pull/7/merge",
            "sha": "b" * 40,
            "run_id": run_id,
            "event": event,
        }
    }


def _group(context: dict[str, Any]) -> str:
    return _render(_workflow()["concurrency"]["group"], context)


def _full_for(context: dict[str, Any]) -> str:
    return _render(_decide_step()["env"]["FULL"], context)


# --- Tests -------------------------------------------------------------------


class TestTheEvaluatorHasTeeth:
    """CONTROL. The concurrency and plan tests below are only as good as this."""

    def test_and_or_return_operands_with_github_precedence(self):
        context = {"github": {"ref": "r"}}
        assert _Expression("'a' && 'b' || 'c'", context).value() == "b"
        assert _Expression("null && 'b' || 'c'", context).value() == "c"
        assert _Expression("'' || github.ref", context).value() == "r"
        assert _Expression("!('x' == 'y') && 'yes'", context).value() == "yes"

    def test_strings_compare_case_insensitively(self):
        assert _Expression("'CI:Full' == 'ci:full'", {}).value() is True
        assert _Expression("'bug' != 'ci:full'", {}).value() is True

    def test_missing_properties_are_null_and_the_filter_spreads(self):
        context = {"github": {"event": {"items": [{"name": "a"}, {"name": "B"}]}}}
        assert _Expression("github.event.nothing.here", context).value() is None
        assert _Expression("contains(github.event.items.*.name, 'b')", context).value()
        assert not _Expression("contains(github.event.none.*.name, 'b')", context).value()

    def test_templates_and_format(self):
        context = {"github": {"workflow": "CI", "run_id": "42"}}
        rendered = _render("${{ github.workflow }}-${{ format('x-{0}', github.run_id) }}", context)
        assert rendered == "CI-x-42"

    def test_what_it_does_not_model_fails_instead_of_guessing(self):
        with pytest.raises(AssertionError, match="not modelled"):
            _Expression("startsWith(github.ref, 'x')", {}).value()
        with pytest.raises(AssertionError, match="does not know"):
            _Expression("github.run_number >= 2", {}).value()


class TestTriggers:
    def test_pull_requests_run_on_these_five_actions(self):
        pull_request = _triggers()["pull_request"]
        assert pull_request["branches"] == ["main"]
        assert sorted(pull_request.get("types", [])) == sorted(
            ["opened", "synchronize", "reopened", "ready_for_review", "labeled"]
        ), "without `labeled`, adding ci:full at approval starts nothing"

    def test_there_is_no_pull_request_target(self):
        """A labelled run stays an ordinary pull_request run, with the same token and secrets
        a push to the pull request gets -- never the privileged _target variant."""
        assert "pull_request_target" not in _triggers()

    def test_pushes_to_main_and_release_branches_still_trigger(self):
        assert _triggers()["push"]["branches"] == ["main", "release/**"]


class TestPlanDecision:
    def test_plan_is_short_unconditional_and_exports_its_decision(self):
        plan = _plan()
        assert plan["runs-on"] == "ubuntu-latest"
        assert plan["timeout-minutes"] == 5
        assert "if" not in plan and "needs" not in plan
        assert plan["outputs"] == {
            "full": "${{ steps.decide.outputs.full }}",
            "matrix": "${{ steps.decide.outputs.matrix }}",
        }

    @pytest.mark.parametrize(
        ("context", "expected"),
        [
            pytest.param(_push("refs/heads/main"), "true", id="push-main"),
            pytest.param(_push("refs/heads/release/v0.76.0"), "true", id="push-release"),
            pytest.param(_pull_request("opened"), "false", id="pr-opened"),
            pytest.param(
                _pull_request("synchronize", labels=("bug",)), "false", id="pr-push-other-label"
            ),
            pytest.param(
                _pull_request("synchronize", labels=("bug", LABEL)), "true", id="pr-push-ci-full"
            ),
            pytest.param(_pull_request("labeled", label=LABEL), "true", id="adds-ci-full"),
            pytest.param(_pull_request("labeled", label="bug"), "false", id="adds-other"),
            pytest.param(
                _pull_request("labeled", label="bug", labels=(LABEL,)),
                "true",
                id="adds-other-while-ci-full",
            ),
        ],
    )
    def test_full_follows_the_event_and_the_current_labels(self, context, expected):
        assert _full_for(context) == expected


class TestGatedJobs:
    @pytest.mark.parametrize("name", GATED_JOBS)
    def test_the_smoke_and_contract_jobs_run_only_on_a_full_run(self, name):
        job = _jobs()[name]
        assert "plan" in _needs(job), f"{name} does not need plan"
        assert job.get("if") == GATE, f"{name} is not gated on plan's decision"

    def test_lint_always_runs_and_waits_for_nothing(self):
        lint = _jobs()["lint"]
        assert "if" not in lint
        assert "needs" not in lint

    def test_every_job_is_classified(self):
        """A new job must be declared quick or full here, or it lands on every PR push."""
        assert sorted(_jobs()) == sorted(["plan", "lint", "test", *GATED_JOBS])


class TestMatrix:
    def test_the_test_job_takes_its_matrix_from_plan(self):
        """A static matrix with a per-cell `if:` would SKIP the eight cells, and a skipped
        job passes branch protection; a matrix from plan never creates them."""
        test = _jobs()["test"]
        assert "plan" in _needs(test)
        assert "if" not in test
        assert test["strategy"]["fail-fast"] is False
        assert test["strategy"]["matrix"] == "${{ fromJSON(needs.plan.outputs.matrix) }}"
        assert test["runs-on"] == "${{ matrix.os }}"

    def test_the_full_matrix_is_the_support_matrix(self):
        full = json.loads(_full_literal())
        assert full == SUPPORT_MATRIX
        assert list(full) == ["os", "python-version"], "the key order names the check contexts"
        assert all(isinstance(python, str) for python in full["python-version"]), (
            "a JSON number 3.10 is 3.1, which renames the check context"
        )

    def test_the_full_matrix_names_the_nine_required_test_contexts(self):
        assert _context_names(json.loads(_full_literal())) == REQUIRED_TEST_CONTEXTS

    def test_the_quick_matrix_is_the_newest_python_on_ubuntu(self):
        quick = json.loads(_quick_literal())
        assert quick == QUICK_MATRIX
        assert list(quick) == ["os", "python-version"]
        names = _context_names(quick)
        assert len(names) == 1 and names <= REQUIRED_TEST_CONTEXTS
        full_pythons = json.loads(_full_literal())["python-version"]
        assert quick["python-version"] == [max(full_pythons, key=_version)]

    def test_the_matrix_literals_are_single_lines(self):
        """$GITHUB_OUTPUT takes one `name=value` per line; a newline would end the value."""
        assert "\n" not in _full_literal()
        assert "\n" not in _quick_literal()


class TestConcurrency:
    def test_a_label_other_than_ci_full_cannot_share_the_pull_request_group(self):
        pull_request = _group(_pull_request("synchronize", labels=(LABEL,)))
        for labels in ((), (LABEL,)):
            labelled = _group(_pull_request("labeled", label="bug", labels=labels))
            assert labelled != pull_request, (
                "a run started by an unrelated label shares the pull request's group, "
                "so it cancels a full matrix already running there"
            )

    def test_each_such_run_is_alone_in_its_group(self):
        first = _group(_pull_request("labeled", label="bug", run_id="1"))
        second = _group(_pull_request("labeled", label="bug", run_id="2"))
        assert first != second

    def test_adding_ci_full_supersedes_the_quick_run(self):
        quick = _group(_pull_request("synchronize"))
        assert _group(_pull_request("labeled", label=LABEL, run_id="2")) == quick

    @pytest.mark.parametrize("action", ["opened", "synchronize", "reopened", "ready_for_review"])
    def test_every_other_pull_request_event_still_supersedes(self, action):
        earlier = _group(_pull_request("synchronize", run_id="1"))
        assert _group(_pull_request(action, run_id="2")) == earlier

    def test_a_push_is_never_in_a_pull_request_or_label_group(self):
        for ref in ("refs/heads/main", "refs/heads/release/v0.76.0"):
            push = _group(_push(ref, run_id="1"))
            assert push != _group(_pull_request("synchronize", run_id="1"))
            assert push != _group(_pull_request("labeled", label="bug", run_id="1"))

    def test_cancellation_stays_pull_request_only(self):
        concurrency = _workflow()["concurrency"]
        assert concurrency["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"

    @pytest.mark.parametrize("label", [LABEL, LABEL.upper(), "bug"])
    def test_the_group_and_the_plan_agree_on_what_ci_full_is(self, label):
        """Renaming the label in one expression only would make the approval label either
        start a quick run or cancel nothing -- both silently."""
        event = _pull_request("labeled", label=label)
        joins = _group(event) == _group(_pull_request("synchronize"))
        assert joins == (_full_for(event) == "true")


def _posix_bash() -> str | None:
    """A bash that runs a POSIX script; on Windows, Git for Windows' copy.

    On Windows ``shutil.which("bash")`` can answer ``System32\\bash.exe``, the WSL
    launcher, which does not run the script at all. Git for Windows ships a real
    bash beside git, and the Windows runners and dev boxes here all have git.
    """
    if sys.platform != "win32":
        return shutil.which("bash")
    git = shutil.which("git")
    if git is None:
        return None
    for parent in Path(git).resolve().parents:
        candidate = parent / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    return None


BASH = _posix_bash()


@pytest.mark.skipif(BASH is None, reason="no POSIX bash found (on Windows: Git for Windows)")
class TestTheDecideStepScript:
    """Run the plan step's own shell, as the runner would, against a fake $GITHUB_OUTPUT."""

    def _outputs(self, tmp_path: Path, full: str) -> list[str]:
        assert BASH is not None
        output = tmp_path / "github_output"
        output.write_text("", encoding="utf-8")
        script = tmp_path / "decide.sh"
        script.write_bytes(_decide_step()["run"].encode("utf-8"))
        env = {
            **os.environ,
            "FULL": full,
            "FULL_MATRIX": _full_literal(),
            "QUICK_MATRIX": _quick_literal(),
            "GITHUB_OUTPUT": output.as_posix(),
        }
        result = subprocess.run(
            [BASH, "--noprofile", "--norc", "-eo", "pipefail", script.as_posix()],
            env=env,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
        return output.read_text(encoding="utf-8").splitlines()

    def test_a_full_run_outputs_the_support_matrix(self, tmp_path):
        lines = self._outputs(tmp_path, "true")
        assert len(lines) == 2, lines
        assert lines[0] == "full=true"
        key, _, value = lines[1].partition("=")
        assert key == "matrix"
        assert json.loads(value) == SUPPORT_MATRIX

    @pytest.mark.parametrize("full", ["false", "", "yes"])
    def test_anything_but_true_is_the_quick_subset(self, tmp_path, full):
        """Quick is the fail-closed answer on a pull request: fewer cells, more missing
        required contexts, a locked merge button."""
        lines = self._outputs(tmp_path, full)
        assert len(lines) == 2, lines
        assert lines[0] == "full=false"
        key, _, value = lines[1].partition("=")
        assert key == "matrix"
        assert json.loads(value) == QUICK_MATRIX


def test_contributing_tells_contributors_about_the_label():
    assert f"`{LABEL}`" in CONTRIBUTING.read_text(encoding="utf-8")
