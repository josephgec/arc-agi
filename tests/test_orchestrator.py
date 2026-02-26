"""Tests for agents/orchestrator.py — the state-machine orchestrator."""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from agents.orchestrator import (
    Orchestrator,
    STATE_CANDIDATE,
    STATE_CODING,
    STATE_CRITICIZING,
    STATE_EVALUATING,
    STATE_HYPOTHESIZING,
)
from agents.roles import Hypothesizer, Coder, Critic


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _grid(*rows):
    return np.array(rows, dtype=np.int32)


def simple_task():
    """Minimal ARC task: recolor 1 → 2."""
    inp = _grid([0, 1], [1, 0])
    out = _grid([0, 2], [2, 0])
    return {
        "train": [{"input": inp, "output": out}],
        "test":  [{"input": inp, "output": out}],
    }


def _correct_code() -> str:
    return "```python\ndef transform(grid):\n    return recolor(grid, 1, 2)\n```"


def _wrong_code() -> str:
    return "```python\ndef transform(grid):\n    return grid.copy()\n```"


# Three hypothesis strings that pass the 80-char minimum filter in _parse_hypotheses.
_H1 = (
    "1. Rotate the grid 90 degrees clockwise and recolor all blue cells to red.\n"
    "   - step 1: apply rotate(grid, 3)\n"
    "   - step 2: apply recolor(grid, 1, 2)"
)
_H2 = (
    "2. Find the background color (most frequent) and replace every other cell with grey.\n"
    "   - step 1: count frequencies with np.bincount\n"
    "   - step 2: np.where to replace non-background cells"
)
_H3 = (
    "3. Tile the top-left quadrant four times to produce a doubled output grid.\n"
    "   - step 1: crop top-left quadrant\n"
    "   - step 2: tile(grid, 2, 2) to fill the output"
)
_HYP3 = f"{_H1}\n{_H2}\n{_H3}"


def _critic_coder(feedback: str = "fix the bug") -> dict:
    return {"route": "coder", "feedback": feedback}


def _critic_hyp(feedback: str = "logic flaw") -> dict:
    return {"route": "hypothesizer", "feedback": feedback}


def _mock_client_factory(backend: str = "ollama", model: str = "test-model"):
    """Return a side_effect callable that produces a fresh MagicMock per LLMClient call.

    Each call to LLMClient(...) returns a new mock whose .model is taken from
    the ``model`` kwarg passed to LLMClient, falling back to the provided default.
    This lets tests verify that per-role models are stored correctly.
    """
    def factory(**kwargs):
        mc = MagicMock()
        mc.backend = backend
        mc.model   = kwargs.get("model") or model
        return mc
    return factory


def make_orchestrator(n_hypotheses: int = 3, max_retries: int = 2) -> Orchestrator:
    """Create an Orchestrator with all three role sub-agents mocked out."""
    with patch("agents.orchestrator.LLMClient",
               side_effect=_mock_client_factory("ollama", "test-model")):
        orch = Orchestrator(
            backend="ollama",
            n_hypotheses=n_hypotheses,
            max_retries=max_retries,
        )
    orch._hypothesizer = MagicMock()
    orch._coder        = MagicMock()
    orch._critic       = MagicMock()
    return orch


# ---------------------------------------------------------------------------
# Orchestrator.__init__
# ---------------------------------------------------------------------------

class TestOrchestratorInit:
    def _make(self, **kwargs) -> Orchestrator:
        backend = kwargs.get("backend", "ollama")
        with patch("agents.orchestrator.LLMClient",
                   side_effect=_mock_client_factory(backend, "m")):
            return Orchestrator(**kwargs)

    def test_backend_stored(self):
        assert self._make(backend="ollama").backend == "ollama"

    def test_model_stored(self):
        """orch.model is an alias for the hypothesizer model."""
        orch = self._make()
        assert orch.model == "m"

    def test_hypothesizer_model_stored(self):
        orch = self._make(hypothesizer_model="deepseek-r1:32b")
        assert orch.hypothesizer_model == "deepseek-r1:32b"

    def test_coder_model_stored(self):
        orch = self._make(coder_model="qwen2.5-coder:7b")
        assert orch.coder_model == "qwen2.5-coder:7b"

    def test_critic_model_stored(self):
        orch = self._make(critic_model="deepseek-r1:8b")
        assert orch.critic_model == "deepseek-r1:8b"

    def test_per_role_models_are_independent(self):
        """Each role gets its own model; they don't bleed into one another."""
        orch = self._make(
            hypothesizer_model="deepseek-r1:32b",
            coder_model="qwen2.5-coder:7b",
            critic_model="deepseek-r1:8b",
        )
        assert orch.hypothesizer_model == "deepseek-r1:32b"
        assert orch.coder_model        == "qwen2.5-coder:7b"
        assert orch.critic_model       == "deepseek-r1:8b"
        assert orch.model              == "deepseek-r1:32b"  # alias → hypothesizer

    def test_fallback_model_used_when_role_not_specified(self):
        """When only ``model`` is given, all roles inherit it."""
        orch = self._make(model="qwen2.5:32b")
        assert orch.hypothesizer_model == "qwen2.5:32b"
        assert orch.coder_model        == "qwen2.5:32b"
        assert orch.critic_model       == "qwen2.5:32b"

    def test_max_retries_stored(self):
        assert self._make(max_retries=4).max_retries == 4

    def test_n_hypotheses_stored(self):
        assert self._make(n_hypotheses=5).n_hypotheses == 5

    def test_debug_stored(self):
        assert self._make(debug=True).debug is True

    def test_roles_are_real_instances(self):
        orch = self._make()
        assert isinstance(orch._hypothesizer, Hypothesizer)
        assert isinstance(orch._coder,        Coder)
        assert isinstance(orch._critic,       Critic)

    def test_hypothesizer_temperature_default(self):
        assert self._make().hypothesizer_temperature == pytest.approx(0.6)

    def test_coder_temperature_default(self):
        assert self._make().coder_temperature == pytest.approx(0.1)

    def test_critic_temperature_default(self):
        assert self._make().critic_temperature == pytest.approx(0.2)

    def test_temperature_overrides_stored(self):
        orch = self._make(
            hypothesizer_temperature=0.9,
            coder_temperature=0.0,
            critic_temperature=0.5,
        )
        assert orch.hypothesizer_temperature == pytest.approx(0.9)
        assert orch.coder_temperature        == pytest.approx(0.0)
        assert orch.critic_temperature       == pytest.approx(0.5)

    def test_max_tokens_defaults(self):
        orch = self._make()
        assert orch.hypothesizer_max_tokens == 32768
        assert orch.coder_max_tokens        == 8192
        assert orch.critic_max_tokens       == 16384

    def test_max_tokens_overrides_stored(self):
        orch = self._make(
            hypothesizer_max_tokens=16000,
            coder_max_tokens=4096,
            critic_max_tokens=8000,
        )
        assert orch.hypothesizer_max_tokens == 16000
        assert orch.coder_max_tokens        == 4096
        assert orch.critic_max_tokens       == 8000


# ---------------------------------------------------------------------------
# solve() — happy path
# ---------------------------------------------------------------------------

class TestSolveSuccess:
    def test_success_flag_set(self):
        orch = make_orchestrator()
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.return_value         = _correct_code()

        result = orch.solve(simple_task())
        assert result["success"] is True

    def test_code_returned(self):
        orch = make_orchestrator()
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.return_value         = _correct_code()

        result = orch.solve(simple_task())
        assert result["code"] is not None
        assert "transform" in result["code"]

    def test_candidate_saved(self):
        orch = make_orchestrator()
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.return_value         = _correct_code()

        result = orch.solve(simple_task())
        assert len(result["candidates"]) >= 1

    def test_candidate_has_required_keys(self):
        orch = make_orchestrator()
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.return_value         = _correct_code()

        result = orch.solve(simple_task())
        cand = result["candidates"][0]
        for key in ("code", "hypothesis", "hypothesis_index", "attempt", "n_correct", "n_total"):
            assert key in cand, f"Missing key: {key}"

    def test_result_has_all_keys(self):
        orch = make_orchestrator()
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.return_value         = _correct_code()

        result = orch.solve(simple_task())
        for key in ("success", "test_correct", "code", "candidates", "log"):
            assert key in result

    def test_test_correct_evaluated_when_ground_truth_present(self):
        orch = make_orchestrator()
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.return_value         = _correct_code()

        result = orch.solve(simple_task())
        assert result["test_correct"] is True

    def test_critic_not_called_on_success(self):
        orch = make_orchestrator()
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.return_value         = _correct_code()

        orch.solve(simple_task())
        orch._critic.analyze.assert_not_called()

    def test_multiple_hypotheses_each_produce_candidate(self):
        """When every hypothesis passes, all are saved as candidates."""
        orch = make_orchestrator(n_hypotheses=3)
        orch._hypothesizer.generate.return_value = _HYP3
        orch._coder.generate.return_value         = _correct_code()

        result = orch.solve(simple_task())
        # All 3 hypotheses were tried and all passed → 3 candidates
        assert len(result["candidates"]) == 3


# ---------------------------------------------------------------------------
# solve() — Critic routing
# ---------------------------------------------------------------------------

class TestCriticRouting:
    def test_coder_route_retries_same_hypothesis(self):
        orch = make_orchestrator(max_retries=1)
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        # hyp 0: attempt 1 wrong → retry → attempt 2 correct; hyps 1 & 2 also correct
        orch._coder.generate.side_effect = [
            _wrong_code(),   # hyp 0, attempt 1
            _correct_code(), # hyp 0, attempt 2 (retry)
            _correct_code(), # hyp 1
            _correct_code(), # hyp 2
        ]
        orch._critic.analyze.return_value = _critic_coder()

        result = orch.solve(simple_task())
        assert result["success"] is True
        # Critic was called (proves the retry path was exercised for hyp 0)
        assert orch._critic.analyze.call_count >= 1
        # At least 2 calls happened (initial + retry for hyp 0)
        assert orch._coder.generate.call_count >= 2

    def test_coder_feedback_forwarded_on_retry(self):
        """The Critic's feedback must be passed to the Coder on the retry call."""
        orch = make_orchestrator(max_retries=1)
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.side_effect = [_wrong_code(), _correct_code()]
        orch._critic.analyze.return_value = _critic_coder("USE_RECOLOR_PRIMITIVE")

        orch.solve(simple_task())

        # Second Coder call — feedback is the second positional argument
        second_call = orch._coder.generate.call_args_list[1]
        feedback_arg = second_call[0][1]   # generate(hypothesis, feedback)
        assert feedback_arg == "USE_RECOLOR_PRIMITIVE"

    def test_hypothesizer_route_abandons_hypothesis(self):
        """HYPOTHESIZER route must skip to the next hypothesis immediately."""
        orch = make_orchestrator(max_retries=2)
        orch._hypothesizer.generate.return_value = _HYP3
        # hyp 0: 1 wrong attempt → Critic → hypothesizer → abandon
        # hyps 1 & 2: correct on first attempt
        orch._coder.generate.side_effect = [
            _wrong_code(),   # hyp 0 (abandoned after Critic)
            _correct_code(), # hyp 1
            _correct_code(), # hyp 2
        ]
        orch._critic.analyze.return_value = _critic_hyp()

        result = orch.solve(simple_task())
        assert result["success"] is True
        # hyp 0 had 1 Coder call; hyps 1 & 2 had 1 each → total 3
        assert orch._coder.generate.call_count == 3

    def test_hypothesizer_route_logged_correctly(self):
        orch = make_orchestrator(max_retries=1)
        orch._hypothesizer.generate.return_value = "1. H1\n2. H2\n3. H3"
        orch._coder.generate.side_effect = [
            _wrong_code(),   # hyp 0 fails → Critic → hypothesizer
            _correct_code(), # hyp 1 succeeds
            _correct_code(), # hyp 2 succeeds
        ]
        orch._critic.analyze.return_value = _critic_hyp("Try tiling.")

        result = orch.solve(simple_task())
        critic_entries = [e for e in result["log"] if e.get("state") == STATE_CRITICIZING]
        assert any(e["route"] == "hypothesizer" for e in critic_entries)

    def test_max_retries_enforced(self):
        """Coder must not be called more than (max_retries + 1) times per hypothesis."""
        orch = make_orchestrator(max_retries=1)   # 2 total Coder calls per hypothesis
        orch._hypothesizer.generate.return_value = "1. H\n2. noop\n3. noop"
        # First hypothesis: both calls fail; Critic always says "coder"
        orch._coder.generate.return_value  = _wrong_code()
        orch._critic.analyze.return_value  = _critic_coder()

        orch.solve(simple_task())

        # Hypothesis 1: 2 calls (initial + 1 retry)
        # Hypotheses 2 & 3: 1 call each (wrong, Critic → coder, but max_retries=1 exhausted
        #   after 2nd attempt within hyp 1; other hyps start fresh)
        # Total: at most 2 + 2 + 2 = 6 calls (all fail, all get 1 retry each)
        assert orch._coder.generate.call_count <= (1 + 1) * 3   # (max_retries+1) * n_hyps

    def test_max_retries_zero_means_no_retry(self):
        """max_retries=0 → only one Coder attempt per hypothesis, no retry."""
        orch = make_orchestrator(max_retries=0)
        orch._hypothesizer.generate.return_value = _HYP3
        orch._coder.generate.return_value  = _wrong_code()
        orch._critic.analyze.return_value  = _critic_coder()

        orch.solve(simple_task())
        # 3 hypotheses × 1 attempt each = 3 total Coder calls
        assert orch._coder.generate.call_count == 3


# ---------------------------------------------------------------------------
# solve() — failure paths
# ---------------------------------------------------------------------------

class TestSolveFailure:
    def test_all_hypotheses_fail_returns_false(self):
        orch = make_orchestrator(max_retries=0)
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.return_value         = _wrong_code()
        orch._critic.analyze.return_value         = _critic_hyp()

        result = orch.solve(simple_task())
        assert result["success"] is False

    def test_best_partial_code_returned_on_failure(self):
        orch = make_orchestrator(max_retries=0)
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.return_value         = _wrong_code()
        orch._critic.analyze.return_value         = _critic_hyp()

        result = orch.solve(simple_task())
        # Even though no candidate passed all pairs, best code should be returned
        assert result["code"] is not None

    def test_empty_candidates_on_failure(self):
        orch = make_orchestrator(max_retries=0)
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.return_value         = _wrong_code()
        orch._critic.analyze.return_value         = _critic_hyp()

        result = orch.solve(simple_task())
        assert result["candidates"] == []

    def test_hypothesizer_exception_exits_early(self):
        orch = make_orchestrator()
        orch._hypothesizer.generate.side_effect = TimeoutError("timed out")

        result = orch.solve(simple_task())
        assert result["success"] is False
        orch._coder.generate.assert_not_called()

    def test_coder_exception_abandons_hypothesis(self):
        """A Coder exception should abandon the current hypothesis and try the next."""
        orch = make_orchestrator(max_retries=0)
        orch._hypothesizer.generate.return_value = _HYP3
        orch._coder.generate.side_effect = [
            TimeoutError("timed out"),  # hypothesis 0 fails
            _correct_code(),            # hypothesis 1 succeeds
            _correct_code(),
        ]

        result = orch.solve(simple_task())
        assert result["success"] is True

    def test_critic_exception_abandons_hypothesis(self):
        orch = make_orchestrator(max_retries=1)
        orch._hypothesizer.generate.return_value = _HYP3
        orch._coder.generate.side_effect = [
            _wrong_code(),      # hyp 0 fails → Critic raises → abandon hyp 0
            _correct_code(),    # hyp 1 succeeds
            _correct_code(),
        ]
        orch._critic.analyze.side_effect = RuntimeError("critic broke")

        result = orch.solve(simple_task())
        assert result["success"] is True

    def test_no_code_block_abandons_hypothesis(self):
        orch = make_orchestrator(max_retries=1)
        orch._hypothesizer.generate.return_value = _HYP3
        orch._coder.generate.side_effect = [
            "I don't know how to do this.",  # no code block → abandon hyp 0
            _correct_code(),                  # hyp 1 succeeds
            _correct_code(),
        ]

        result = orch.solve(simple_task())
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Log structure
# ---------------------------------------------------------------------------

class TestLog:
    def _states(self, log: list[dict]) -> list[str]:
        return [e["state"] for e in log]

    def test_hypothesizing_logged(self):
        orch = make_orchestrator()
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.return_value         = _correct_code()

        result = orch.solve(simple_task())
        assert STATE_HYPOTHESIZING in self._states(result["log"])

    def test_coding_logged(self):
        orch = make_orchestrator()
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.return_value         = _correct_code()

        result = orch.solve(simple_task())
        assert STATE_CODING in self._states(result["log"])

    def test_evaluating_logged(self):
        orch = make_orchestrator()
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.return_value         = _correct_code()

        result = orch.solve(simple_task())
        assert STATE_EVALUATING in self._states(result["log"])

    def test_candidate_logged_on_success(self):
        orch = make_orchestrator()
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.return_value         = _correct_code()

        result = orch.solve(simple_task())
        assert STATE_CANDIDATE in self._states(result["log"])

    def test_criticizing_logged_on_failure(self):
        # max_retries=1 → two attempts per hypothesis; Critic runs after attempt 1
        # (not the last), so its "coder" feedback is actually used.
        orch = make_orchestrator(max_retries=1)
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.side_effect = [
            _wrong_code(), _correct_code(),   # hyp 0: fail → critic → fix
            _correct_code(), _correct_code(),  # hyp 1 & 2 (not reached)
        ]
        orch._critic.analyze.return_value = _critic_coder()

        result = orch.solve(simple_task())
        assert STATE_CRITICIZING in self._states(result["log"])

    def test_evaluating_log_has_n_correct(self):
        orch = make_orchestrator()
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.return_value         = _correct_code()

        result = orch.solve(simple_task())
        eval_entries = [e for e in result["log"] if e.get("state") == STATE_EVALUATING]
        assert all("n_correct" in e for e in eval_entries)

    def test_criticizing_log_has_route(self):
        # Needs max_retries=1 so the Critic is called after a non-final attempt.
        orch = make_orchestrator(max_retries=1)
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.side_effect = [
            _wrong_code(), _correct_code(),
            _correct_code(), _correct_code(),
        ]
        orch._critic.analyze.return_value = _critic_coder("specific fix")

        result = orch.solve(simple_task())
        crit_entries = [e for e in result["log"] if e.get("state") == STATE_CRITICIZING]
        assert all("route" in e for e in crit_entries)

    def test_hypothesizing_error_logged_on_exception(self):
        orch = make_orchestrator()
        orch._hypothesizer.generate.side_effect = RuntimeError("boom")

        result = orch.solve(simple_task())
        hyp_entries = [e for e in result["log"] if e.get("state") == STATE_HYPOTHESIZING]
        assert any("error" in e for e in hyp_entries)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestPredict:
    def test_returns_ndarray_on_success(self):
        orch = make_orchestrator()
        orch._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        orch._coder.generate.return_value         = _correct_code()

        result = orch.predict(simple_task())
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_returns_none_when_no_code(self):
        orch = make_orchestrator()
        orch._hypothesizer.generate.side_effect = RuntimeError("fail")

        result = orch.predict(simple_task())
        assert result is None
