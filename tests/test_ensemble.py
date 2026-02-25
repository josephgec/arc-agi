"""Tests for agents/ensemble.py — majority-voting ensemble."""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from agents.ensemble import Ensemble, _grids_equal, _majority_vote, _vote_summary


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _grid(*rows):
    return np.array(rows, dtype=np.int32)


def simple_task():
    """Minimal ARC task: recolor 1 → 2 (test output present for accuracy checks)."""
    inp = _grid([0, 1], [1, 0])
    out = _grid([0, 2], [2, 0])
    return {
        "train": [{"input": inp, "output": out}],
        "test":  [{"input": inp, "output": out}],
    }


# Three distinct codes that each produce recolor(1→2) — used for majority tests.
_CODE_A = "def transform(grid):\n    return recolor(grid, 1, 2)\n"
_CODE_B = "def transform(grid):\n    out = recolor(grid, 1, 2)\n    return out\n"
_CODE_C = "def transform(grid):\n    g = grid.copy()\n    return recolor(g, 1, 2)\n"
# A code that produces a different (wrong) output: recolor(1→3)
_CODE_WRONG = "def transform(grid):\n    return recolor(grid, 1, 3)\n"


def _candidate(code: str, hyp_index: int = 0, attempt: int = 1) -> dict:
    """Build a minimal candidate dict as the Orchestrator would produce."""
    return {
        "code":             code,
        "hypothesis":       "some rule",
        "hypothesis_index": hyp_index,
        "attempt":          attempt,
        "n_correct":        1,
        "n_total":          1,
    }


def _orch_result(candidates: list[dict]) -> dict:
    """Build a minimal Orchestrator result dict."""
    return {
        "success":      len(candidates) > 0,
        "test_correct": None,
        "code":         candidates[0]["code"] if candidates else None,
        "candidates":   candidates,
        "log":          [],
    }


def make_ensemble(target_candidates: int = 3, max_runs: int = 5) -> Ensemble:
    """Create an Ensemble with its Orchestrator replaced by a MagicMock."""
    mock_client         = MagicMock()
    mock_client.backend = "ollama"
    mock_client.model   = "test-model"
    with patch("agents.ensemble.Orchestrator") as MockOrch:
        mock_orch_instance         = MagicMock()
        mock_orch_instance.backend = "ollama"
        mock_orch_instance.model   = "test-model"
        MockOrch.return_value      = mock_orch_instance
        ens = Ensemble(target_candidates=target_candidates, max_runs=max_runs)
    ens._orchestrator = MagicMock()
    return ens


# ---------------------------------------------------------------------------
# _grids_equal
# ---------------------------------------------------------------------------

class TestGridsEqual:
    def test_identical_grids(self):
        g = _grid([1, 2], [3, 4])
        assert _grids_equal(g, g.copy()) is True

    def test_different_values(self):
        assert _grids_equal(_grid([1, 2]), _grid([1, 3])) is False

    def test_different_shapes(self):
        assert _grids_equal(_grid([1, 2]), _grid([1], [2])) is False

    def test_zeros(self):
        assert _grids_equal(_grid([0, 0]), _grid([0, 0])) is True


# ---------------------------------------------------------------------------
# _majority_vote
# ---------------------------------------------------------------------------

class TestMajorityVote:
    def test_empty_returns_none(self):
        assert _majority_vote([]) is None

    def test_single_grid_returned(self):
        g = _grid([1, 2], [3, 4])
        result = _majority_vote([g])
        assert _grids_equal(result, g)

    def test_unanimous_returned(self):
        g = _grid([0, 1])
        result = _majority_vote([g.copy(), g.copy(), g.copy()])
        assert _grids_equal(result, g)

    def test_clear_majority_wins(self):
        """3 votes for X, 1 vote for Y → X is returned."""
        x = _grid([0, 2], [2, 0])
        y = _grid([0, 3], [3, 0])
        result = _majority_vote([x, x, x, y])
        assert _grids_equal(result, x)

    def test_minority_not_returned(self):
        x = _grid([0, 2], [2, 0])
        y = _grid([0, 3], [3, 0])
        result = _majority_vote([x, x, x, y])
        assert not _grids_equal(result, y)

    def test_tie_returns_first_group(self):
        """Equal vote counts → the first group formed wins."""
        a = _grid([1, 0])
        b = _grid([0, 1])
        result = _majority_vote([a, b, a, b])   # tie: a appeared first
        assert _grids_equal(result, a)

    def test_all_different_returns_first(self):
        grids = [_grid([i]) for i in range(4)]
        result = _majority_vote(grids)
        assert _grids_equal(result, grids[0])

    def test_shape_mismatch_treated_as_different(self):
        a = _grid([1, 2])          # shape (1, 2)
        b = _grid([1], [2])        # shape (2, 1)
        result = _majority_vote([a, b, a])   # a wins 2 vs 1
        assert _grids_equal(result, a)


# ---------------------------------------------------------------------------
# _vote_summary
# ---------------------------------------------------------------------------

class TestVoteSummary:
    def test_empty_returns_empty(self):
        assert _vote_summary([]) == []

    def test_sorted_by_count_descending(self):
        x = _grid([0, 2], [2, 0])
        y = _grid([0, 3], [3, 0])
        summary = _vote_summary([x, y, x, x])
        assert summary[0]["count"] == 3
        assert summary[1]["count"] == 1

    def test_group_count_correct(self):
        x = _grid([1])
        summary = _vote_summary([x, x, x])
        assert summary[0]["count"] == 3

    def test_candidate_indices_recorded(self):
        x = _grid([1])
        y = _grid([2])
        summary = _vote_summary([x, y, x])   # x at 0,2 ; y at 1
        # x should be first (2 votes)
        assert sorted(summary[0]["candidate_indices"]) == [0, 2]
        assert summary[1]["candidate_indices"] == [1]

    def test_output_grid_present_in_group(self):
        x = _grid([7, 7])
        summary = _vote_summary([x])
        assert _grids_equal(summary[0]["output"], x)

    def test_single_grid_one_group(self):
        assert len(_vote_summary([_grid([1])])) == 1


# ---------------------------------------------------------------------------
# Ensemble.__init__
# ---------------------------------------------------------------------------

class TestEnsembleInit:
    def _make(self, **kwargs) -> Ensemble:
        mock_orch         = MagicMock()
        mock_orch.backend = kwargs.get("backend", "ollama")
        mock_orch.model   = "m"
        with patch("agents.ensemble.Orchestrator", return_value=mock_orch):
            return Ensemble(**kwargs)

    def test_target_candidates_stored(self):
        assert self._make(target_candidates=5).target_candidates == 5

    def test_max_runs_stored(self):
        assert self._make(max_runs=10).max_runs == 10

    def test_backend_forwarded(self):
        assert self._make(backend="ollama").backend == "ollama"

    def test_orchestrator_created(self):
        with patch("agents.ensemble.Orchestrator") as MockOrch:
            mock_orch         = MagicMock()
            mock_orch.backend = "ollama"
            mock_orch.model   = "m"
            MockOrch.return_value = mock_orch
            Ensemble(backend="ollama", n_hypotheses=5, max_retries=1)
        MockOrch.assert_called_once()
        _, call_kwargs = MockOrch.call_args
        assert call_kwargs["n_hypotheses"] == 5
        assert call_kwargs["max_retries"]  == 1


# ---------------------------------------------------------------------------
# Ensemble.solve — collection behaviour
# ---------------------------------------------------------------------------

class TestEnsembleSolveCollection:
    def test_stops_when_target_reached(self):
        """Orchestrator should not be called again once target_candidates is met."""
        ens = make_ensemble(target_candidates=2, max_runs=10)
        # Single run yields 3 unique candidates → target reached after run 1
        ens._orchestrator.solve.return_value = _orch_result([
            _candidate(_CODE_A),
            _candidate(_CODE_B),
            _candidate(_CODE_C),
        ])

        ens.solve(simple_task())
        assert ens._orchestrator.solve.call_count == 1

    def test_runs_again_when_not_enough_candidates(self):
        """If run 1 yields 1 candidate and target is 3, must run again."""
        ens = make_ensemble(target_candidates=3, max_runs=5)
        ens._orchestrator.solve.side_effect = [
            _orch_result([_candidate(_CODE_A)]),          # run 1: 1 candidate
            _orch_result([_candidate(_CODE_B), _candidate(_CODE_C)]),  # run 2: 2 more
        ]

        ens.solve(simple_task())
        assert ens._orchestrator.solve.call_count == 2

    def test_respects_max_runs_hard_cap(self):
        """Never exceed max_runs even if target_candidates is not met."""
        ens = make_ensemble(target_candidates=10, max_runs=3)
        ens._orchestrator.solve.return_value = _orch_result([])   # always empty

        ens.solve(simple_task())
        assert ens._orchestrator.solve.call_count == 3

    def test_deduplicates_by_code_text(self):
        """Same code from two different runs must only be counted once."""
        ens = make_ensemble(target_candidates=5, max_runs=3)
        ens._orchestrator.solve.return_value = _orch_result([
            _candidate(_CODE_A),   # same code every run
        ])

        result = ens.solve(simple_task())
        # 3 runs × same code = 1 unique candidate (deduped)
        assert len(result["candidates"]) == 1

    def test_different_codes_across_runs_all_kept(self):
        ens = make_ensemble(target_candidates=5, max_runs=3)
        ens._orchestrator.solve.side_effect = [
            _orch_result([_candidate(_CODE_A)]),
            _orch_result([_candidate(_CODE_B)]),
            _orch_result([_candidate(_CODE_C)]),
        ]

        result = ens.solve(simple_task())
        assert len(result["candidates"]) == 3

    def test_n_runs_reflects_actual_call_count(self):
        ens = make_ensemble(target_candidates=10, max_runs=2)
        ens._orchestrator.solve.return_value = _orch_result([])

        result = ens.solve(simple_task())
        assert result["n_runs"] == 2

    def test_n_runs_stops_early_when_target_reached(self):
        ens = make_ensemble(target_candidates=1, max_runs=5)
        ens._orchestrator.solve.return_value = _orch_result([_candidate(_CODE_A)])

        result = ens.solve(simple_task())
        assert result["n_runs"] == 1


# ---------------------------------------------------------------------------
# Ensemble.solve — voting behaviour
# ---------------------------------------------------------------------------

class TestEnsembleSolveVoting:
    def test_majority_output_selected(self):
        """3 programs outputting grid X beat 1 outputting grid Y."""
        ens = make_ensemble(target_candidates=4, max_runs=1)
        ens._orchestrator.solve.return_value = _orch_result([
            _candidate(_CODE_A),       # recolor 1→2 (correct)
            _candidate(_CODE_B),       # recolor 1→2 (correct)
            _candidate(_CODE_C),       # recolor 1→2 (correct)
            _candidate(_CODE_WRONG),   # recolor 1→3 (wrong)
        ])

        result = ens.solve(simple_task())
        assert result["prediction"] is not None
        # Majority predicts recolor 1→2 → matches the task's expected output
        assert result["test_correct"] is True

    def test_minority_output_not_selected(self):
        ens = make_ensemble(target_candidates=4, max_runs=1)
        ens._orchestrator.solve.return_value = _orch_result([
            _candidate(_CODE_A),
            _candidate(_CODE_B),
            _candidate(_CODE_C),
            _candidate(_CODE_WRONG),
        ])

        result = ens.solve(simple_task())
        # The minority (1→3) should not have won
        wrong_out = np.array([[0, 3], [3, 0]], dtype=np.int32)
        assert not np.array_equal(result["prediction"], wrong_out)

    def test_vote_summary_sorted_descending(self):
        ens = make_ensemble(target_candidates=4, max_runs=1)
        ens._orchestrator.solve.return_value = _orch_result([
            _candidate(_CODE_A),
            _candidate(_CODE_B),
            _candidate(_CODE_C),
            _candidate(_CODE_WRONG),
        ])

        result = ens.solve(simple_task())
        summary = result["vote_summary"]
        assert summary[0]["count"] >= summary[-1]["count"]

    def test_vote_summary_has_output_and_count(self):
        ens = make_ensemble(target_candidates=2, max_runs=1)
        ens._orchestrator.solve.return_value = _orch_result([
            _candidate(_CODE_A),
            _candidate(_CODE_B),
        ])

        result = ens.solve(simple_task())
        for group in result["vote_summary"]:
            assert "output"            in group
            assert "count"             in group
            assert "candidate_indices" in group

    def test_test_correct_true_when_majority_correct(self):
        ens = make_ensemble(target_candidates=2, max_runs=1)
        ens._orchestrator.solve.return_value = _orch_result([
            _candidate(_CODE_A),
            _candidate(_CODE_B),
        ])

        result = ens.solve(simple_task())
        assert result["test_correct"] is True

    def test_test_correct_none_when_no_ground_truth(self):
        ens = make_ensemble(target_candidates=1, max_runs=1)
        ens._orchestrator.solve.return_value = _orch_result([_candidate(_CODE_A)])

        task_no_gt = {
            "train": simple_task()["train"],
            "test":  [{"input": simple_task()["test"][0]["input"]}],   # no "output" key
        }
        result = ens.solve(task_no_gt)
        assert result["test_correct"] is None

    def test_failed_execution_excluded_from_vote(self):
        """A candidate whose code raises on the test input should not be voted."""
        ens = make_ensemble(target_candidates=2, max_runs=1)
        crash_code = "def transform(grid):\n    raise RuntimeError('boom')\n"
        ens._orchestrator.solve.return_value = _orch_result([
            _candidate(_CODE_A),
            _candidate(crash_code),
        ])

        result = ens.solve(simple_task())
        # vote_summary total count = 1 (only _CODE_A succeeded)
        total_votes = sum(g["count"] for g in result["vote_summary"])
        assert total_votes == 1


# ---------------------------------------------------------------------------
# Ensemble.solve — failure paths
# ---------------------------------------------------------------------------

class TestEnsembleSolveFailure:
    def test_no_candidates_returns_success_false(self):
        ens = make_ensemble(target_candidates=3, max_runs=3)
        ens._orchestrator.solve.return_value = _orch_result([])

        result = ens.solve(simple_task())
        assert result["success"] is False

    def test_no_candidates_prediction_is_none(self):
        ens = make_ensemble(target_candidates=3, max_runs=3)
        ens._orchestrator.solve.return_value = _orch_result([])

        result = ens.solve(simple_task())
        assert result["prediction"] is None

    def test_result_has_all_required_keys(self):
        ens = make_ensemble(target_candidates=3, max_runs=1)
        ens._orchestrator.solve.return_value = _orch_result([])

        result = ens.solve(simple_task())
        for key in ("success", "prediction", "test_correct", "candidates",
                    "vote_summary", "n_runs"):
            assert key in result


# ---------------------------------------------------------------------------
# Ensemble.predict
# ---------------------------------------------------------------------------

class TestEnsemblePredict:
    def test_returns_ndarray_on_success(self):
        ens = make_ensemble(target_candidates=1, max_runs=1)
        ens._orchestrator.solve.return_value = _orch_result([_candidate(_CODE_A)])

        result = ens.predict(simple_task())
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_returns_none_on_failure(self):
        ens = make_ensemble(target_candidates=3, max_runs=1)
        ens._orchestrator.solve.return_value = _orch_result([])

        assert ens.predict(simple_task()) is None
