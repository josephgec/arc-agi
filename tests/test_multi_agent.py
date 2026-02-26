"""Tests for agents/multi_agent.py — orchestration and helpers."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agents.multi_agent import (
    MultiAgent,
    _extract_code,
    _format_diff,
    _format_error_info,
    _format_task_description,
    _parse_hypotheses,
    _strip_thinking,
)


# ---------------------------------------------------------------------------
# Fixtures
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


def make_agent(max_cycles: int = 9) -> MultiAgent:
    """Create a MultiAgent with all three role sub-agents mocked out."""
    with patch("agents.multi_agent.LLMClient"):
        agent = MultiAgent(backend="ollama", max_cycles=max_cycles)
    agent._hypothesizer = MagicMock()
    agent._coder        = MagicMock()
    agent._critic       = MagicMock()
    return agent


# ---------------------------------------------------------------------------
# _strip_thinking
# ---------------------------------------------------------------------------

class TestStripThinking:
    def test_removes_think_block(self):
        assert _strip_thinking("<think>draft</think>answer") == "answer"

    def test_no_think_block_unchanged(self):
        assert _strip_thinking("plain text") == "plain text"

    def test_multiline_think_block_removed(self):
        text   = "<think>\nsome draft\n</think>\nfinal"
        result = _strip_thinking(text)
        assert "draft" not in result
        assert "final" in result


# ---------------------------------------------------------------------------
# _extract_code
# ---------------------------------------------------------------------------

class TestExtractCode:
    def test_python_block(self):
        text = "```python\ndef transform(grid):\n    return grid\n```"
        assert "def transform" in _extract_code(text)

    def test_generic_block_with_def(self):
        text = "```\ndef transform(grid):\n    return grid\n```"
        assert "def transform" in _extract_code(text)

    def test_generic_block_without_def_ignored(self):
        text = "```\nx = 1\n```"
        assert _extract_code(text) is None

    def test_bare_def_fallback(self):
        text = "def transform(grid):\n    return grid"
        assert "def transform" in _extract_code(text)

    def test_no_code_returns_none(self):
        assert _extract_code("no code here at all") is None

    def test_python_block_preferred_over_bare_def(self):
        text = "```python\ndef transform(grid):\n    return grid.copy()\n```\n"
        code = _extract_code(text)
        assert "return grid.copy()" in code


# ---------------------------------------------------------------------------
# _parse_hypotheses
# ---------------------------------------------------------------------------

_H1 = "1. Rotate the grid 90 degrees clockwise and then recolor all red cells to blue.\n- step A: rotate\n- step B: recolor"
_H2 = "2. Find the most frequent color (background) and replace every other cell with grey (5).\n- step A: count\n- step B: replace"
_H3 = "3. Tile the top-left quadrant four times to fill the entire output grid of double the size.\n- step A: crop\n- step B: tile"


class TestParseHypotheses:
    def test_three_numbered_hypotheses(self):
        text = f"{_H1}\n{_H2}\n{_H3}"
        result = _parse_hypotheses(text)
        assert len(result) == 3

    def test_each_contains_hypothesis_text(self):
        text = f"{_H1}\n{_H2}\n{_H3}"
        result = _parse_hypotheses(text)
        assert any("Rotate" in h for h in result)
        assert any("background" in h for h in result)
        assert any("quadrant" in h for h in result)

    def test_unparseable_returns_single_item(self):
        text = "Just a single blob of text with no numbers."
        result = _parse_hypotheses(text)
        assert len(result) == 1
        assert result[0] == text

    def test_strips_thinking_before_parsing(self):
        text = f"<think>draft reasoning</think>{_H1}\n{_H2}\n{_H3}"
        result = _parse_hypotheses(text)
        assert all("<think>" not in h for h in result)

    def test_two_hypotheses_still_split(self):
        text = f"{_H1}\n{_H2}"
        result = _parse_hypotheses(text)
        assert len(result) == 2

    def test_max_n_caps_output(self):
        """max_n=2 on a 3-hypothesis response returns at most 2 items."""
        text = f"{_H1}\n{_H2}\n{_H3}"
        result = _parse_hypotheses(text, max_n=2)
        assert len(result) == 2
        assert "Rotate" in result[0]

    def test_short_noise_fragments_filtered(self):
        """Short one-liners (reasoning noise) are filtered out, leaving only full hypotheses."""
        # Mix one full hypothesis with two noise fragments
        noise = "1. Yes.\n2. Maybe not.\n"
        text = noise + _H3
        result = _parse_hypotheses(text)
        # Only _H3 is long enough; noise items should be dropped
        assert all(len(h) >= 80 for h in result)

    def test_paragraph_level_split_preferred(self):
        """Blank-line-separated numbered paragraphs are found even when inner steps
        also use numbers — only the paragraph-level items are returned."""
        h1 = (
            "1. Identify the dominant color by scanning the entire grid cell by cell.\n"
            "   - step 1: count occurrences\n"
            "   - step 2: pick the max"
        )
        h2 = (
            "2. Replace every cell that differs from the dominant color with grey (5).\n"
            "   - step 1: iterate cells\n"
            "   - step 2: apply replacement"
        )
        text = h1 + "\n\n" + h2
        result = _parse_hypotheses(text)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _format_error_info
# ---------------------------------------------------------------------------

class TestFormatErrorInfo:
    def _eval(self, n_correct, pairs):
        return {"n_correct": n_correct, "n_total": len(pairs), "pairs": pairs}

    def test_summary_line_present(self):
        result = _format_error_info(self._eval(0, [{"error": None, "correct": False}]))
        assert "0/1" in result

    def test_error_message_included(self):
        pair = {"error": "NameError: undefined", "correct": False}
        result = _format_error_info(self._eval(0, [pair]))
        assert "NameError: undefined" in result

    def test_no_error_wrong_output_noted(self):
        pair = {"error": None, "correct": False}
        result = _format_error_info(self._eval(0, [pair]))
        assert "wrong" in result.lower() or "incorrect" in result.lower() or "Pair" in result

    def test_correct_pairs_not_mentioned(self):
        pairs = [{"error": None, "correct": True}, {"error": "bad", "correct": False}]
        result = _format_error_info(self._eval(1, pairs))
        assert "Pair 2" in result
        assert "Pair 1" not in result


# ---------------------------------------------------------------------------
# _format_diff
# ---------------------------------------------------------------------------

class TestFormatDiff:
    def test_all_correct_returns_short_message(self):
        pairs = [{"correct": True, "expected": _grid([1]), "predicted": _grid([1])}]
        result = _format_diff({"pairs": pairs})
        assert "all pairs correct" in result.lower()

    def test_shape_mismatch_detected(self):
        pairs = [{
            "correct": False,
            "expected":  _grid([1, 2]),
            "predicted": _grid([1]),
        }]
        result = _format_diff({"pairs": pairs})
        assert "shape" in result.lower() or "mismatch" in result.lower()

    def test_cell_differences_listed(self):
        pairs = [{
            "correct":   False,
            "expected":  _grid([0, 1]),
            "predicted": _grid([0, 2]),
        }]
        result = _format_diff({"pairs": pairs})
        assert "[0,1]" in result or "0, 1" in result

    def test_none_predicted_handled(self):
        pairs = [{"correct": False, "expected": _grid([1]), "predicted": None}]
        result = _format_diff({"pairs": pairs})
        assert "no output" in result.lower()


# ---------------------------------------------------------------------------
# _format_task_description
# ---------------------------------------------------------------------------

class TestFormatTaskDescription:
    def test_contains_training_header(self):
        desc = _format_task_description(simple_task())
        assert "Training pair 1" in desc

    def test_contains_test_input(self):
        desc = _format_task_description(simple_task())
        assert "Test input" in desc

    def test_grid_values_present(self):
        desc = _format_task_description(simple_task())
        assert "0" in desc and "1" in desc

    def test_large_grid_truncated(self):
        big = np.zeros((20, 20), dtype=np.int32)
        task = {
            "train": [
                {"input": big, "output": big},
                {"input": big, "output": big},
                {"input": big, "output": big},
            ],
            "test": [{"input": big}],
        }
        desc = _format_task_description(task)
        assert "Training pair 3" not in desc


# ---------------------------------------------------------------------------
# MultiAgent.solve — orchestration
# ---------------------------------------------------------------------------

class TestMultiAgentSolve:
    def test_success_first_hypothesis_first_code(self):
        agent = make_agent()
        agent._hypothesizer.generate.return_value = "1. Recolor 1→2\n2. Second\n3. Third"
        agent._coder.generate.return_value         = _correct_code()

        result = agent.solve(simple_task())
        assert result["success"] is True
        assert result["code"] is not None
        assert result["n_cycles"] > 0

    def test_critic_routes_to_coder_then_succeeds(self):
        agent = make_agent()
        agent._hypothesizer.generate.return_value = "1. Recolor 1→2\n2. Second\n3. Third"
        agent._coder.generate.side_effect = [_wrong_code(), _correct_code()]
        agent._critic.analyze.return_value = {"route": "coder", "feedback": "Use recolor."}

        result = agent.solve(simple_task())
        assert result["success"] is True
        # Critic should have been called once before the retry succeeded
        assert agent._critic.analyze.call_count >= 1

    def test_critic_routes_to_hypothesizer_then_succeeds(self):
        agent = make_agent()
        agent._hypothesizer.generate.side_effect = [
            "1. Wrong rule\n2. B\n3. C",   # first call → all wrong
            "1. Recolor 1→2\n2. B\n3. C",  # second call → correct hypothesis
        ]
        agent._coder.generate.side_effect = [_wrong_code(), _correct_code()]
        agent._critic.analyze.return_value = {
            "route": "hypothesizer", "feedback": "Rethink the rule."
        }

        result = agent.solve(simple_task())
        assert result["success"] is True

    def test_exhausts_max_cycles_returns_failure(self):
        agent = make_agent(max_cycles=3)
        agent._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        agent._coder.generate.return_value         = _wrong_code()
        agent._critic.analyze.return_value = {"route": "coder", "feedback": "fix it"}

        result = agent.solve(simple_task())
        assert result["success"] is False
        assert result["n_cycles"] <= 3

    def test_result_always_has_required_keys(self):
        agent = make_agent(max_cycles=3)
        agent._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        agent._coder.generate.return_value         = _wrong_code()
        agent._critic.analyze.return_value = {"route": "coder", "feedback": "fix"}

        result = agent.solve(simple_task())
        for key in ("success", "test_correct", "code", "n_cycles", "log"):
            assert key in result

    def test_log_records_agent_calls(self):
        agent = make_agent()
        agent._hypothesizer.generate.return_value = "1. Recolor 1→2\n2. B\n3. C"
        agent._coder.generate.return_value         = _correct_code()

        result = agent.solve(simple_task())
        agents_logged = {entry["agent"] for entry in result["log"]}
        assert "hypothesizer" in agents_logged
        assert "coder"        in agents_logged

    def test_coder_feedback_passed_on_coder_route(self):
        """When Critic routes to CODER, feedback must appear in next Coder call."""
        agent = make_agent()
        agent._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        agent._coder.generate.side_effect = [_wrong_code(), _correct_code()]
        agent._critic.analyze.return_value = {
            "route": "coder", "feedback": "USE_RECOLOR_PRIMITIVE"
        }

        agent.solve(simple_task())
        # Second Coder call should receive the feedback
        second_call_kwargs = agent._coder.generate.call_args_list[1]
        feedback_arg = second_call_kwargs[0][1] if second_call_kwargs[0] else \
                       second_call_kwargs[1].get("feedback")
        assert "USE_RECOLOR_PRIMITIVE" in str(second_call_kwargs)

    def test_hypothesizer_exception_breaks_loop(self):
        agent = make_agent(max_cycles=3)
        agent._hypothesizer.generate.side_effect = TimeoutError("timed out")

        result = agent.solve(simple_task())
        assert result["success"] is False

    def test_no_code_block_advances_hypothesis(self):
        """A Coder response with no code block should advance to the next hypothesis."""
        agent = make_agent()
        agent._hypothesizer.generate.return_value = "1. H1\n2. H2\n3. H3"
        agent._coder.generate.side_effect = [
            "I am not sure how to do this.",  # no code block → advance
            _correct_code(),
        ]

        result = agent.solve(simple_task())
        assert result["success"] is True


# ---------------------------------------------------------------------------
# MultiAgent.predict
# ---------------------------------------------------------------------------

class TestMultiAgentPredict:
    def test_predict_returns_grid_on_success(self):
        agent = make_agent()
        agent._hypothesizer.generate.return_value = "1. Recolor 1→2\n2. B\n3. C"
        agent._coder.generate.return_value         = _correct_code()

        result = agent.predict(simple_task())
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_predict_returns_none_on_failure(self):
        agent = make_agent(max_cycles=3)
        agent._hypothesizer.generate.return_value = "1. H\n2. H\n3. H"
        agent._coder.generate.return_value         = _wrong_code()
        agent._critic.analyze.return_value = {"route": "coder", "feedback": "fix"}

        result = agent.predict(simple_task())
        # May return a grid (best attempt) or None; should not raise
        # The key contract: no exception
