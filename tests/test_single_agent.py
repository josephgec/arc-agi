"""Tests for agents/single_agent.py (Anthropic API is mocked throughout)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agents.single_agent import SingleAgent, _grid_to_str, _diff_summary, _strip_thinking


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent() -> SingleAgent:
    """Return a SingleAgent backed by mocked Anthropic client."""
    with patch("anthropic.Anthropic"):
        agent = SingleAgent(backend="anthropic", model="claude-sonnet-4-6", max_retries=2)
    agent.client = MagicMock()
    return agent


def _mock_response(text: str) -> MagicMock:
    """Simulate an Anthropic API response."""
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


# ---------------------------------------------------------------------------
# _strip_thinking
# ---------------------------------------------------------------------------

class TestStripThinking:
    def test_removes_think_block(self):
        text = "<think>\nsome internal reasoning\n</think>\nActual answer"
        assert _strip_thinking(text) == "Actual answer"

    def test_no_think_block_unchanged(self):
        text = "Normal response without thinking."
        assert _strip_thinking(text) == text

    def test_multiline_think_block(self):
        text = "<think>\nline1\nline2\nline3\n</think>\n```python\ndef transform(g): return g\n```"
        result = _strip_thinking(text)
        assert "<think>" not in result
        assert "def transform" in result

    def test_empty_think_block(self):
        text = "<think></think>answer"
        assert _strip_thinking(text) == "answer"


# ---------------------------------------------------------------------------
# SingleAgent init
# ---------------------------------------------------------------------------

class TestSingleAgentInit:
    def test_ollama_backend(self):
        with patch("openai.OpenAI"):
            agent = SingleAgent(backend="ollama")
        assert agent.backend == "ollama"
        assert "deepseek" in agent.model

    def test_anthropic_backend(self):
        with patch("anthropic.Anthropic"):
            agent = SingleAgent(backend="anthropic")
        assert agent.backend == "anthropic"
        assert "claude" in agent.model

    def test_custom_model_overrides_default(self):
        with patch("openai.OpenAI"):
            agent = SingleAgent(backend="ollama", model="qwen2.5:32b")
        assert agent.model == "qwen2.5:32b"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            SingleAgent(backend="invalid")


# ---------------------------------------------------------------------------
# _grid_to_str
# ---------------------------------------------------------------------------

class TestGridToStr:
    def test_single_row(self):
        g = np.array([[1, 2, 3]], dtype=np.int32)
        s = _grid_to_str(g)
        assert "1" in s and "2" in s and "3" in s

    def test_multirow(self):
        g = np.array([[1, 0], [0, 1]], dtype=np.int32)
        s = _grid_to_str(g)
        assert s.count("[") >= 2  # at least 2 inner rows

    def test_is_string(self):
        g = np.zeros((2, 2), dtype=np.int32)
        assert isinstance(_grid_to_str(g), str)


# ---------------------------------------------------------------------------
# _diff_summary
# ---------------------------------------------------------------------------

class TestDiffSummary:
    def test_identical_grids(self):
        g = np.array([[1, 2]], dtype=np.int32)
        assert "no differences" in _diff_summary(g, g.copy())

    def test_shape_mismatch(self):
        a = np.array([[1, 2]], dtype=np.int32)
        b = np.array([[1]], dtype=np.int32)
        summary = _diff_summary(a, b)
        assert "Shape mismatch" in summary

    def test_lists_differing_cells(self):
        a = np.array([[1, 2]], dtype=np.int32)
        b = np.array([[1, 3]], dtype=np.int32)
        summary = _diff_summary(a, b)
        assert "(0,1)" in summary

    def test_truncates_at_max_show(self):
        a = np.zeros((5, 5), dtype=np.int32)
        b = np.ones((5, 5), dtype=np.int32)
        summary = _diff_summary(a, b, max_show=3)
        assert "more" in summary


# ---------------------------------------------------------------------------
# _format_task_prompt
# ---------------------------------------------------------------------------

class TestFormatTaskPrompt:
    def test_contains_training_pair_header(self, simple_task):
        agent = make_agent()
        prompt = agent._format_task_prompt(simple_task)
        assert "Training pair 1" in prompt

    def test_contains_test_input(self, simple_task):
        agent = make_agent()
        prompt = agent._format_task_prompt(simple_task)
        assert "Test input" in prompt

    def test_contains_grid_values(self, simple_task):
        agent = make_agent()
        prompt = agent._format_task_prompt(simple_task)
        assert "1" in prompt  # a value that appears in the grids

    def test_is_string(self, simple_task):
        agent = make_agent()
        assert isinstance(agent._format_task_prompt(simple_task), str)

    def test_multiple_training_pairs_shown(self, simple_task):
        agent = make_agent()
        prompt = agent._format_task_prompt(simple_task)
        assert "Training pair 2" in prompt


# ---------------------------------------------------------------------------
# _extract_code
# ---------------------------------------------------------------------------

class TestExtractCode:
    def test_extracts_from_fenced_block(self):
        agent = make_agent()
        text = "Some reasoning.\n```python\ndef transform(grid):\n    return grid\n```"
        code = agent._extract_code(text)
        assert code is not None
        assert "def transform" in code

    def test_extracts_bare_def(self):
        agent = make_agent()
        text = "Here you go:\ndef transform(grid):\n    return grid\n"
        code = agent._extract_code(text)
        assert code is not None
        assert "def transform" in code

    def test_returns_none_when_no_code(self):
        agent = make_agent()
        assert agent._extract_code("No code here.") is None

    def test_strips_whitespace(self):
        agent = make_agent()
        text = "```python\n\ndef transform(grid):\n    return grid\n\n```"
        code = agent._extract_code(text)
        assert not code.startswith("\n")


# ---------------------------------------------------------------------------
# _execute
# ---------------------------------------------------------------------------

class TestExecute:
    def _g(self):
        return np.array([[1, 2], [3, 4]], dtype=np.int32)

    def test_valid_code_runs(self):
        agent = make_agent()
        code = "def transform(grid):\n    return grid.copy()"
        output, error = agent._execute(code, self._g())
        assert error is None
        assert output is not None
        np.testing.assert_array_equal(output, self._g())

    def test_compile_error_caught(self):
        agent = make_agent()
        code = "def transform(grid):\n    return grid +"  # syntax error
        output, error = agent._execute(code, self._g())
        assert output is None
        assert "error" in error.lower()

    def test_runtime_error_caught(self):
        agent = make_agent()
        code = "def transform(grid):\n    raise RuntimeError('oops')"
        output, error = agent._execute(code, self._g())
        assert output is None
        assert "oops" in error

    def test_missing_transform_fn(self):
        agent = make_agent()
        code = "x = 1"  # no transform defined
        output, error = agent._execute(code, self._g())
        assert output is None
        assert error is not None

    def test_output_is_int32(self):
        agent = make_agent()
        code = "def transform(grid):\n    return np.zeros_like(grid)"
        output, error = agent._execute(code, self._g())
        assert output.dtype == np.int32

    def test_list_output_converted(self):
        agent = make_agent()
        code = "def transform(grid):\n    return [[0, 0], [0, 0]]"
        output, error = agent._execute(code, self._g())
        assert error is None
        assert isinstance(output, np.ndarray)

    def test_dsl_available(self):
        agent = make_agent()
        code = "def transform(grid):\n    return recolor(grid, 1, 9)"
        g = np.array([[1, 0]], dtype=np.int32)
        output, error = agent._execute(code, g)
        assert error is None
        assert output[0, 0] == 9

    def test_numpy_available_as_np(self):
        agent = make_agent()
        code = "def transform(grid):\n    return np.zeros((2, 2), dtype=np.int32)"
        output, error = agent._execute(code, self._g())
        assert error is None
        assert output.shape == (2, 2)

    def test_input_not_mutated(self):
        agent = make_agent()
        code = "def transform(grid):\n    grid[0, 0] = 99\n    return grid"
        original = self._g()
        original_val = int(original[0, 0])
        agent._execute(code, original)
        assert original[0, 0] == original_val


# ---------------------------------------------------------------------------
# _evaluate_code
# ---------------------------------------------------------------------------

class TestEvaluateCode:
    def test_all_correct(self, simple_task):
        agent = make_agent()
        code = "def transform(grid):\n    return recolor(grid, 1, 2)"
        result = agent._evaluate_code(code, simple_task)
        assert result["all_correct"]

    def test_all_wrong(self, simple_task):
        agent = make_agent()
        code = "def transform(grid):\n    return grid.copy()"
        result = agent._evaluate_code(code, simple_task)
        assert result["n_correct"] == 0

    def test_error_captured_in_pair(self, simple_task):
        agent = make_agent()
        code = "def transform(grid):\n    raise ValueError('x')"
        result = agent._evaluate_code(code, simple_task)
        for pair in result["pairs"]:
            assert pair["error"] is not None

    def test_result_keys(self, simple_task):
        agent = make_agent()
        code = "def transform(grid):\n    return grid"
        result = agent._evaluate_code(code, simple_task)
        assert all(k in result for k in ["pairs", "n_correct", "n_total", "all_correct"])


# ---------------------------------------------------------------------------
# _format_error_feedback
# ---------------------------------------------------------------------------

class TestFormatErrorFeedback:
    def test_mentions_wrong_pairs(self, simple_task):
        agent = make_agent()
        code = "def transform(grid):\n    return grid"
        eval_result = agent._evaluate_code(code, simple_task)
        feedback = agent._format_error_feedback(eval_result, attempt=1)
        assert "WRONG" in feedback

    def test_mentions_correct_pairs(self):
        agent = make_agent()
        task = {
            "train": [
                {
                    "input": np.array([[1]], dtype=np.int32),
                    "output": np.array([[1]], dtype=np.int32),
                },
                {
                    "input": np.array([[0]], dtype=np.int32),
                    "output": np.array([[1]], dtype=np.int32),
                },
            ],
            "test": [],
        }
        eval_result = agent._evaluate_code("def transform(grid):\n    return grid.copy()", task)
        feedback = agent._format_error_feedback(eval_result, attempt=1)
        assert "CORRECT" in feedback

    def test_includes_revise_instruction(self, simple_task):
        agent = make_agent()
        eval_result = agent._evaluate_code("def transform(grid):\n    return grid", simple_task)
        feedback = agent._format_error_feedback(eval_result, attempt=1)
        assert "revise" in feedback.lower() or "correct" in feedback.lower()


# ---------------------------------------------------------------------------
# solve — end-to-end with mocked Claude
# ---------------------------------------------------------------------------

class TestSolve:
    def _correct_code(self):
        return "```python\ndef transform(grid):\n    return recolor(grid, 1, 2)\n```"

    def _wrong_code(self):
        return "```python\ndef transform(grid):\n    return grid.copy()\n```"

    def test_solve_succeeds_on_first_try(self, simple_task):
        agent = make_agent()
        agent.client.messages.create.return_value = _mock_response(self._correct_code())
        result = agent.solve(simple_task)
        assert result["success"]
        assert result["n_attempts"] == 1

    def test_solve_retries_and_succeeds(self, simple_task):
        agent = make_agent()
        responses = [
            _mock_response(self._wrong_code()),
            _mock_response(self._wrong_code()),
            _mock_response(self._correct_code()),
        ]
        agent.client.messages.create.side_effect = responses
        result = agent.solve(simple_task)
        assert result["success"]
        assert result["n_attempts"] == 3

    def test_solve_fails_after_max_retries(self, simple_task):
        agent = make_agent()
        agent.client.messages.create.return_value = _mock_response(self._wrong_code())
        result = agent.solve(simple_task)
        assert not result["success"]

    def test_solve_handles_missing_code_block(self, simple_task):
        agent = make_agent()
        responses = [
            _mock_response("I don't know"),
            _mock_response(self._correct_code()),
        ]
        agent.client.messages.create.side_effect = responses
        result = agent.solve(simple_task)
        assert result["success"]

    def test_solve_returns_log(self, simple_task):
        agent = make_agent()
        agent.client.messages.create.return_value = _mock_response(self._correct_code())
        result = agent.solve(simple_task)
        assert isinstance(result["log"], list)
        assert len(result["log"]) >= 1

    def test_solve_returns_code(self, simple_task):
        agent = make_agent()
        agent.client.messages.create.return_value = _mock_response(self._correct_code())
        result = agent.solve(simple_task)
        assert result["code"] is not None
        assert "def transform" in result["code"]

    def test_predict_returns_grid(self, simple_task):
        agent = make_agent()
        agent.client.messages.create.return_value = _mock_response(self._correct_code())
        output = agent.predict(simple_task)
        assert output is not None
        assert isinstance(output, np.ndarray)

    def test_predict_returns_best_guess_on_failure(self, simple_task):
        """predict() returns the best-effort output even when solve() fails."""
        agent = make_agent()
        agent.client.messages.create.return_value = _mock_response(self._wrong_code())
        output = agent.predict(simple_task)
        # Wrong answer but still returns a grid (best guess)
        assert isinstance(output, np.ndarray)

    def test_predict_returns_none_when_code_crashes(self, simple_task):
        agent = make_agent()
        crash_code = "```python\ndef transform(grid):\n    raise RuntimeError('crash')\n```"
        agent.client.messages.create.return_value = _mock_response(crash_code)
        output = agent.predict(simple_task)
        assert output is None
