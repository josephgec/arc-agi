"""Tests for agents/single_agent.py (Anthropic API is mocked throughout)."""
from __future__ import annotations

import json
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
        agent = SingleAgent(backend="ollama")
        assert agent.backend == "ollama"
        assert "deepseek" in agent.model

    def test_ollama_client_is_none(self):
        """Ollama backend uses urllib directly; no client object needed."""
        agent = SingleAgent(backend="ollama")
        assert agent.client is None

    def test_anthropic_backend(self):
        with patch("anthropic.Anthropic"):
            agent = SingleAgent(backend="anthropic")
        assert agent.backend == "anthropic"
        assert "claude" in agent.model

    def test_custom_model_overrides_default(self):
        agent = SingleAgent(backend="ollama", model="qwen2.5:32b")
        assert agent.model == "qwen2.5:32b"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            SingleAgent(backend="invalid")

    def test_timeout_stored(self):
        agent = SingleAgent(backend="ollama", timeout=42.0)
        assert agent._timeout == 42.0

    def test_debug_flag_stored(self):
        agent = SingleAgent(backend="ollama", debug=True)
        assert agent.debug is True

    def test_debug_defaults_false(self):
        agent = SingleAgent(backend="ollama")
        assert agent.debug is False


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

    def test_large_grid_caps_pairs_shown(self):
        """Tasks with large grids (>300 cells/pair) should cap training pairs shown."""
        # 20×20 = 400 cells > threshold of 300
        big = np.zeros((20, 20), dtype=np.int32)
        task = {
            "train": [
                {"input": big.copy(), "output": big.copy()},
                {"input": big.copy(), "output": big.copy()},
                {"input": big.copy(), "output": big.copy()},
                {"input": big.copy(), "output": big.copy()},
            ],
            "test": [{"input": big.copy()}],
        }
        agent = make_agent()
        prompt = agent._format_task_prompt(task)
        # Should show at most _LARGE_GRID_MAX_PAIRS (2) pairs
        from agents.single_agent import SingleAgent
        assert "Training pair 3" not in prompt

    def test_small_grid_shows_all_pairs(self):
        """Small grids should not be truncated."""
        small = np.ones((2, 2), dtype=np.int32)
        task = {
            "train": [
                {"input": small.copy(), "output": small.copy()},
                {"input": small.copy(), "output": small.copy()},
                {"input": small.copy(), "output": small.copy()},
            ],
            "test": [{"input": small.copy()}],
        }
        agent = make_agent()
        prompt = agent._format_task_prompt(task)
        assert "Training pair 3" in prompt


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

    def test_returns_last_block_not_first(self):
        """Reasoning models iterate — the last code block is most refined."""
        agent = make_agent()
        text = (
            "First attempt:\n```python\ndef transform(grid):\n    return grid * 0\n```\n"
            "Correction:\n```python\ndef transform(grid):\n    return grid.copy()\n```"
        )
        code = agent._extract_code(text)
        assert "return grid.copy()" in code
        assert "return grid * 0" not in code

    def test_multiple_blocks_last_is_returned(self):
        """With 3+ blocks, the very last one is returned."""
        agent = make_agent()
        blocks = "\n".join(
            f"```python\ndef transform(grid):\n    return grid + {i}\n```"
            for i in range(5)
        )
        code = agent._extract_code(blocks)
        assert "return grid + 4" in code

    def test_grid_literal_block_rejected(self):
        """A code block containing only a grid literal (no def) returns None."""
        agent = make_agent()
        text = "```python\n[[1, 2], [3, 4]]\n```"
        code = agent._extract_code(text)
        assert code is None

    def test_prefers_def_block_over_grid_literal(self):
        """When both a grid literal and a def block exist, the def block wins."""
        agent = make_agent()
        text = (
            "```python\n[[1, 2], [3, 4]]\n```\n"
            "```python\ndef transform(grid):\n    return grid.copy()\n```"
        )
        code = agent._extract_code(text)
        assert code is not None
        assert "def transform" in code

    def test_ast_valid_block_preferred_over_invalid(self):
        """Returns the last syntactically-valid block with a def."""
        agent = make_agent()
        text = (
            "```python\ndef transform(grid):\n    return grid +\n```\n"  # syntax error
            "```python\ndef transform(grid):\n    return grid.copy()\n```"
        )
        code = agent._extract_code(text)
        assert "return grid.copy()" in code

    def test_prose_trailing_function_trimmed(self):
        """Code block with prose after the function body should be salvaged."""
        agent = make_agent()
        text = (
            "```python\n"
            "def transform(grid):\n"
            "    return grid.copy()\n"
            "\n"
            "Then we test it with example 1.\n"  # prose trailing after function
            "Output: [[1,2],[3,4]]\n"
            "```"
        )
        code = agent._extract_code(text)
        assert code is not None
        assert "def transform" in code
        assert "Then we test" not in code


# ---------------------------------------------------------------------------
# _truncate_to_valid_function
# ---------------------------------------------------------------------------

class TestTruncateToValidFunction:
    def test_clean_function_unchanged(self):
        from agents.single_agent import SingleAgent
        block = "def transform(grid):\n    return grid.copy()"
        result = SingleAgent._truncate_to_valid_function(block)
        assert result is not None
        assert "def transform" in result

    def test_prose_suffix_removed(self):
        from agents.single_agent import SingleAgent
        block = "def transform(grid):\n    return grid.copy()\n\nThen test it.\nOutput: [[1]]"
        result = SingleAgent._truncate_to_valid_function(block)
        assert result is not None
        assert "Then test it" not in result
        assert "def transform" in result

    def test_returns_none_for_pure_prose(self):
        from agents.single_agent import SingleAgent
        result = SingleAgent._truncate_to_valid_function("This is just prose.")
        assert result is None

    def test_returns_none_for_no_def(self):
        from agents.single_agent import SingleAgent
        result = SingleAgent._truncate_to_valid_function("x = 1\ny = 2")
        assert result is None


# ---------------------------------------------------------------------------
# _block_analysis
# ---------------------------------------------------------------------------

class TestBlockAnalysis:
    def _inp(self):
        return np.array([[0, 1], [1, 0]], dtype=np.int32)

    def test_returns_none_when_not_divisible(self):
        inp = np.array([[1, 2, 3]], dtype=np.int32)
        out = np.array([[1, 2, 3, 4]], dtype=np.int32)
        assert SingleAgent._block_analysis(inp, out) is None

    def test_returns_string_for_divisible_output(self):
        inp = self._inp()
        out = np.tile(inp, (2, 2))  # 4×4 — two blocks each direction
        result = SingleAgent._block_analysis(inp, out)
        assert isinstance(result, str)

    def test_identifies_all_zeros_block(self):
        inp = np.array([[0, 1], [1, 1]], dtype=np.int32)
        # Manually build a 4×4 output where block(0,0) is zeros
        out = np.zeros((4, 4), dtype=np.int32)
        out[0:2, 2:4] = inp  # block(0,1) = inp
        out[2:4, 0:2] = inp  # block(1,0) = inp
        out[2:4, 2:4] = inp  # block(1,1) = inp
        analysis = SingleAgent._block_analysis(inp, out)
        assert "all zeros" in analysis

    def test_identifies_input_grid_block(self):
        inp = self._inp()
        # block(0,0) = inp, rest zeros
        out = np.zeros((4, 4), dtype=np.int32)
        out[0:2, 0:2] = inp
        analysis = SingleAgent._block_analysis(inp, out)
        assert "input grid" in analysis.lower()

    def test_block_positions_referenced(self):
        inp = self._inp()
        out = np.tile(inp, (2, 2))
        analysis = SingleAgent._block_analysis(inp, out)
        assert "block(0,0)" in analysis
        assert "block(1,1)" in analysis

    def test_prompt_includes_analysis_when_output_is_multiple_of_input(self):
        """_format_task_prompt should embed block analysis for tiling tasks."""
        inp = np.array([[0, 1], [1, 0]], dtype=np.int32)
        out = np.tile(inp, (2, 2))
        task = {
            "train": [{"input": inp, "output": out}],
            "test": [{"input": inp}],
        }
        agent = make_agent()
        prompt = agent._format_task_prompt(task)
        assert "block(" in prompt


# ---------------------------------------------------------------------------
# _call_ollama — native Ollama API (urllib mocked)
# ---------------------------------------------------------------------------

def _ollama_ctx(content: str = "", thinking: str = "") -> MagicMock:
    """Return a context-manager mock for urllib.request.urlopen."""
    body = json.dumps(
        {"message": {"role": "assistant", "content": content, "thinking": thinking}}
    ).encode()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.read = MagicMock(return_value=body)
    return cm


class TestCallOllama:
    def _agent(self) -> SingleAgent:
        return SingleAgent(backend="ollama", timeout=10.0)

    def test_content_returned_directly(self):
        agent = self._agent()
        with patch("urllib.request.urlopen", return_value=_ollama_ctx(content="Hello")):
            result = agent._call_ollama([{"role": "user", "content": "hi"}])
        assert result == "Hello"

    def test_thinking_only_wrapped_in_tags(self):
        agent = self._agent()
        with patch("urllib.request.urlopen", return_value=_ollama_ctx(thinking="I reasoned")):
            result = agent._call_ollama([{"role": "user", "content": "hi"}])
        assert result == "<think>I reasoned</think>"

    def test_thinking_and_content_combined(self):
        agent = self._agent()
        with patch("urllib.request.urlopen", return_value=_ollama_ctx(content="Answer", thinking="reasoning")):
            result = agent._call_ollama([{"role": "user", "content": "hi"}])
        assert "<think>reasoning</think>" in result
        assert "Answer" in result

    def test_empty_response_returns_empty_string(self):
        agent = self._agent()
        with patch("urllib.request.urlopen", return_value=_ollama_ctx()):
            result = agent._call_ollama([{"role": "user", "content": "hi"}])
        assert result == ""

    def test_timeout_raises_timeout_error(self):
        import socket
        agent = self._agent()
        with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
            with pytest.raises(TimeoutError):
                agent._call_ollama([{"role": "user", "content": "hi"}])


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

    def test_fallback_to_any_callable_when_no_transform(self):
        """If the model names the function differently, we still use it."""
        agent = make_agent()
        code = "def my_func(grid):\n    return grid.copy()"
        output, error = agent._execute(code, self._g())
        assert error is None
        np.testing.assert_array_equal(output, self._g())

    def test_fallback_ignores_dsl_helpers(self):
        """The fallback should not accidentally pick a DSL function."""
        agent = make_agent()
        # Only a non-transform name defined; DSL names should be ignored
        code = "def solve(grid):\n    return np.zeros((1, 1), dtype=np.int32)"
        output, error = agent._execute(code, self._g())
        assert error is None
        assert output.shape == (1, 1)

    def test_input_not_mutated(self):
        agent = make_agent()
        code = "def transform(grid):\n    grid[0, 0] = 99\n    return grid"
        original = self._g()
        original_val = int(original[0, 0])
        agent._execute(code, original)
        assert original[0, 0] == original_val

    def test_stdin_input_call_rejected(self):
        """Code using input() must be rejected with a descriptive error."""
        agent = make_agent()
        code = "def transform(grid):\n    n = int(input())\n    return grid"
        output, error = agent._execute(code, self._g())
        assert output is None
        assert error is not None
        assert "stdin" in error.lower() or "input" in error.lower()

    def test_stdin_sys_stdin_rejected(self):
        """Code using sys.stdin must be rejected."""
        agent = make_agent()
        code = "import sys\ndef transform(grid):\n    data = sys.stdin.read()\n    return grid"
        output, error = agent._execute(code, self._g())
        assert output is None
        assert error is not None


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

    def test_solve_logs_timeout_and_does_not_crash(self, simple_task):
        """A TimeoutError from the model call should be logged, not raised."""
        agent = make_agent()
        agent.client.messages.create.side_effect = TimeoutError("timed out")
        result = agent.solve(simple_task)
        assert not result["success"]
        assert any("timeout" in str(e.get("error", "")) for e in result["log"])

    def test_solve_extracts_code_from_think_block(self, simple_task):
        """Code embedded inside <think> tags should be found via the raw-response fallback."""
        agent = make_agent()
        code_block = "```python\ndef transform(grid):\n    return recolor(grid, 1, 2)\n```"
        # Entire response is inside <think>; stripping it leaves nothing
        response = f"<think>Let me reason...\n{code_block}\n</think>"
        agent.client.messages.create.return_value = _mock_response(response)
        result = agent.solve(simple_task)
        assert result["success"]
