"""Tests for agents/single_agent.py.

The Anthropic API and urllib are mocked throughout so no network calls are made.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agents.single_agent import SingleAgent, _grid_to_str, _diff_summary, _strip_thinking


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_agent() -> SingleAgent:
    """Return a SingleAgent backed by a mocked Anthropic client."""
    with patch("anthropic.Anthropic"):
        agent = SingleAgent(backend="anthropic", model="claude-sonnet-4-6", max_retries=2)
    agent.client = MagicMock()
    return agent


def _mock_response(text: str) -> MagicMock:
    """Build a mock object that mimics an Anthropic API response."""
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _ollama_ctx(content: str = "", thinking: str = "") -> MagicMock:
    """Return a context-manager mock for urllib.request.urlopen (streaming NDJSON).

    The mock yields NDJSON lines that mirror Ollama's streaming chat format:
    - Optional thinking chunk (if thinking is non-empty)
    - Optional content chunk (if content is non-empty)
    - A final done=True chunk
    """
    lines: list[bytes] = []
    if thinking:
        lines.append(
            json.dumps({"message": {"content": "", "thinking": thinking}, "done": False}).encode() + b"\n"
        )
    if content:
        lines.append(
            json.dumps({"message": {"content": content, "thinking": ""}, "done": False}).encode() + b"\n"
        )
    lines.append(
        json.dumps({"message": {"content": "", "thinking": ""}, "done": True}).encode() + b"\n"
    )

    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.__iter__ = lambda self: iter(lines)
    return cm


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
# SingleAgent.__init__
# ---------------------------------------------------------------------------

class TestSingleAgentInit:
    def test_ollama_backend(self):
        agent = SingleAgent(backend="ollama")
        assert agent.backend == "ollama"
        assert "deepseek" in agent.model

    def test_ollama_client_is_none(self):
        """Ollama uses urllib directly; no SDK client object is needed."""
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
        """Multi-row grids should have at least one bracket per row."""
        g = np.array([[1, 0], [0, 1]], dtype=np.int32)
        s = _grid_to_str(g)
        assert s.count("[") >= 2

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
        b = np.array([[1]],    dtype=np.int32)
        assert "Shape mismatch" in _diff_summary(a, b)

    def test_lists_differing_cells(self):
        a = np.array([[1, 2]], dtype=np.int32)
        b = np.array([[1, 3]], dtype=np.int32)
        assert "(0,1)" in _diff_summary(a, b)

    def test_truncates_at_max_show(self):
        """More than max_show differences should append a '…more' annotation."""
        a = np.zeros((5, 5), dtype=np.int32)
        b = np.ones( (5, 5), dtype=np.int32)
        assert "more" in _diff_summary(a, b, max_show=3)


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
        """Both training pairs in simple_task should appear in the prompt."""
        agent = make_agent()
        prompt = agent._format_task_prompt(simple_task)
        assert "Training pair 2" in prompt

    def test_large_grid_caps_pairs_shown(self):
        """Tasks with grids over the cell threshold should cap training pairs."""
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
        # _LARGE_GRID_MAX_PAIRS == 2, so pair 3 must not appear
        assert "Training pair 3" not in prompt

    def test_small_grid_shows_all_pairs(self):
        """Small grids should not be truncated regardless of count."""
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
        """A bare `def` outside any code fence should be extracted as a fallback."""
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
        """Reasoning models iterate; the last code block is the most refined."""
        agent = make_agent()
        text = (
            "First attempt:\n```python\ndef transform(grid):\n    return grid * 0\n```\n"
            "Correction:\n```python\ndef transform(grid):\n    return grid.copy()\n```"
        )
        code = agent._extract_code(text)
        assert "return grid.copy()" in code
        assert "return grid * 0" not in code

    def test_multiple_blocks_last_is_returned(self):
        """With 5 blocks, the very last syntactically-valid one is returned."""
        agent = make_agent()
        blocks = "\n".join(
            f"```python\ndef transform(grid):\n    return grid + {i}\n```"
            for i in range(5)
        )
        code = agent._extract_code(blocks)
        assert "return grid + 4" in code

    def test_grid_literal_block_rejected(self):
        """A code block with no `def` (just a grid literal) should return None."""
        agent = make_agent()
        text = "```python\n[[1, 2], [3, 4]]\n```"
        assert agent._extract_code(text) is None

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
        """Code with prose after the function body should be salvaged by trimming."""
        agent = make_agent()
        text = (
            "```python\n"
            "def transform(grid):\n"
            "    return grid.copy()\n"
            "\n"
            "Then we test it with example 1.\n"
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
        block = "def transform(grid):\n    return grid.copy()"
        result = SingleAgent._truncate_to_valid_function(block)
        assert result is not None
        assert "def transform" in result

    def test_prose_suffix_removed(self):
        """Trailing prose lines must be stripped to yield a valid function."""
        block = "def transform(grid):\n    return grid.copy()\n\nThen test it.\nOutput: [[1]]"
        result = SingleAgent._truncate_to_valid_function(block)
        assert result is not None
        assert "Then test it" not in result
        assert "def transform" in result

    def test_returns_none_for_pure_prose(self):
        result = SingleAgent._truncate_to_valid_function("This is just prose.")
        assert result is None

    def test_returns_none_for_no_def(self):
        """Valid Python without a function definition should return None."""
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
        out = np.tile(inp, (2, 2))  # 4×4 = two blocks in each direction
        assert isinstance(SingleAgent._block_analysis(inp, out), str)

    def test_identifies_all_zeros_block(self):
        """A block filled with zeros should be labelled 'all zeros'."""
        inp = np.array([[0, 1], [1, 1]], dtype=np.int32)
        # Build a 4×4 output where block(0,0) is all zeros
        out = np.zeros((4, 4), dtype=np.int32)
        out[0:2, 2:4] = inp  # block(0,1)
        out[2:4, 0:2] = inp  # block(1,0)
        out[2:4, 2:4] = inp  # block(1,1)
        assert "all zeros" in SingleAgent._block_analysis(inp, out)

    def test_identifies_input_grid_block(self):
        """A block that equals the full input should be annotated accordingly."""
        inp = self._inp()
        out = np.zeros((4, 4), dtype=np.int32)
        out[0:2, 0:2] = inp  # block(0,0) = inp
        assert "input grid" in SingleAgent._block_analysis(inp, out).lower()

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
            "test":  [{"input": inp}],
        }
        agent = make_agent()
        assert "block(" in agent._format_task_prompt(task)


# ---------------------------------------------------------------------------
# _call_ollama — streaming Ollama API (urllib mocked)
# ---------------------------------------------------------------------------

def _ollama_chunks(*content_pieces: str, thinking: str = "") -> MagicMock:
    """Build a streaming mock that yields one chunk per content piece.

    Unlike _ollama_ctx (which puts everything in one or two chunks), this
    helper lets tests send content incrementally — essential for verifying
    early-exit behaviour mid-stream.
    """
    lines: list[bytes] = []
    if thinking:
        lines.append(
            json.dumps({"message": {"content": "", "thinking": thinking}, "done": False}).encode() + b"\n"
        )
    for piece in content_pieces:
        lines.append(
            json.dumps({"message": {"content": piece, "thinking": ""}, "done": False}).encode() + b"\n"
        )
    lines.append(
        json.dumps({"message": {"content": "", "thinking": ""}, "done": True}).encode() + b"\n"
    )
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__  = MagicMock(return_value=False)
    cm.__iter__  = lambda self: iter(lines)
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
        """Thinking-only responses should be wrapped in <think> tags."""
        agent = self._agent()
        with patch("urllib.request.urlopen", return_value=_ollama_ctx(thinking="I reasoned")):
            result = agent._call_ollama([{"role": "user", "content": "hi"}])
        assert result == "<think>I reasoned</think>"

    def test_thinking_and_content_combined(self):
        """Responses with both thinking and content should include both parts."""
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
        """A socket.timeout with no partial data must be re-raised as TimeoutError."""
        import socket
        agent = self._agent()
        with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
            with pytest.raises(TimeoutError):
                agent._call_ollama([{"role": "user", "content": "hi"}])

    def test_partial_result_returned_on_deadline(self):
        """When the wall-clock deadline fires mid-stream, partial data is returned."""
        import time
        agent = SingleAgent(backend="ollama", timeout=0.001)  # nearly-zero deadline

        thinking_chunk = (
            json.dumps({"message": {"content": "", "thinking": "partial thinking"}, "done": False}).encode()
            + b"\n"
        )
        done_chunk = (
            json.dumps({"message": {"content": "", "thinking": ""}, "done": True}).encode()
            + b"\n"
        )

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__  = MagicMock(return_value=False)

        def slow_iter():
            yield thinking_chunk
            time.sleep(0.01)  # sleep past the 0.001s deadline
            yield done_chunk

        cm.__iter__ = lambda self: slow_iter()

        with patch("urllib.request.urlopen", return_value=cm):
            result = agent._call_ollama([{"role": "user", "content": "hi"}])

        # Should return partial thinking rather than raise TimeoutError
        assert "partial thinking" in result

    # ------------------------------------------------------------------
    # Early-exit / <think> interaction tests
    # ------------------------------------------------------------------

    def test_no_early_exit_while_think_block_open(self):
        """Draft code inside an unclosed <think> block must not trigger early-exit.

        The stream must continue so that </think> arrives and the final answer
        (after the closing tag) is captured.
        """
        agent = self._agent()
        draft  = "<think>\n```python\ndef transform(grid):\n    return grid * 0\n```\nLet me reconsider...\n"
        final  = "</think>\n```python\ndef transform(grid):\n    return grid.copy()\n```"

        with patch("urllib.request.urlopen", return_value=_ollama_chunks(draft, final)):
            result = agent._call_ollama([{"role": "user", "content": "hi"}])

        # The full content (both pieces) must be returned
        assert "return grid * 0"   in result   # draft present in raw output
        assert "return grid.copy()" in result  # final answer also present

    def test_strip_thinking_removes_draft_leaving_final(self):
        """After _call_ollama returns, _strip_thinking must expose only the final code.

        This is the end-to-end fix: draft inside <think> is stripped; only the
        real answer after </think> is available for code extraction.
        """
        agent = self._agent()
        draft  = "<think>\n```python\ndef transform(grid):\n    return grid * 0\n```\n</think>\n"
        final  = "```python\ndef transform(grid):\n    return grid.copy()\n```"

        with patch("urllib.request.urlopen", return_value=_ollama_chunks(draft, final)):
            raw = agent._call_ollama([{"role": "user", "content": "hi"}])

        cleaned = _strip_thinking(raw)
        assert "return grid * 0"    not in cleaned  # draft stripped
        assert "return grid.copy()" in cleaned       # final answer kept

    def test_early_exit_fires_on_final_code_after_closed_think(self):
        """Early-exit IS allowed once </think> has closed and final code is complete."""
        agent = self._agent()
        thinking_and_final = (
            "<think>some reasoning</think>\n"
            "```python\ndef transform(grid):\n    return grid.copy()\n```"
        )
        trailing_prose = "Here is why this works..."  # should never arrive

        with patch("urllib.request.urlopen",
                   return_value=_ollama_chunks(thinking_and_final, trailing_prose)):
            result = agent._call_ollama([{"role": "user", "content": "hi"}])

        # Early-exit fired before trailing_prose was consumed
        assert "trailing_prose" not in result
        assert "return grid.copy()" in result

    def test_early_exit_fires_normally_when_no_think_tags(self):
        """Without any <think> tags, early-exit should still work as before."""
        agent = self._agent()
        code_chunk    = "```python\ndef transform(grid):\n    return grid.copy()\n```"
        trailing_text = " extra text that should be cut off"

        with patch("urllib.request.urlopen",
                   return_value=_ollama_chunks(code_chunk, trailing_text)):
            result = agent._call_ollama([{"role": "user", "content": "hi"}])

        assert "return grid.copy()" in result
        assert "extra text" not in result


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
        code = "x = 1"  # no function defined at all
        output, error = agent._execute(code, self._g())
        assert output is None
        assert error is not None

    def test_output_is_int32(self):
        agent = make_agent()
        code = "def transform(grid):\n    return np.zeros_like(grid)"
        output, error = agent._execute(code, self._g())
        assert output.dtype == np.int32

    def test_list_output_converted(self):
        """A function returning a nested list should be auto-converted to ndarray."""
        agent = make_agent()
        code = "def transform(grid):\n    return [[0, 0], [0, 0]]"
        output, error = agent._execute(code, self._g())
        assert error is None
        assert isinstance(output, np.ndarray)

    def test_dsl_available(self):
        """DSL helpers like recolor() must be available without imports."""
        agent = make_agent()
        code = "def transform(grid):\n    return recolor(grid, 1, 9)"
        g = np.array([[1, 0]], dtype=np.int32)
        output, error = agent._execute(code, g)
        assert error is None
        assert output[0, 0] == 9

    def test_numpy_available_as_np(self):
        """numpy must be importable as `np` inside generated code."""
        agent = make_agent()
        code = "def transform(grid):\n    return np.zeros((2, 2), dtype=np.int32)"
        output, error = agent._execute(code, self._g())
        assert error is None
        assert output.shape == (2, 2)

    def test_fallback_to_any_callable_when_no_transform(self):
        """If the model names its function differently, the agent should still use it."""
        agent = make_agent()
        code = "def my_func(grid):\n    return grid.copy()"
        output, error = agent._execute(code, self._g())
        assert error is None
        np.testing.assert_array_equal(output, self._g())

    def test_fallback_ignores_dsl_helpers(self):
        """The callable fallback must not accidentally pick a DSL function."""
        agent = make_agent()
        code = "def solve(grid):\n    return np.zeros((1, 1), dtype=np.int32)"
        output, error = agent._execute(code, self._g())
        assert error is None
        assert output.shape == (1, 1)

    def test_input_not_mutated(self):
        """The original input grid must not be modified, even if code tries to."""
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
        """Code referencing sys.stdin must also be rejected."""
        agent = make_agent()
        code = "import sys\ndef transform(grid):\n    data = sys.stdin.read()\n    return grid"
        output, error = agent._execute(code, self._g())
        assert output is None
        assert error is not None


# ---------------------------------------------------------------------------
# _evaluate_test
# ---------------------------------------------------------------------------

class TestEvaluateTest:
    """Tests for the held-out test evaluation — the honest accuracy metric."""

    def _task_with_test_output(self, correct: bool):
        """Build a simple task whose test pair has a ground-truth output.

        The rule is recolor 1→2.  `correct=True` means the test output
        expects that transformation; `correct=False` expects the wrong answer.
        """
        test_input  = np.array([[0, 1, 1]], dtype=np.int32)
        test_output = np.array([[0, 2, 2]], dtype=np.int32) if correct else np.array([[0, 9, 9]], dtype=np.int32)
        return {
            "train": [
                {
                    "input":  np.array([[1]], dtype=np.int32),
                    "output": np.array([[2]], dtype=np.int32),
                }
            ],
            "test": [{"input": test_input, "output": test_output}],
        }

    def test_correct_prediction_returns_true(self):
        agent = make_agent()
        code = "def transform(grid):\n    return recolor(grid, 1, 2)"
        test_pair = self._task_with_test_output(correct=True)["test"][0]
        assert agent._evaluate_test(code, test_pair) is True

    def test_wrong_prediction_returns_false(self):
        agent = make_agent()
        code = "def transform(grid):\n    return grid.copy()"  # identity — wrong
        test_pair = self._task_with_test_output(correct=True)["test"][0]
        assert agent._evaluate_test(code, test_pair) is False

    def test_runtime_error_returns_false(self):
        agent = make_agent()
        code = "def transform(grid):\n    raise RuntimeError('crash')"
        test_pair = self._task_with_test_output(correct=True)["test"][0]
        assert agent._evaluate_test(code, test_pair) is False

    def test_solve_sets_test_correct_true_when_code_is_right(self):
        """solve() must set test_correct=True when best code passes the test pair."""
        agent = make_agent()
        task = self._task_with_test_output(correct=True)
        agent.client.messages.create.return_value = _mock_response(
            "```python\ndef transform(grid):\n    return recolor(grid, 1, 2)\n```"
        )
        result = agent.solve(task)
        assert result["success"] is True
        assert result["test_correct"] is True

    def test_solve_sets_test_correct_false_when_code_overfits(self):
        """An overfit function passes training but fails the test pair."""
        agent = make_agent()
        # The task has one training pair: [[1]] → [[2]]
        # The overfit function hardcodes that exact input and returns the wrong
        # answer for anything else — including the test input [[0, 1, 1]].
        task = self._task_with_test_output(correct=True)
        overfit_code = (
            "```python\n"
            "def transform(grid):\n"
            "    import numpy as np\n"
            "    if np.array_equal(grid, [[1]]):\n"
            "        return np.array([[2]], dtype=np.int32)\n"
            "    return grid.copy()  # wrong for all other inputs\n"
            "```"
        )
        agent.client.messages.create.return_value = _mock_response(overfit_code)
        result = agent.solve(task)
        # Training passes (the hardcoded answer matches), but test fails
        assert result["success"] is True
        assert result["test_correct"] is False

    def test_solve_returns_test_correct_none_when_no_ground_truth(self, simple_task):
        """simple_task has no test output — test_correct must be None."""
        agent = make_agent()
        agent.client.messages.create.return_value = _mock_response(
            "```python\ndef transform(grid):\n    return recolor(grid, 1, 2)\n```"
        )
        result = agent.solve(simple_task)
        assert result["test_correct"] is None

    def test_solve_sets_test_correct_on_failed_solve_too(self):
        """Even when success=False, test_correct reflects the best code's result."""
        agent = make_agent()
        task = self._task_with_test_output(correct=True)
        # Correct code for the test but deliberately wrong for training
        # (training pair expects [[2]] but we return [[1]])
        agent.client.messages.create.return_value = _mock_response(
            "```python\ndef transform(grid):\n    return recolor(grid, 1, 2)\n```"
        )
        # Force all training pairs to fail by making training output something else
        task["train"][0]["output"] = np.array([[9]], dtype=np.int32)
        result = agent.solve(task)
        assert result["success"] is False
        # The best code still produces the right answer on the test pair
        assert result["test_correct"] is True


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
        """Exceptions during evaluation must be stored per-pair, not raised."""
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
        eval_result = agent._evaluate_code("def transform(grid):\n    return grid", simple_task)
        feedback = agent._format_error_feedback(eval_result, attempt=1)
        assert "WRONG" in feedback

    def test_mentions_correct_pairs(self):
        """Pairs that pass should be marked CORRECT in the feedback."""
        agent = make_agent()
        task = {
            "train": [
                {
                    "input":  np.array([[1]], dtype=np.int32),
                    "output": np.array([[1]], dtype=np.int32),  # identity works
                },
                {
                    "input":  np.array([[0]], dtype=np.int32),
                    "output": np.array([[1]], dtype=np.int32),  # identity fails
                },
            ],
            "test": [],
        }
        eval_result = agent._evaluate_code("def transform(grid):\n    return grid.copy()", task)
        feedback = agent._format_error_feedback(eval_result, attempt=1)
        assert "CORRECT" in feedback

    def test_includes_revise_instruction(self, simple_task):
        """Feedback must ask the model to revise its answer."""
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
        """Agent should succeed after two wrong attempts followed by a correct one."""
        agent = make_agent()
        agent.client.messages.create.side_effect = [
            _mock_response(self._wrong_code()),
            _mock_response(self._wrong_code()),
            _mock_response(self._correct_code()),
        ]
        result = agent.solve(simple_task)
        assert result["success"]
        assert result["n_attempts"] == 3

    def test_solve_fails_after_max_retries(self, simple_task):
        """Exhausting all retries without a correct solution should return success=False."""
        agent = make_agent()
        agent.client.messages.create.return_value = _mock_response(self._wrong_code())
        result = agent.solve(simple_task)
        assert not result["success"]

    def test_solve_handles_missing_code_block(self, simple_task):
        """A response with no code block should be retried; eventual success must be detected."""
        agent = make_agent()
        agent.client.messages.create.side_effect = [
            _mock_response("I don't know"),
            _mock_response(self._correct_code()),
        ]
        result = agent.solve(simple_task)
        assert result["success"]

    def test_solve_result_always_has_test_correct_key(self, simple_task):
        """test_correct must always be present in the result dict."""
        agent = make_agent()
        agent.client.messages.create.return_value = _mock_response(self._correct_code())
        result = agent.solve(simple_task)
        assert "test_correct" in result

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
        # Wrong answer but still a Grid (best guess from failed code)
        assert isinstance(output, np.ndarray)

    def test_predict_returns_none_when_code_crashes(self, simple_task):
        """If the best code raises on the test input, predict() must return None."""
        agent = make_agent()
        crash_code = "```python\ndef transform(grid):\n    raise RuntimeError('crash')\n```"
        agent.client.messages.create.return_value = _mock_response(crash_code)
        assert agent.predict(simple_task) is None

    def test_solve_logs_timeout_and_does_not_crash(self, simple_task):
        """A TimeoutError from the model must be logged gracefully, not re-raised."""
        agent = make_agent()
        agent.client.messages.create.side_effect = TimeoutError("timed out")
        result = agent.solve(simple_task)
        assert not result["success"]
        assert any("timeout" in str(e.get("error", "")) for e in result["log"])

    def test_solve_extracts_code_from_think_block(self, simple_task):
        """Code embedded inside <think> tags should be found via the raw-response fallback."""
        agent = make_agent()
        code_block = "```python\ndef transform(grid):\n    return recolor(grid, 1, 2)\n```"
        # The entire response is inside <think>; stripping it leaves nothing in clean_response.
        response = f"<think>Let me reason...\n{code_block}\n</think>"
        agent.client.messages.create.return_value = _mock_response(response)
        result = agent.solve(simple_task)
        assert result["success"]
