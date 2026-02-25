"""Tests for arc/sandbox.py — sandboxed execution and evaluation.

These tests exercise arc.sandbox directly, independent of any agent.
This matters for the multi-agent framework: every future agent (Coder,
Critic, Analyst, …) will call sandbox.execute() and sandbox.evaluate_code(),
so this suite forms the contract those agents depend on.
"""
from __future__ import annotations

import numpy as np
import pytest

from arc import sandbox
from arc.sandbox import EXECUTION_TIMEOUT, DSL_NAMESPACE, execute, evaluate_code


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def grid(*rows):
    """Build an int32 Grid from row tuples."""
    return np.array(rows, dtype=np.int32)


def simple_task():
    """Minimal ARC task: recolor 1→2."""
    inp  = grid([0, 1], [1, 0])
    out  = grid([0, 2], [2, 0])
    return {
        "train": [{"input": inp, "output": out}],
        "test":  [{"input": inp}],
    }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_execution_timeout_is_positive(self):
        assert EXECUTION_TIMEOUT > 0

    def test_dsl_namespace_contains_numpy(self):
        assert "np" in DSL_NAMESPACE
        assert "numpy" in DSL_NAMESPACE

    def test_dsl_namespace_contains_all_primitives(self):
        expected = {
            "crop", "rotate", "flip", "translate", "scale", "tile",
            "recolor", "mask", "overlay", "flood_fill",
            "find_objects", "bounding_box", "crop_to_content",
        }
        assert expected.issubset(DSL_NAMESPACE)


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------

class TestExecute:
    def _g(self):
        return grid([1, 2], [3, 4])

    def test_valid_code_returns_grid(self):
        out, err = execute("def transform(grid):\n    return grid.copy()", self._g())
        assert err is None
        np.testing.assert_array_equal(out, self._g())

    def test_output_is_int32(self):
        out, err = execute("def transform(grid):\n    return np.zeros_like(grid)", self._g())
        assert err is None
        assert out.dtype == np.int32

    def test_list_output_converted_to_ndarray(self):
        out, err = execute("def transform(grid):\n    return [[0, 0], [0, 0]]", self._g())
        assert err is None
        assert isinstance(out, np.ndarray)

    def test_compile_error_returned_as_error(self):
        out, err = execute("def transform(grid):\n    return grid +", self._g())
        assert out is None
        assert "error" in err.lower()

    def test_runtime_error_returned_as_error(self):
        out, err = execute("def transform(grid):\n    raise ValueError('oops')", self._g())
        assert out is None
        assert "oops" in err

    def test_missing_transform_returned_as_error(self):
        out, err = execute("x = 1", self._g())
        assert out is None
        assert err is not None

    def test_fallback_to_any_callable(self):
        """A function named something other than 'transform' should still be called."""
        out, err = execute("def solve(grid):\n    return grid.copy()", self._g())
        assert err is None
        np.testing.assert_array_equal(out, self._g())

    def test_dsl_available_without_imports(self):
        out, err = execute("def transform(grid):\n    return recolor(grid, 1, 9)", self._g())
        assert err is None
        assert out[0, 0] == 9

    def test_numpy_available_as_np(self):
        out, err = execute(
            "def transform(grid):\n    return np.zeros((2, 2), dtype=np.int32)",
            self._g(),
        )
        assert err is None
        assert out.shape == (2, 2)

    def test_stdin_rejected_before_subprocess(self):
        out, err = execute("def transform(grid):\n    n = int(input())\n    return grid", self._g())
        assert out is None
        assert "stdin" in err.lower() or "input" in err.lower()

    def test_sys_stdin_rejected(self):
        out, err = execute(
            "import sys\ndef transform(grid):\n    sys.stdin.read()\n    return grid",
            self._g(),
        )
        assert out is None
        assert err is not None

    def test_input_grid_not_mutated(self):
        original = self._g()
        original_val = int(original[0, 0])
        execute("def transform(grid):\n    grid[0, 0] = 99\n    return grid", original)
        assert original[0, 0] == original_val

    def test_infinite_loop_killed_by_timeout(self):
        out, err = execute("def transform(grid):\n    while True: pass", self._g(), timeout=2.0)
        assert out is None
        assert "timed out" in err.lower() or "timeout" in err.lower()

    def test_custom_timeout_respected(self):
        """A very short timeout must fire before a slow computation completes."""
        slow_code = (
            "import time\n"
            "def transform(grid):\n"
            "    time.sleep(10)\n"
            "    return grid.copy()"
        )
        out, err = execute(slow_code, self._g(), timeout=1.0)
        assert out is None
        assert err is not None


# ---------------------------------------------------------------------------
# evaluate_code()
# ---------------------------------------------------------------------------

class TestEvaluateCode:
    def test_all_correct(self):
        task = simple_task()
        result = evaluate_code("def transform(grid):\n    return recolor(grid, 1, 2)", task)
        assert result["all_correct"] is True
        assert result["n_correct"] == 1
        assert result["n_total"] == 1

    def test_all_wrong(self):
        task = simple_task()
        result = evaluate_code("def transform(grid):\n    return grid.copy()", task)
        assert result["n_correct"] == 0
        assert result["all_correct"] is False

    def test_result_keys_present(self):
        task = simple_task()
        result = evaluate_code("def transform(grid):\n    return grid.copy()", task)
        assert all(k in result for k in ["pairs", "n_correct", "n_total", "all_correct"])

    def test_error_captured_per_pair(self):
        task = simple_task()
        result = evaluate_code("def transform(grid):\n    raise RuntimeError('bad')", task)
        assert result["pairs"][0]["error"] is not None
        assert result["pairs"][0]["correct"] is False

    def test_multiple_pairs(self):
        g = np.zeros((2, 2), dtype=np.int32)
        task = {
            "train": [
                {"input": g.copy(), "output": g.copy()},
                {"input": g.copy(), "output": g.copy()},
                {"input": g.copy(), "output": g.copy()},
            ],
            "test": [],
        }
        result = evaluate_code("def transform(grid):\n    return grid.copy()", task)
        assert result["n_total"] == 3
        assert result["n_correct"] == 3
        assert result["all_correct"] is True

    def test_passes_timeout_to_execute(self):
        """A short timeout must propagate to each execute() call."""
        task = simple_task()
        slow_code = (
            "import time\n"
            "def transform(grid):\n"
            "    time.sleep(10)\n"
            "    return grid.copy()"
        )
        result = evaluate_code(slow_code, task, timeout=1.0)
        assert result["n_correct"] == 0
        assert "timed out" in result["pairs"][0]["error"].lower() or \
               "timeout"   in result["pairs"][0]["error"].lower()
