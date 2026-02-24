"""Tests for arc/grid.py."""
from __future__ import annotations

import json
import numpy as np
import pytest

from arc.grid import (
    COLOR_NAMES,
    Grid,
    background_color,
    grid_from_list,
    grid_to_list,
    grids_equal,
    load_task,
    unique_colors,
)


# ---------------------------------------------------------------------------
# COLOR_NAMES
# ---------------------------------------------------------------------------

class TestColorNames:
    def test_has_ten_entries(self):
        assert len(COLOR_NAMES) == 10

    def test_known_values(self):
        assert COLOR_NAMES[0] == "black"
        assert COLOR_NAMES[1] == "blue"
        assert COLOR_NAMES[9] == "maroon"

    def test_all_keys_are_ints_0_to_9(self):
        assert set(COLOR_NAMES.keys()) == set(range(10))


# ---------------------------------------------------------------------------
# load_task
# ---------------------------------------------------------------------------

class TestLoadTask:
    def test_loads_train_and_test(self, task_json_file):
        task = load_task(task_json_file)
        assert "train" in task
        assert "test" in task

    def test_train_pairs_are_dicts_with_input_output(self, task_json_file):
        task = load_task(task_json_file)
        for pair in task["train"]:
            assert "input" in pair
            assert "output" in pair

    def test_grids_are_numpy_int32(self, task_json_file):
        """All loaded grids must be int32 numpy arrays."""
        task = load_task(task_json_file)
        for pair in task["train"]:
            assert isinstance(pair["input"], np.ndarray)
            assert pair["input"].dtype == np.int32
            assert isinstance(pair["output"], np.ndarray)
            assert pair["output"].dtype == np.int32

    def test_test_pairs_have_input(self, task_json_file):
        task = load_task(task_json_file)
        assert len(task["test"]) == 1
        assert "input" in task["test"][0]

    def test_test_output_loaded_when_present(self, task_json_file):
        """Test outputs are optional in ARC but should be included when present."""
        task = load_task(task_json_file)
        assert "output" in task["test"][0]

    def test_values_are_correct(self, task_json_file):
        task = load_task(task_json_file)
        np.testing.assert_array_equal(task["train"][0]["input"],  [[0, 1], [1, 0]])
        np.testing.assert_array_equal(task["train"][0]["output"], [[0, 2], [2, 0]])

    def test_real_training_file(self, training_dir):
        """Smoke-test against the actual ARC dataset if it is present."""
        path = next(training_dir.glob("*.json"))
        task = load_task(path)
        assert len(task["train"]) >= 1
        assert len(task["test"])  >= 1

    def test_test_without_output(self, tmp_path):
        """Test pairs that have no output key should not get one added."""
        raw = {
            "train": [{"input": [[0, 1]], "output": [[1, 0]]}],
            "test":  [{"input": [[0, 0]]}],
        }
        p = tmp_path / "no_output.json"
        p.write_text(json.dumps(raw))
        task = load_task(p)
        assert "output" not in task["test"][0]

    def test_accepts_pathlib_path(self, task_json_file):
        task = load_task(task_json_file)
        assert task is not None

    def test_accepts_string_path(self, task_json_file):
        task = load_task(str(task_json_file))
        assert task is not None


# ---------------------------------------------------------------------------
# grids_equal
# ---------------------------------------------------------------------------

class TestGridsEqual:
    def test_identical_grids(self):
        g = np.array([[1, 2], [3, 4]], dtype=np.int32)
        assert grids_equal(g, g.copy())

    def test_different_values(self):
        a = np.array([[1, 2]], dtype=np.int32)
        b = np.array([[1, 3]], dtype=np.int32)
        assert not grids_equal(a, b)

    def test_different_shapes(self):
        a = np.array([[1, 2]], dtype=np.int32)
        b = np.array([[1], [2]], dtype=np.int32)
        assert not grids_equal(a, b)

    def test_single_cell_equal(self):
        assert grids_equal(np.array([[5]], dtype=np.int32), np.array([[5]], dtype=np.int32))

    def test_single_cell_unequal(self):
        assert not grids_equal(np.array([[5]], dtype=np.int32), np.array([[6]], dtype=np.int32))

    def test_all_zeros(self):
        a = np.zeros((3, 3), dtype=np.int32)
        assert grids_equal(a, a.copy())


# ---------------------------------------------------------------------------
# grid_from_list / grid_to_list
# ---------------------------------------------------------------------------

class TestGridConversions:
    def test_from_list_dtype(self):
        g = grid_from_list([[1, 2], [3, 4]])
        assert g.dtype == np.int32

    def test_from_list_values(self):
        g = grid_from_list([[1, 2], [3, 4]])
        assert g[0, 0] == 1
        assert g[1, 1] == 4

    def test_round_trip(self):
        """Converting list → Grid → list should reproduce the original."""
        original = [[0, 1, 2], [3, 4, 5]]
        g = grid_from_list(original)
        result = grid_to_list(g)
        assert result == original

    def test_to_list_returns_native_ints(self):
        """grid_to_list must return plain Python ints, not numpy scalars."""
        g = np.array([[1, 2]], dtype=np.int32)
        lst = grid_to_list(g)
        assert isinstance(lst[0][0], int)


# ---------------------------------------------------------------------------
# unique_colors
# ---------------------------------------------------------------------------

class TestUniqueColors:
    def test_single_color(self):
        g = np.zeros((2, 2), dtype=np.int32)
        assert unique_colors(g) == [0]

    def test_multiple_colors_sorted(self):
        g = np.array([[3, 1, 2], [0, 3, 1]], dtype=np.int32)
        assert unique_colors(g) == [0, 1, 2, 3]

    def test_all_ten_colors(self):
        g = np.arange(10, dtype=np.int32).reshape(2, 5)
        assert unique_colors(g) == list(range(10))


# ---------------------------------------------------------------------------
# background_color
# ---------------------------------------------------------------------------

class TestBackgroundColor:
    def test_most_frequent_is_background(self):
        g = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int32)
        assert background_color(g) == 0

    def test_non_zero_background(self):
        g = np.array([[7, 7, 7], [7, 1, 7]], dtype=np.int32)
        assert background_color(g) == 7

    def test_tie_picks_first(self):
        """numpy argmax picks the first maximum on a tie — just verify a valid colour."""
        g = np.array([[0, 1]], dtype=np.int32)
        result = background_color(g)
        assert result in [0, 1]

    def test_all_same(self):
        g = np.full((3, 3), 5, dtype=np.int32)
        assert background_color(g) == 5
