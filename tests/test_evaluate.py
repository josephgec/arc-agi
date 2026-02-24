"""Tests for arc/evaluate.py and arc/visualize.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from arc.evaluate import evaluate_task, evaluate_task_file, print_errors, score_all
from arc.grid import load_task


# ---------------------------------------------------------------------------
# evaluate_task
# ---------------------------------------------------------------------------

class TestEvaluateTask:
    def test_all_correct_with_identity(self, simple_task):
        """Identity function is correct when input == output."""
        identity_task = {
            "train": [
                {
                    "input":  np.array([[1, 2]], dtype=np.int32),
                    "output": np.array([[1, 2]], dtype=np.int32),
                }
            ],
            "test": [],
        }
        result = evaluate_task(identity_task, lambda g: g.copy())
        assert result["all_correct"]
        assert result["n_correct"] == 1
        assert result["n_total"] == 1

    def test_all_wrong_with_identity(self, simple_task):
        """simple_task applies recolor 1→2; identity should fail every pair."""
        result = evaluate_task(simple_task, lambda g: g.copy())
        assert result["n_correct"] == 0
        assert not result["all_correct"]

    def test_correct_transform(self, simple_task):
        """Recolor 1→2 is the rule for simple_task; all pairs should pass."""
        from arc.dsl import recolor
        result = evaluate_task(simple_task, lambda g: recolor(g, 1, 2))
        assert result["all_correct"]
        assert result["n_correct"] == result["n_total"]

    def test_partial_correct(self):
        """One pair matches, one does not — partial scores should be reflected."""
        task = {
            "train": [
                {
                    "input":  np.array([[0]], dtype=np.int32),
                    "output": np.array([[0]], dtype=np.int32),  # identity works
                },
                {
                    "input":  np.array([[1]], dtype=np.int32),
                    "output": np.array([[2]], dtype=np.int32),  # identity fails
                },
            ],
            "test": [],
        }
        result = evaluate_task(task, lambda g: g.copy())
        assert result["n_correct"] == 1
        assert result["n_total"] == 2
        assert not result["all_correct"]

    def test_exception_in_transform_is_caught(self, simple_task):
        """Exceptions inside the transform function must be caught and recorded."""
        def bad_fn(g):
            raise ValueError("boom")

        result = evaluate_task(simple_task, bad_fn)
        assert result["n_correct"] == 0
        for pair in result["pairs"]:
            assert pair["error"] is not None

    def test_shape_mismatch_is_wrong(self, simple_task):
        """A transform that returns the wrong shape should count as incorrect."""
        result = evaluate_task(simple_task, lambda g: np.zeros((1, 1), dtype=np.int32))
        assert result["n_correct"] == 0

    def test_result_has_required_keys(self, simple_task):
        result = evaluate_task(simple_task, lambda g: g)
        assert "pairs"       in result
        assert "all_correct" in result
        assert "n_correct"   in result
        assert "n_total"     in result

    def test_pairs_have_correct_bool(self, simple_task):
        result = evaluate_task(simple_task, lambda g: g)
        for pair in result["pairs"]:
            assert isinstance(pair["correct"], (bool, np.bool_))


# ---------------------------------------------------------------------------
# evaluate_task_file
# ---------------------------------------------------------------------------

class TestEvaluateTaskFile:
    def test_loads_and_evaluates(self, task_json_file):
        result = evaluate_task_file(task_json_file, lambda g: g)
        assert "task_id" in result
        assert result["task_id"] == "test_task"

    def test_accepts_string_path(self, task_json_file):
        result = evaluate_task_file(str(task_json_file), lambda g: g)
        assert result is not None

    def test_task_id_is_filename_stem(self, task_json_file):
        """task_id must equal the file stem, not the full path."""
        result = evaluate_task_file(task_json_file, lambda g: g)
        assert result["task_id"] == task_json_file.stem


# ---------------------------------------------------------------------------
# score_all
# ---------------------------------------------------------------------------

class TestScoreAll:
    def test_zero_solve_rate_with_identity(self, tmp_path):
        """Two tasks where identity is wrong → n_solved == 0."""
        raw = {
            "train": [{"input": [[0, 1]], "output": [[1, 0]]}],
            "test":  [{"input": [[0, 1]]}],
        }
        for name in ["task_a.json", "task_b.json"]:
            (tmp_path / name).write_text(json.dumps(raw))

        result = score_all(tmp_path, lambda g: g, verbose=False)
        assert result["n_tasks"]  == 2
        assert result["n_solved"] == 0
        assert result["accuracy"] == 0.0

    def test_full_solve_rate_with_correct_fn(self, tmp_path):
        """A task where identity IS correct → accuracy == 1.0."""
        raw = {
            "train": [{"input": [[1, 2]], "output": [[1, 2]]}],
            "test":  [{"input": [[1, 2]]}],
        }
        (tmp_path / "task_x.json").write_text(json.dumps(raw))

        result = score_all(tmp_path, lambda g: g.copy(), verbose=False)
        assert result["n_solved"] == 1
        assert result["accuracy"] == 1.0

    def test_limit_parameter(self, tmp_path):
        """limit=3 should evaluate exactly 3 tasks from a directory of 5."""
        raw = {"train": [{"input": [[0]], "output": [[1]]}], "test": [{"input": [[0]]}]}
        for i in range(5):
            (tmp_path / f"task_{i}.json").write_text(json.dumps(raw))

        result = score_all(tmp_path, lambda g: g, verbose=False, limit=3)
        assert result["n_tasks"] == 3

    def test_pair_accuracy_in_result(self, tmp_path):
        raw = {"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[1]]}]}
        (tmp_path / "task.json").write_text(json.dumps(raw))
        result = score_all(tmp_path, lambda g: g.copy(), verbose=False)
        assert result["pair_accuracy"] == 1.0

    def test_results_list_length(self, tmp_path):
        """results list must contain one entry per evaluated task."""
        raw = {"train": [{"input": [[0]], "output": [[0]]}], "test": [{"input": [[0]]}]}
        for i in range(3):
            (tmp_path / f"t{i}.json").write_text(json.dumps(raw))
        result = score_all(tmp_path, lambda g: g.copy(), verbose=False)
        assert len(result["results"]) == 3

    def test_empty_dir_returns_zero(self, tmp_path):
        """An empty directory should produce zeroed metrics without crashing."""
        result = score_all(tmp_path, lambda g: g, verbose=False)
        assert result["n_tasks"]  == 0
        assert result["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# print_errors
# ---------------------------------------------------------------------------

class TestPrintErrors:
    def test_prints_without_crash(self, capsys, simple_task):
        result = evaluate_task(simple_task, lambda g: g)
        result["task_id"] = "dummy"
        print_errors(result)
        captured = capsys.readouterr()
        assert "dummy" in captured.out

    def test_prints_correct_pairs(self, capsys, simple_task):
        from arc.dsl import recolor
        result = evaluate_task(simple_task, lambda g: recolor(g, 1, 2))
        result["task_id"] = "solved"
        print_errors(result)
        captured = capsys.readouterr()
        assert "correct" in captured.out.lower()

    def test_prints_error_when_exception(self, capsys, simple_task):
        """When the transform raises, the task_id must still appear in output."""
        result = evaluate_task(simple_task, lambda g: (_ for _ in ()).throw(RuntimeError("oops")))
        result["task_id"] = "err_task"
        print_errors(result)
        captured = capsys.readouterr()
        assert "err_task" in captured.out


# ---------------------------------------------------------------------------
# visualize — terminal output (no matplotlib required)
# ---------------------------------------------------------------------------

class TestVisualizeTerminal:
    def test_print_grid_no_crash(self, capsys):
        from arc.visualize import print_grid
        grid = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        print_grid(grid)
        out = capsys.readouterr().out
        assert len(out) > 0

    def test_print_grid_with_label(self, capsys):
        from arc.visualize import print_grid
        grid = np.array([[0]], dtype=np.int32)
        print_grid(grid, label="test label")
        out = capsys.readouterr().out
        assert "test label" in out

    def test_print_task_no_crash(self, capsys, simple_task):
        from arc.visualize import print_task
        print_task(simple_task)
        out = capsys.readouterr().out
        assert "Train pair" in out

    def test_print_task_respects_max_pairs(self, capsys, simple_task):
        """max_pairs=1 should suppress the second training pair."""
        from arc.visualize import print_task
        print_task(simple_task, max_pairs=1)
        out = capsys.readouterr().out
        assert "Train pair 0" in out
        assert "Train pair 1" not in out

    def test_all_colors_render(self, capsys):
        """All ten ARC colours (0–9) must produce non-empty ANSI output."""
        from arc.visualize import print_grid
        grid = np.arange(10, dtype=np.int32).reshape(2, 5)
        print_grid(grid)
        out = capsys.readouterr().out
        assert len(out) > 0


# ---------------------------------------------------------------------------
# visualize — matplotlib (mocked to avoid display dependencies)
# ---------------------------------------------------------------------------

class TestVisualizePlot:
    def test_plot_grid_calls_imshow(self):
        import arc.visualize as viz

        grid = np.array([[0, 1], [2, 3]], dtype=np.int32)
        mock_ax = MagicMock()
        viz.plot_grid(grid, ax=mock_ax)
        mock_ax.imshow.assert_called_once()

    def test_plot_grid_sets_title(self):
        import arc.visualize as viz

        grid = np.array([[0, 1]], dtype=np.int32)
        mock_ax = MagicMock()
        viz.plot_grid(grid, ax=mock_ax, title="hello")
        mock_ax.set_title.assert_called_once_with("hello", fontsize=8)

    def test_plot_task_calls_show(self, simple_task):
        import arc.visualize as viz

        with patch("matplotlib.pyplot.show") as mock_show, \
             patch("matplotlib.pyplot.subplots") as mock_subplots, \
             patch("matplotlib.pyplot.tight_layout"):
            n_cols = len(simple_task["train"]) * 2 + len(simple_task["test"])
            mock_subplots.return_value = (MagicMock(), [MagicMock() for _ in range(n_cols)])
            viz.plot_task(simple_task)
            mock_show.assert_called_once()

    def test_plot_task_saves_file(self, simple_task, tmp_path):
        import arc.visualize as viz

        save_path = str(tmp_path / "out.png")
        with patch("matplotlib.pyplot.subplots") as mock_subplots, \
             patch("matplotlib.pyplot.tight_layout"), \
             patch("matplotlib.pyplot.savefig") as mock_save:
            n_cols = len(simple_task["train"]) * 2 + len(simple_task["test"])
            mock_subplots.return_value = (MagicMock(), [MagicMock() for _ in range(n_cols)])
            viz.plot_task(simple_task, save_path=save_path)
            mock_save.assert_called_once()
