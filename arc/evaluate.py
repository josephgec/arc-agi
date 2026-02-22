"""Evaluation harness for ARC-AGI tasks."""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Callable

from .grid import Grid, grids_equal, load_task


TransformFn = Callable[[Grid], Grid]


def evaluate_task(task: dict, transform_fn: TransformFn) -> dict:
    """Apply transform_fn to every training pair and return per-pair results.

    Returns:
        {
            'pairs': [{'correct': bool, 'predicted': Grid, 'expected': Grid, 'error': str|None}],
            'all_correct': bool,
            'n_correct': int,
            'n_total': int,
        }
    """
    pairs = []
    for pair in task["train"]:
        inp = pair["input"]
        expected = pair["output"]
        try:
            predicted = transform_fn(inp)
            correct = grids_equal(predicted, expected)
            pairs.append(
                {"correct": correct, "predicted": predicted, "expected": expected, "error": None}
            )
        except Exception as e:
            pairs.append(
                {
                    "correct": False,
                    "predicted": None,
                    "expected": expected,
                    "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                }
            )

    n_correct = sum(p["correct"] for p in pairs)
    return {
        "pairs": pairs,
        "all_correct": n_correct == len(pairs),
        "n_correct": n_correct,
        "n_total": len(pairs),
    }


def evaluate_task_file(path: str | Path, transform_fn: TransformFn) -> dict:
    """Load a task file and evaluate it."""
    task = load_task(path)
    result = evaluate_task(task, transform_fn)
    result["task_id"] = Path(path).stem
    return result


def score_all(
    tasks_dir: str | Path,
    transform_fn: TransformFn,
    limit: int | None = None,
    verbose: bool = True,
) -> dict:
    """Score transform_fn against all tasks in a directory.

    Args:
        tasks_dir: directory containing .json task files
        transform_fn: function Grid -> Grid
        limit: only evaluate the first N tasks (useful for quick smoke-tests)
        verbose: print per-task results

    Returns:
        {
            'n_tasks': int,
            'n_solved': int,          # tasks where all pairs correct
            'accuracy': float,        # n_solved / n_tasks
            'pair_accuracy': float,   # correct pairs / total pairs
            'results': list[dict],
        }
    """
    tasks_dir = Path(tasks_dir)
    task_files = sorted(tasks_dir.glob("*.json"))
    if limit:
        task_files = task_files[:limit]

    results = []
    total_pairs = 0
    correct_pairs = 0
    n_solved = 0

    for path in task_files:
        result = evaluate_task_file(path, transform_fn)
        results.append(result)
        n_solved += int(result["all_correct"])
        total_pairs += result["n_total"]
        correct_pairs += result["n_correct"]

        if verbose:
            status = "SOLVED" if result["all_correct"] else f"{result['n_correct']}/{result['n_total']}"
            print(f"  {result['task_id']}  [{status}]")

    n_tasks = len(task_files)
    summary = {
        "n_tasks": n_tasks,
        "n_solved": n_solved,
        "accuracy": n_solved / n_tasks if n_tasks else 0.0,
        "pair_accuracy": correct_pairs / total_pairs if total_pairs else 0.0,
        "results": results,
    }

    if verbose:
        print(
            f"\nSolved {n_solved}/{n_tasks} tasks "
            f"({summary['accuracy']:.1%})  |  "
            f"Pair accuracy: {summary['pair_accuracy']:.1%}"
        )

    return summary


def print_errors(result: dict) -> None:
    """Pretty-print the failure details for a single evaluated task."""
    from .visualize import print_grid

    task_id = result.get("task_id", "?")
    print(f"\n=== {task_id} ===")
    for i, pair in enumerate(result["pairs"]):
        if pair["correct"]:
            print(f"  Pair {i}: correct")
            continue
        print(f"  Pair {i}: WRONG")
        if pair["error"]:
            print(f"    Error: {pair['error']}")
        else:
            print("  Expected:")
            print_grid(pair["expected"])
            print("  Predicted:")
            print_grid(pair["predicted"])
