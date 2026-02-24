"""Evaluation harness for ARC-AGI tasks.

Provides utilities for running a transform function against task training pairs
and aggregating correctness metrics across a directory of task files.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Callable

from .grid import Grid, grids_equal, load_task


TransformFn = Callable[[Grid], Grid]


def evaluate_task(task: dict, transform_fn: TransformFn) -> dict:
    """Apply transform_fn to every training pair and return per-pair results.

    Exceptions raised by transform_fn are caught and recorded as failures so
    that a single bad pair does not abort the whole evaluation.

    Returns:
        {
            'pairs':       list of per-pair result dicts
                           {'correct': bool, 'predicted': Grid,
                            'expected': Grid, 'error': str | None}
            'all_correct': True only if every pair is correct
            'n_correct':   number of correct pairs
            'n_total':     total number of pairs
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
    """Load a task JSON file and evaluate it with transform_fn.

    The returned dict is the same as evaluate_task() with an added 'task_id'
    key set to the file stem (e.g. '007bbfb7').
    """
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
    """Score transform_fn against every task file in a directory.

    Args:
        tasks_dir:    Directory containing .json task files.
        transform_fn: Function Grid → Grid to evaluate.
        limit:        If set, only evaluate the first N tasks (useful for
                      quick smoke-tests without running the full suite).
        verbose:      Print a one-line result per task plus a final summary.

    Returns:
        {
            'n_tasks':       int   — total tasks evaluated
            'n_solved':      int   — tasks where all pairs were correct
            'accuracy':      float — n_solved / n_tasks
            'pair_accuracy': float — correct pairs / total pairs across all tasks
            'results':       list[dict] — per-task evaluate_task_file results
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
    """Pretty-print the failure details for a single evaluated task result dict."""
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
