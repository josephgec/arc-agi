#!/usr/bin/env python3
"""Run the single-agent baseline on ARC-AGI tasks.

Usage:
  # Solve one task with Ollama/deepseek-r1 (default):
  python run_baseline.py --task data/data/training/007bbfb7.json

  # Solve first 10 tasks:
  python run_baseline.py --dir data/data/training --limit 10

  # Use Anthropic backend:
  python run_baseline.py --task data/data/training/007bbfb7.json --backend anthropic

  # Use a different local model:
  python run_baseline.py --task data/data/training/007bbfb7.json --model qwen2.5-coder:32b
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure the project root is importable regardless of the working directory.
sys.path.insert(0, str(Path(__file__).parent))

from arc.grid import load_task
from arc.visualize import print_task
from agents.single_agent import SingleAgent


def solve_one(task_path: Path, agent: SingleAgent, verbose: bool = True) -> dict:
    """Solve a single task file and return the result dict.

    Args:
        task_path: Path to the .json task file.
        agent:     Configured SingleAgent instance.
        verbose:   If True, print the task visualisation and a per-attempt summary.

    Returns:
        The result dict from agent.solve() augmented with 'task_id' and 'elapsed_s'.
    """
    task_id = task_path.stem
    task = load_task(task_path)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Task: {task_id}  ({len(task['train'])} training pairs)")
        print_task(task)

    t0 = time.time()
    result = agent.solve(task, task_id=task_id)
    elapsed = time.time() - t0

    result["task_id"] = task_id
    result["elapsed_s"] = round(elapsed, 2)

    if verbose:
        status = "SOLVED" if result["success"] else "FAILED"
        print(f"\n[{status}] {task_id}  ({result['n_attempts']} attempt(s), {elapsed:.1f}s)")
        for i, entry in enumerate(result["log"]):
            correct = entry.get("n_correct", 0)
            total = entry.get("n_total", "?")
            err = entry.get("error", "")
            print(f"  Attempt {i + 1}: {correct}/{total} pairs correct  {err or ''}")
        if result["code"] and result["success"]:
            print("\nWinning code:")
            print("-" * 40)
            print(result["code"])
            print("-" * 40)

    return result


def save_log(results: list[dict], log_dir: Path) -> Path:
    """Serialise results to a timestamped JSON file inside log_dir.

    Numpy arrays in the log entries are stripped before serialisation.
    Returns the path to the written file.
    """
    log_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"baseline_{ts}.json"

    # Drop the per-attempt log list; it may contain Grid objects.
    serialisable = [
        {k: v for k, v in r.items() if k != "log"}
        for r in results
    ]

    with open(log_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC-AGI single-agent baseline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", type=Path, help="Path to a single task JSON file")
    group.add_argument("--dir",  type=Path, help="Directory of task JSON files")
    parser.add_argument("--limit",   type=int,   default=None,
                        help="Max number of tasks to run")
    parser.add_argument("--backend", default="ollama",
                        choices=["ollama", "anthropic"],
                        help="LLM backend (default: ollama)")
    parser.add_argument("--model",   default=None,
                        help="Model name override (default depends on backend)")
    parser.add_argument("--retries", type=int,   default=3,
                        help="Max self-correction retries (default: 3)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="Seconds to wait per model call, ollama only (default: 600)")
    parser.add_argument("--quiet",   action="store_true",
                        help="Suppress per-task details")
    parser.add_argument("--debug",   action="store_true",
                        help="Print raw model responses for troubleshooting")
    args = parser.parse_args()

    agent = SingleAgent(
        backend=args.backend,
        model=args.model,
        max_retries=args.retries,
        timeout=args.timeout,
        debug=args.debug,
    )
    results: list[dict] = []

    if args.task:
        result = solve_one(args.task, agent, verbose=not args.quiet)
        results.append(result)
    else:
        task_files = sorted(args.dir.glob("*.json"))
        if args.limit:
            task_files = task_files[: args.limit]

        model_label = args.model or f"default ({args.backend})"
        print(
            f"Running baseline on {len(task_files)} tasks  "
            f"(backend={args.backend}, model={model_label}, retries={args.retries})"
        )

        n_solved = 0
        for i, path in enumerate(task_files, 1):
            result = solve_one(path, agent, verbose=not args.quiet)
            results.append(result)
            n_solved += int(result["success"])
            pct = n_solved / i * 100
            print(f"Progress: {i}/{len(task_files)}  solved={n_solved} ({pct:.1f}%)")

        print(f"\nFinal: solved {n_solved}/{len(task_files)} ({n_solved / len(task_files) * 100:.1f}%)")

    log_path = save_log(results, Path("logs"))
    print(f"\nLog saved to {log_path}")


if __name__ == "__main__":
    main()
