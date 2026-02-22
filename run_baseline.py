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

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from arc.grid import load_task
from arc.visualize import print_task
from agents.single_agent import SingleAgent


def solve_one(task_path: Path, agent: SingleAgent, verbose: bool = True) -> dict:
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
    log_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"baseline_{ts}.json"

    # Strip numpy arrays from log before serialising
    serialisable = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != "log"}
        serialisable.append(entry)

    with open(log_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC-AGI single-agent baseline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", type=Path, help="Path to a single task JSON file")
    group.add_argument("--dir", type=Path, help="Directory of task JSON files")
    parser.add_argument("--limit", type=int, default=None, help="Max number of tasks to run")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "anthropic"],
                        help="LLM backend (default: ollama)")
    parser.add_argument("--model", default=None,
                        help="Model name (default: deepseek-r1:70b for ollama, claude-sonnet-4-6 for anthropic)")
    parser.add_argument("--retries", type=int, default=3, help="Max self-correction retries")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-task details")
    args = parser.parse_args()

    agent = SingleAgent(backend=args.backend, model=args.model, max_retries=args.retries)
    results: list[dict] = []

    if args.task:
        result = solve_one(args.task, agent, verbose=not args.quiet)
        results.append(result)
    else:
        task_files = sorted(args.dir.glob("*.json"))
        if args.limit:
            task_files = task_files[: args.limit]

        model_label = args.model or f"default ({args.backend})"
        print(f"Running baseline on {len(task_files)} tasks  (backend={args.backend}, model={model_label}, retries={args.retries})")

        n_solved = 0
        for i, path in enumerate(task_files, 1):
            result = solve_one(path, agent, verbose=not args.quiet)
            results.append(result)
            n_solved += int(result["success"])
            pct = n_solved / i * 100
            print(f"Progress: {i}/{len(task_files)}  solved={n_solved} ({pct:.1f}%)")

        print(f"\nFinal: solved {n_solved}/{len(task_files)} ({n_solved/len(task_files)*100:.1f}%)")

    log_path = save_log(results, Path("logs"))
    print(f"\nLog saved to {log_path}")


if __name__ == "__main__":
    main()
