"""CLI for the multi-agent Orchestrator on ARC-AGI tasks.

Each agent role (Hypothesizer, Coder, Critic) can use a different model,
which is the key advantage over the single-agent baseline.  A strong
reasoning model handles hypothesis generation while a coding-focused model
handles implementation — matching model strengths to task requirements.

Examples
--------
# Run with default model for all roles
python run_multi_agent.py --task data/data/training/007bbfb7.json

# Use per-role models (recommended)
python run_multi_agent.py \\
    --task data/data/training/007bbfb7.json \\
    --hypothesizer-model deepseek-r1:32b \\
    --coder-model qwen2.5-coder:7b \\
    --critic-model deepseek-r1:8b

# Run over a directory, limit to 20 tasks
python run_multi_agent.py \\
    --dir data/data/training \\
    --limit 20 \\
    --hypothesizer-model deepseek-r1:32b \\
    --coder-model qwen2.5-coder:7b \\
    --critic-model deepseek-r1:8b
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from agents.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the multi-agent Orchestrator on ARC-AGI tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task selection (one of --task or --dir is required)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--task", metavar="FILE",
        help="Path to a single ARC task JSON file.",
    )
    group.add_argument(
        "--dir", metavar="DIR",
        help="Directory of ARC task JSON files.  All *.json files are run.",
    )
    p.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Maximum number of tasks to run (directory mode only).",
    )

    # Backend
    p.add_argument(
        "--backend", default="ollama", choices=["ollama", "anthropic"],
        help="LLM backend to use.",
    )

    # Model selection — shared fallback + per-role overrides
    p.add_argument(
        "--model", default=None, metavar="MODEL",
        help="Default model for all roles (overridden by per-role flags).",
    )
    p.add_argument(
        "--hypothesizer-model", default=None, dest="hypothesizer_model",
        metavar="MODEL",
        help="Model for the Hypothesizer (spatial reasoning).  "
             "Recommended: deepseek-r1:32b.",
    )
    p.add_argument(
        "--coder-model", default=None, dest="coder_model", metavar="MODEL",
        help="Model for the Coder (code generation).  "
             "Recommended: qwen2.5-coder:7b.",
    )
    p.add_argument(
        "--critic-model", default=None, dest="critic_model", metavar="MODEL",
        help="Model for the Critic (failure diagnosis).  "
             "Recommended: deepseek-r1:8b.",
    )

    # Orchestrator tuning
    p.add_argument(
        "--n-hypotheses", type=int, default=3, dest="n_hypotheses",
        metavar="N",
        help="Number of hypotheses to generate per task.",
    )
    p.add_argument(
        "--max-retries", type=int, default=2, dest="max_retries", metavar="N",
        help="Additional Coder attempts per hypothesis after the first.",
    )
    p.add_argument(
        "--timeout", type=float, default=300.0, metavar="SECS",
        help="Seconds per LLM call (Ollama only).",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Print per-state diagnostic output.",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-task output (summary only).",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------

def _load_task(path: pathlib.Path) -> dict:
    with open(path) as f:
        task = json.load(f)
    for split in ("train", "test"):
        for pair in task.get(split, []):
            for key in ("input", "output"):
                if key in pair:
                    pair[key] = np.array(pair[key], dtype=np.int32)
    return task


def _run_task(orch: Orchestrator, path: pathlib.Path, quiet: bool) -> dict:
    task   = _load_task(path)
    name   = path.stem
    n_train = len(task.get("train", []))

    if not quiet:
        print(f"\nTask: {name}  ({n_train} train pairs)")

    result = orch.solve(task)

    if not quiet:
        status   = "SOLVED" if result["success"] else "failed"
        n_cands  = len(result["candidates"])
        tc       = result["test_correct"]
        tc_str   = f"  test={'correct' if tc else 'wrong'}" if tc is not None else ""
        print(f"  {status}  candidates={n_cands}{tc_str}")
        if result["code"] and result["success"]:
            print(f"  --- solution ---\n{result['code']}\n  ----------------")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    orch = Orchestrator(
        backend=args.backend,
        model=args.model,
        hypothesizer_model=args.hypothesizer_model,
        coder_model=args.coder_model,
        critic_model=args.critic_model,
        n_hypotheses=args.n_hypotheses,
        max_retries=args.max_retries,
        timeout=args.timeout,
        debug=args.debug,
    )

    print(f"Backend:  {orch.backend}")
    print(f"  Hypothesizer: {orch.hypothesizer_model}")
    print(f"  Coder:        {orch.coder_model}")
    print(f"  Critic:       {orch.critic_model}")
    print(f"  n_hypotheses={orch.n_hypotheses}  max_retries={orch.max_retries}")

    if args.task:
        _run_task(orch, pathlib.Path(args.task), quiet=args.quiet)
        return

    # Directory mode
    task_files = sorted(pathlib.Path(args.dir).glob("*.json"))
    if args.limit:
        task_files = task_files[: args.limit]

    solved = 0
    test_correct = 0
    has_ground_truth = 0

    for tf in task_files:
        result = _run_task(orch, tf, quiet=args.quiet)
        if result["success"]:
            solved += 1
        if result["test_correct"] is not None:
            has_ground_truth += 1
            if result["test_correct"]:
                test_correct += 1

    total = len(task_files)
    print(f"\n{'='*50}")
    print(f"Tasks:        {total}")
    print(f"Train solved: {solved}/{total}  ({100*solved//total if total else 0}%)")
    if has_ground_truth:
        print(
            f"Test correct: {test_correct}/{has_ground_truth}"
            f"  ({100*test_correct//has_ground_truth}%)"
        )


if __name__ == "__main__":
    main()
