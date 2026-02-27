"""CLI for the multi-agent Orchestrator on ARC-AGI tasks.

Each agent role (Hypothesizer, Coder, Critic) uses a different model,
matching model strengths to task requirements.

Default loadout (optimised for 48 GB unified memory, ~29 GB footprint):
  Hypothesizer  deepseek-r1:32b        ~20 GB  — distilled Qwen reasoning model
  Coder         qwen2.5-coder:14b       ~9 GB  — instruction-following code gen
  Critic        qwen2.5-coder:14b       ~0 GB  — reuses the Coder model slot

Start Ollama with enough parallelism for concurrent model loading:
  source start_ollama.sh   # sets OLLAMA_NUM_PARALLEL=3, OLLAMA_MAX_VRAM

Examples
--------
# Single task with default models
python run_multi_agent.py --task data/data/training/007bbfb7.json

# Explicit per-role models
python run_multi_agent.py \\
    --task data/data/training/007bbfb7.json \\
    --hypothesizer-model deepseek-r1:32b \\
    --coder-model qwen2.5-coder:14b \\
    --critic-model qwen2.5-coder:14b

# Run over a directory, limit to 20 tasks
python run_multi_agent.py \\
    --dir data/data/training \\
    --limit 20
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
        "--hypothesizer-model", default="deepseek-r1:32b",
        dest="hypothesizer_model", metavar="MODEL",
        help="Model for the Hypothesizer (spatial reasoning).  "
             "Default: deepseek-r1:32b (~20 GB Q4_K_M).",
    )
    p.add_argument(
        "--coder-model", default="qwen2.5-coder:14b",
        dest="coder_model", metavar="MODEL",
        help="Model for the Coder (code generation).  "
             "Default: qwen2.5-coder:14b (~9 GB Q4_K_M).",
    )
    p.add_argument(
        "--critic-model", default="deepseek-r1:14b",
        dest="critic_model", metavar="MODEL",
        help="Model for the Critic (failure diagnosis).  "
             "Default: deepseek-r1:14b (~9 GB Q4_K_M).",
    )

    # Per-role temperatures
    p.add_argument(
        "--hypothesizer-temperature", type=float, default=0.6,
        dest="hypothesizer_temperature", metavar="T",
        help="Sampling temperature for the Hypothesizer (default 0.6 — creative reasoning).",
    )
    p.add_argument(
        "--coder-temperature", type=float, default=0.1,
        dest="coder_temperature", metavar="T",
        help="Base sampling temperature for the Coder (default 0.1 — deterministic code).",
    )
    p.add_argument(
        "--critic-temperature", type=float, default=0.2,
        dest="critic_temperature", metavar="T",
        help="Sampling temperature for the Critic (default 0.2 — nuanced analysis).",
    )

    # Per-role token budgets
    p.add_argument(
        "--hypothesizer-max-tokens", type=int, default=32768,
        dest="hypothesizer_max_tokens", metavar="N",
        help="Max tokens for Hypothesizer (default 32768 — needs long reasoning chains).",
    )
    p.add_argument(
        "--coder-max-tokens", type=int, default=8192,
        dest="coder_max_tokens", metavar="N",
        help="Max tokens for Coder (default 8192 — code only, no reasoning overhead).",
    )
    p.add_argument(
        "--critic-max-tokens", type=int, default=4096,
        dest="critic_max_tokens", metavar="N",
        help="Max tokens for Critic (default 4096 — concise analysis + routing feedback).",
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
        "--max-rehypothesizing", type=int, default=2, dest="max_rehypothesizing", metavar="N",
        help="Extra Hypothesizer rounds seeded with Critic feedback (default 2).",
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
        hypothesizer_temperature=args.hypothesizer_temperature,
        coder_temperature=args.coder_temperature,
        critic_temperature=args.critic_temperature,
        hypothesizer_max_tokens=args.hypothesizer_max_tokens,
        coder_max_tokens=args.coder_max_tokens,
        critic_max_tokens=args.critic_max_tokens,
        n_hypotheses=args.n_hypotheses,
        max_retries=args.max_retries,
        max_rehypothesizing=args.max_rehypothesizing,
        timeout=args.timeout,
        debug=args.debug,
    )

    print(f"Backend:  {orch.backend}")
    print(f"  Hypothesizer: {orch.hypothesizer_model}  (temp={orch.hypothesizer_temperature}, max_tokens={orch.hypothesizer_max_tokens})")
    print(f"  Coder:        {orch.coder_model}  (temp={orch.coder_temperature}, max_tokens={orch.coder_max_tokens})")
    print(f"  Critic:       {orch.critic_model}  (temp={orch.critic_temperature}, max_tokens={orch.critic_max_tokens})")
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
