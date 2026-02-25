"""Multi-agent orchestrator for ARC-AGI puzzles.

Coordinates three specialized agents in a feedback loop:

  Hypothesizer → generates 3 competing natural-language transformation rules
  Coder        → translates one rule into executable Python DSL code
  Critic       → diagnoses failures and routes feedback to the right agent

The loop runs up to ``max_cycles`` total agent calls.  On each cycle either:
  - A correct solution is found (success), or
  - The Critic routes to the Hypothesizer (try next/new hypothesis), or
  - The Critic routes to the Coder (fix the current implementation).
"""
from __future__ import annotations

import re

import numpy as np

from arc import sandbox
from arc.grid import Grid, grids_equal
from agents.llm_client import LLMClient
from agents.roles import Hypothesizer, Coder, Critic, ROUTE_HYPOTHESIZER


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_LARGE_GRID_CELL_THRESHOLD = 300
_LARGE_GRID_MAX_PAIRS      = 2

_COLOR_NAMES = {
    0: "black", 1: "blue",   2: "red",    3: "green",  4: "yellow",
    5: "grey",  6: "fuschia", 7: "orange", 8: "azure",  9: "maroon",
}


# ---------------------------------------------------------------------------
# Grid / task formatting helpers
# ---------------------------------------------------------------------------

def _grid_to_str(grid) -> str:
    return (
        "["
        + ", ".join("[" + ", ".join(str(v) for v in row) + "]" for row in grid.tolist())
        + "]"
    )


def _block_analysis(inp, out) -> str | None:
    ih, iw = inp.shape
    oh, ow = out.shape
    if oh % ih != 0 or ow % iw != 0:
        return None
    br, bc = oh // ih, ow // iw
    lines  = [f"  (Output divided into {br}×{bc} blocks, each {ih}×{iw}:)"]
    for r in range(br):
        for c in range(bc):
            block    = out[r * ih:(r + 1) * ih, c * iw:(c + 1) * iw]
            cell_val = inp[r, c] if r < ih and c < iw else "?"
            if (block == 0).all():
                content = "all zeros"
            else:
                content = _grid_to_str(block)
                if block.shape == inp.shape and (block == inp).all():
                    content += " (= input)"
            lines.append(f"  block({r},{c}): input[{r}][{c}]={cell_val} → {content}")
    return "\n".join(lines)


def _format_training_examples(task: dict) -> str:
    """Format training pairs as a compact reference for the Coder.

    Unlike _format_task_description (which includes the test input and is
    written for the Hypothesizer), this is written for the Coder: it shows
    only the training input→output pairs so the model can verify its
    implementation mentally before returning code.
    """
    lines = ["Training examples (use these to verify your implementation):"]
    for i, pair in enumerate(task["train"]):
        inp, out = pair["input"], pair["output"]
        ih, iw   = inp.shape
        oh, ow   = out.shape
        lines.append(f"Example {i + 1}: input ({ih}×{iw}) → output ({oh}×{ow})")
        lines.append(f"  Input:  {_grid_to_str(inp)}")
        lines.append(f"  Output: {_grid_to_str(out)}")
    return "\n".join(lines)


def _format_task_description(task: dict) -> str:
    """Format training pairs and test input as a grid description for the Hypothesizer."""
    pairs    = task["train"]
    max_cells = max(max(p["input"].size, p["output"].size) for p in pairs)
    if max_cells > _LARGE_GRID_CELL_THRESHOLD:
        pairs = pairs[:_LARGE_GRID_MAX_PAIRS]
        note  = f"(Note: grids are large; showing {len(pairs)} of {len(task['train'])} training pairs.)\n"
    else:
        note = ""

    lines = [f"Here is an ARC-AGI puzzle.\n{note}"]
    for i, pair in enumerate(pairs):
        inp, out = pair["input"], pair["output"]
        ih, iw   = inp.shape
        oh, ow   = out.shape
        lines.append(f"### Training pair {i + 1}")
        lines.append(f"Input  ({ih}×{iw}):\n{_grid_to_str(inp)}")
        lines.append(f"Output ({oh}×{ow}):\n{_grid_to_str(out)}")
        ba = _block_analysis(inp, out)
        if ba:
            lines.append(ba)
        lines.append("")

    test_inp = task["test"][0]["input"]
    th, tw   = test_inp.shape
    lines.append(f"### Test input ({th}×{tw}):\n{_grid_to_str(test_inp)}")
    lines.append("\nStudy the training pairs and identify the transformation rule.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Code extraction / response cleaning
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _truncate_to_valid_function(text: str) -> str:
    """Keep lines up to and including the end of the first complete function body."""
    lines   = text.splitlines()
    in_func = False
    result  = []
    for line in lines:
        if line.startswith("def "):
            in_func = True
        if in_func:
            result.append(line)
            stripped = line.rstrip()
            if stripped and not stripped[0].isspace() and not stripped.startswith("def "):
                break
    return "\n".join(result) if result else text


def _extract_code(text: str) -> str | None:
    """Extract the first ```python block from text. Falls back to bare def."""
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        if "def " in candidate:
            return candidate
    m = re.search(r"(def \w+\(.*)", text)
    if m:
        return _truncate_to_valid_function(text[m.start():])
    return None


# ---------------------------------------------------------------------------
# Hypothesis parsing
# ---------------------------------------------------------------------------

def _parse_hypotheses(text: str) -> list[str]:
    """Split the Hypothesizer's output into individual hypothesis strings.

    Splits on lines beginning with a digit followed by a period (1., 2., 3.).
    Returns the original text as a single-item list when no numbered split found.
    """
    text  = _strip_thinking(text)
    parts = re.split(r"(?m)^(?=[1-9]\.\s)", text.strip())
    hypotheses = [p.strip() for p in parts if p.strip()]
    return hypotheses if len(hypotheses) >= 2 else [text.strip()]


# ---------------------------------------------------------------------------
# Error / diff formatting for the Critic
# ---------------------------------------------------------------------------

def _format_error_info(eval_result: dict) -> str:
    """Summarise evaluation failures as a human-readable string."""
    lines = [f"{eval_result['n_correct']}/{eval_result['n_total']} training pairs correct."]
    for i, pair in enumerate(eval_result["pairs"]):
        if pair["error"]:
            lines.append(f"Pair {i + 1} error: {pair['error']}")
        elif not pair["correct"]:
            lines.append(f"Pair {i + 1}: produced wrong output (no exception).")
    return "\n".join(lines)


def _diff_summary(expected, predicted, max_show: int = 12) -> str:
    """Return a human-readable cell-level diff between two grids."""
    if predicted is None:
        return "(no output produced)"
    if expected.shape != predicted.shape:
        return f"Shape mismatch: expected {expected.shape}, got {predicted.shape}."
    diffs = list(zip(*np.where(expected != predicted)))
    if not diffs:
        return "(no differences)"
    lines = []
    for r, c in diffs[:max_show]:
        ev = int(expected[r, c])
        pv = int(predicted[r, c])
        lines.append(
            f"  [{r},{c}] expected {ev} ({_COLOR_NAMES.get(ev, ev)}), "
            f"got {pv} ({_COLOR_NAMES.get(pv, pv)})"
        )
    if len(diffs) > max_show:
        lines.append(f"  … and {len(diffs) - max_show} more differences")
    return "\n".join(lines)


def _format_diff(eval_result: dict) -> str:
    """Collect per-pair diffs into a single string."""
    parts = []
    for i, pair in enumerate(eval_result["pairs"]):
        if not pair["correct"]:
            diff = _diff_summary(pair["expected"], pair["predicted"])
            parts.append(f"Pair {i + 1}:\n{diff}")
    return "\n\n".join(parts) if parts else "(all pairs correct)"


# ---------------------------------------------------------------------------
# MultiAgent orchestrator
# ---------------------------------------------------------------------------

class MultiAgent:
    """Orchestrates Hypothesizer, Coder, and Critic to solve an ARC task.

    The loop structure per cycle set:
      1. Hypothesizer generates 3 hypotheses (one agent call).
      2. For each hypothesis:
         a. Coder generates code (one agent call).
         b. Evaluate against training pairs.
         c. If correct → success.
         d. Critic diagnoses failure (one agent call).
            - HYPOTHESIZER route → move to next hypothesis.
            - CODER route        → retry same hypothesis with feedback.

    Args:
        backend:    'ollama' or 'anthropic'.
        model:      Model name override.
        timeout:    Seconds to wait per LLM call (Ollama only).
        debug:      Print diagnostic output to stdout.
        max_cycles: Maximum total agent calls before giving up.
    """

    def __init__(
        self,
        backend:    str   = "ollama",
        model:      str | None = None,
        timeout:    float = 120.0,
        debug:      bool  = False,
        max_cycles: int   = 9,
    ) -> None:
        client             = LLMClient(backend=backend, model=model, timeout=timeout, debug=debug)
        self._hypothesizer = Hypothesizer(client)
        self._coder        = Coder(client)
        self._critic       = Critic(client)
        self.max_cycles    = max_cycles
        self.debug         = debug
        self.backend       = client.backend
        self.model         = client.model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, task: dict) -> dict:
        """Run the multi-agent loop and return a result dict.

        Returns:
            {
                'success':      bool
                'test_correct': bool | None  — None when test output is absent
                'code':         str  | None  — best code found (even if not fully correct)
                'n_cycles':     int          — total agent calls made
                'log':          list[dict]   — per-call details
            }
        """
        log:            list[dict] = []
        best_code:      str | None = None
        best_n_correct: int        = -1
        cycle:          int        = 0

        test_pair             = task.get("test", [{}])[0]
        has_test_ground_truth = "output" in test_pair

        task_description: str        = _format_task_description(task)
        training_examples: str       = _format_training_examples(task)
        hypotheses:        list[str] = []
        hyp_index:         int       = 0
        hyp_feedback:      str | None = None
        coder_feedback:    str | None = None
        prev_n_correct:    int        = -1
        no_improve_count:  int        = 0
        coder_attempt:     int        = 0  # Coder calls on the current hypothesis

        while cycle < self.max_cycles:

            # --- Hypothesizer: generate (or regenerate) hypotheses ---
            if not hypotheses or hyp_index >= len(hypotheses):
                cycle += 1
                if cycle > self.max_cycles:
                    break
                try:
                    hyp_response = self._hypothesizer.generate(task_description, hyp_feedback)
                except Exception as e:
                    log.append({"cycle": cycle, "agent": "hypothesizer", "error": str(e)})
                    break

                hyp_feedback = None
                hypotheses   = _parse_hypotheses(hyp_response)
                hyp_index    = 0

                log.append({
                    "cycle": cycle, "agent": "hypothesizer",
                    "n_hypotheses": len(hypotheses),
                })
                if self.debug:
                    print(f"[debug] Hypothesizer: {len(hypotheses)} hypothesis(es)")

            # Reset per-hypothesis attempt tracking when we switch hypothesis.
            if not hypotheses or hyp_index == 0 or (
                log and log[-1].get("agent") == "hypothesizer"
            ):
                coder_attempt    = 0
                prev_n_correct   = -1
                no_improve_count = 0

            current_hypothesis = hypotheses[hyp_index]
            coder_attempt += 1

            # Temperature diversity: greedy on first attempt, ramp up on retries.
            temperature = min((coder_attempt - 1) * 0.3, 0.8)

            # --- Coder: translate hypothesis to code ---
            cycle += 1
            if cycle > self.max_cycles:
                break
            try:
                code_response = self._coder.generate(
                    current_hypothesis, coder_feedback,
                    training_context=training_examples,
                    temperature=temperature,
                )
            except Exception as e:
                log.append({
                    "cycle": cycle, "agent": "coder",
                    "hypothesis_index": hyp_index, "error": str(e),
                })
                hyp_index     += 1
                coder_feedback = None
                coder_attempt  = 0
                continue

            coder_feedback = None  # consumed; reset for next round

            clean  = _strip_thinking(code_response)
            code   = _extract_code(clean) or _extract_code(code_response)

            if self.debug:
                print(
                    f"[debug] Coder (hyp {hyp_index}): "
                    f"response={len(code_response)} chars, code={code is not None}"
                )

            if code is None:
                log.append({
                    "cycle": cycle, "agent": "coder",
                    "hypothesis_index": hyp_index, "error": "no_code_block",
                })
                hyp_index += 1
                continue

            # --- Evaluate ---
            eval_result = sandbox.evaluate_code(code, task)
            n_correct   = eval_result["n_correct"]

            if n_correct > best_n_correct:
                best_n_correct = n_correct
                best_code      = code

            log.append({
                "cycle":            cycle,
                "agent":            "coder",
                "hypothesis_index": hyp_index,
                "n_correct":        n_correct,
                "n_total":          eval_result["n_total"],
                "all_correct":      eval_result["all_correct"],
            })

            if eval_result["all_correct"]:
                return {
                    "success":      True,
                    "test_correct": self._evaluate_test(best_code, test_pair) if has_test_ground_truth else None,
                    "code":         best_code,
                    "n_cycles":     cycle,
                    "log":          log,
                }

            # Track whether the hypothesis is making any progress.
            if n_correct <= prev_n_correct:
                no_improve_count += 1
            else:
                no_improve_count = 0
            prev_n_correct = n_correct

            # Skip the Critic when stuck (saves a cycle and moves on faster).
            _hypothesis_stuck = (n_correct == 0 and no_improve_count >= 2)
            if _hypothesis_stuck:
                if self.debug:
                    print(f"[debug] Stuck at 0/{eval_result['n_total']} — skipping Critic, next hyp")
                hyp_index    += 1
                coder_attempt = 0
                continue

            # --- Critic: diagnose failure ---
            cycle += 1
            if cycle > self.max_cycles:
                break
            try:
                critic_result = self._critic.analyze(
                    current_hypothesis, code,
                    _format_error_info(eval_result),
                    _format_diff(eval_result),
                )
            except Exception as e:
                log.append({"cycle": cycle, "agent": "critic", "error": str(e)})
                hyp_index    += 1
                coder_attempt = 0
                continue

            log.append({
                "cycle":    cycle,
                "agent":    "critic",
                "route":    critic_result["route"],
                "feedback": critic_result["feedback"],
            })
            if self.debug:
                print(f"[debug] Critic → {critic_result['route']}")

            if critic_result["route"] == ROUTE_HYPOTHESIZER:
                hyp_feedback  = critic_result["feedback"]
                hyp_index    += 1      # advance; triggers regeneration when all exhausted
                coder_attempt = 0
            else:
                coder_feedback = critic_result["feedback"]  # stay on same hypothesis

        return {
            "success":      False,
            "test_correct": (
                self._evaluate_test(best_code, test_pair)
                if has_test_ground_truth and best_code else None
            ),
            "code":         best_code,
            "n_cycles":     cycle,
            "log":          log,
        }

    def predict(self, task: dict) -> Grid | None:
        """Solve a task and return the predicted output grid, or None on failure."""
        result = self.solve(task)
        if not result["code"]:
            return None
        out, _ = sandbox.execute(result["code"], task["test"][0]["input"])
        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_test(self, code: str, test_pair: dict) -> bool:
        """Run code on the held-out test input and compare against ground truth."""
        out, err = sandbox.execute(code, test_pair["input"])
        if err or out is None:
            return False
        return grids_equal(out, test_pair["output"])
