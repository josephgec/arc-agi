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
    """Keep lines up to the end of the first complete function body.

    Stops (without including) the first non-empty, non-indented, non-'def'
    line encountered after the function starts.  This strips reasoning prose
    that reasoning models sometimes write after `return` inside a code fence.
    """
    lines   = text.splitlines()
    in_func = False
    result  = []
    for line in lines:
        if line.startswith("def "):
            in_func = True
        if in_func:
            stripped = line.rstrip()
            # A non-empty, column-0 line that isn't another def is outside the
            # function body — stop here and do NOT include this line.
            if stripped and not stripped[0].isspace() and not stripped.startswith("def "):
                break
            result.append(line)
    return "\n".join(result).rstrip() if result else text


def _extract_code(text: str) -> str | None:
    """Extract executable Python code from a model response.

    Strips ``<think>`` blocks first (belt-and-suspenders — callers may or may
    not have done this already), then applies a five-step cascade with
    increasing tolerance for missing formatting.

    Critically, when multiple matches exist (reasoning models write draft code
    inside their chain-of-thought before producing the final answer), the
    **last** match is preferred, as it is most likely to be the converged
    solution rather than an intermediate draft.

    Steps:
    1. Fenced ```python … ``` block — last occurrence preferred.
    2. Generic ``` … ``` block that contains a function definition.
    3. Bare ``def transform(`` — last occurrence preferred.
    4. Any bare ``def <name>(`` — last occurrence.
    5. ``import numpy`` followed by ``def`` (no fence at all).

    After extraction, _truncate_to_valid_function strips trailing prose that
    reasoning models sometimes write after the ``return`` statement.
    """
    # Strip thinking tags first — idempotent if caller already stripped them,
    # but essential when a raw-thinking response is passed directly.
    text = _strip_thinking(text)
    if not text.strip():
        return None

    # ── 1. ```python … ``` — last block preferred ───────────────────────────
    matches = list(re.finditer(r"```python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE))
    if matches:
        candidate = matches[-1].group(1).strip()
        return _truncate_to_valid_function(candidate) if "def " in candidate else candidate

    # ── 2. ``` … ``` with a def — last qualifying block ─────────────────────
    for m in reversed(list(re.finditer(r"```\s*(.*?)\s*```", text, re.DOTALL))):
        candidate = m.group(1).strip()
        if "def " in candidate:
            return _truncate_to_valid_function(candidate)

    # ── 3. Bare def transform( — last occurrence ─────────────────────────────
    # Prefer the specific function name to avoid capturing helper functions.
    all_transforms = list(re.finditer(r"def transform\(", text))
    if all_transforms:
        return _truncate_to_valid_function(text[all_transforms[-1].start():])

    # ── 4. Any bare def — last occurrence ───────────────────────────────────
    all_defs = list(re.finditer(r"def \w+\(", text))
    if all_defs:
        return _truncate_to_valid_function(text[all_defs[-1].start():])

    # ── 5. import numpy … def (no fence) ────────────────────────────────────
    # deepseek-r1 in thinking-only mode sometimes produces code starting with
    # numpy imports but without a code fence.  numpy is pre-imported in the
    # sandbox, so the redundant import is harmless.
    if "import numpy" in text and "def " in text:
        start = text.rfind("import numpy")
        return _truncate_to_valid_function(text[start:])

    return None


# ---------------------------------------------------------------------------
# Hypothesis parsing
# ---------------------------------------------------------------------------

_MIN_HYPOTHESIS_CHARS = 80  # filter out short noise fragments from reasoning


def _parse_hypotheses(text: str, max_n: int | None = None) -> list[str]:
    """Split the Hypothesizer's output into individual hypothesis strings.

    Strategy:
    1. Try paragraph-level splitting — only keeps items that begin a new blank-
       line-separated paragraph with a digit+period.  This avoids matching
       numbered sub-steps that reasoning models embed inside prose.
    2. Fall back to line-level splitting when paragraph splitting finds < 2 items.
    3. Filter out short fragments (< _MIN_HYPOTHESIS_CHARS) that are noise.
    4. Cap at max_n when provided.

    Returns the original text as a single-item list when no numbered items found.
    """
    text    = _strip_thinking(text)
    stripped = text.strip()

    # ── Strategy 1: paragraph-level (blank-line delimited) ──────────────────
    paragraphs  = re.split(r"\n\s*\n", stripped)
    hyp_paras   = [p.strip() for p in paragraphs
                   if re.match(r"^[1-9]\.\s", p.strip())]
    if len(hyp_paras) >= 2:
        hypotheses = hyp_paras
    else:
        # ── Strategy 2: line-level fallback ─────────────────────────────────
        parts      = re.split(r"(?m)^(?=[1-9]\.\s)", stripped)
        hypotheses = [p.strip() for p in parts if p.strip()]
        if len(hypotheses) < 2:
            return [stripped] if stripped else []

    # Filter noise (short fragments are almost always reasoning fragments, not
    # full algorithm descriptions).
    hypotheses = [h for h in hypotheses if len(h) >= _MIN_HYPOTHESIS_CHARS]

    if max_n is not None:
        hypotheses = hypotheses[:max_n]

    return hypotheses if hypotheses else [stripped]


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
    """Collect per-pair diffs into a single string.

    For the first failing pair, also shows the full predicted and expected
    grids (capped at 200 cells each) so the Critic can spot high-level
    patterns such as "output equals input" or "output is a partial mapping".
    """
    _FULL_GRID_CELL_LIMIT = 200
    parts = []
    first_fail = True
    for i, pair in enumerate(eval_result["pairs"]):
        if not pair["correct"]:
            diff = _diff_summary(pair["expected"], pair["predicted"])
            section = f"Pair {i + 1}:\n{diff}"
            if first_fail and pair["predicted"] is not None:
                exp = pair["expected"]
                pred = pair["predicted"]
                if exp.size <= _FULL_GRID_CELL_LIMIT:
                    section += (
                        f"\n  Full expected:  {_grid_to_str(exp)}"
                        f"\n  Full predicted: {_grid_to_str(pred)}"
                    )
                first_fail = False
            parts.append(section)
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
        backend:                  'ollama' or 'anthropic'.
        model:                    Fallback model for any role that doesn't specify one.
        hypothesizer_model:       Model for the Hypothesizer role.
        coder_model:              Model for the Coder role.
        critic_model:             Model for the Critic role.
        hypothesizer_temperature: Sampling temperature for the Hypothesizer (default 0.6).
        coder_temperature:        Base sampling temperature for the Coder (default 0.1).
        critic_temperature:       Sampling temperature for the Critic (default 0.2).
        timeout:                  Seconds to wait per LLM call (Ollama only).
        debug:                    Print diagnostic output to stdout.
        max_cycles:               Maximum total agent calls before giving up.
    """

    def __init__(
        self,
        backend:                  str        = "ollama",
        model:                    str | None = None,
        hypothesizer_model:       str | None = None,
        coder_model:              str | None = None,
        critic_model:             str | None = None,
        hypothesizer_temperature: float      = 0.6,
        coder_temperature:        float      = 0.1,
        critic_temperature:       float      = 0.2,
        hypothesizer_max_tokens:  int        = 32768,
        coder_max_tokens:         int        = 8192,
        critic_max_tokens:        int        = 16384,
        timeout:                  float      = 120.0,
        debug:                    bool       = False,
        max_cycles:               int        = 9,
    ) -> None:
        def _make_client(role_model: str | None, temperature: float, max_tokens: int) -> LLMClient:
            return LLMClient(
                backend=backend,
                model=role_model or model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                debug=debug,
            )

        hyp_client = _make_client(hypothesizer_model, hypothesizer_temperature, hypothesizer_max_tokens)
        cod_client = _make_client(coder_model,        coder_temperature,        coder_max_tokens)
        cri_client = _make_client(critic_model,       critic_temperature,       critic_max_tokens)

        self._hypothesizer            = Hypothesizer(hyp_client)
        self._coder                   = Coder(cod_client)
        self._critic                  = Critic(cri_client)
        self.max_cycles               = max_cycles
        self.debug                    = debug
        self.backend                  = backend
        self.hypothesizer_model       = hyp_client.model
        self.coder_model              = cod_client.model
        self.critic_model             = cri_client.model
        self.hypothesizer_temperature = hypothesizer_temperature
        self.coder_temperature        = coder_temperature
        self.critic_temperature       = critic_temperature
        self.hypothesizer_max_tokens  = hypothesizer_max_tokens
        self.coder_max_tokens         = coder_max_tokens
        self.critic_max_tokens        = critic_max_tokens
        self.model                    = self.hypothesizer_model  # backward-compat alias

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
                hypotheses   = _parse_hypotheses(hyp_response, max_n=3)
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

            # Temperature diversity: start from role baseline, ramp up on retries.
            temperature = min(self.coder_temperature + (coder_attempt - 1) * 0.3, 0.9)

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
