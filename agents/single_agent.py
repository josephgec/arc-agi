"""Single-agent baseline for ARC-AGI.

Solving flow per task:
  1. Format all training pairs into a prompt.
  2. Ask the model to write `def transform(grid: np.ndarray) -> np.ndarray`.
  3. Extract the Python function from the response and execute it.
  4. Evaluate the function against all training pairs.
  5. If any pair is wrong, feed pixel-level error details back and ask for a fix.
  6. Repeat up to max_retries times with increasing temperature for diversity.

Supported backends:
  - "ollama"    : local Ollama server (free, default model: deepseek-r1:32b)
  - "anthropic" : Anthropic API (requires credits, default model: claude-sonnet-4-6)
"""
from __future__ import annotations

import re
import textwrap

import numpy as np

from arc.grid import Grid, grids_equal, COLOR_NAMES
from arc import sandbox
from agents.llm_client import LLMClient

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# System prompt sent to the model at the start of every conversation.
_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Python programmer solving ARC-AGI puzzles.

    Each puzzle shows input→output grid pairs sharing a hidden rule.
    Grids are 2-D numpy arrays of integers 0-9 (0=black, 1=blue, 2=red, 3=green,
    4=yellow, 5=grey, 6=magenta, 7=orange, 8=azure, 9=maroon).

    DSL helpers (already imported — no extra imports needed):
      tile(grid, n_rows, n_cols)   crop(grid, r1, c1, r2, c2)
      flip(grid, axis)             rotate(grid, n)
      translate(grid, dr, dc)      scale(grid, factor)
      recolor(grid, from_c, to_c)  flood_fill(grid, row, col, color)
      find_objects(grid)           bounding_box(grid)   crop_to_content(grid)
      mask(grid, mask_grid)        overlay(base, top)   np (numpy)

    Useful pattern — "stamp input into output blocks based on cell value":
        h, w = grid.shape
        out = np.zeros((h * h, w * w), dtype=np.int32)
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0:
                    out[r*h:(r+1)*h, c*w:(c+1)*w] = grid  # full grid, not grid[r,c]

    TASK: Figure out the rule, then write ONE Python function in a ```python block.
    Your response MUST end with valid Python code like this:

    ```python
    def transform(grid: np.ndarray) -> np.ndarray:
        # your implementation
        return result.astype(np.int32)
    ```

    STRICT RULES:
    - ONLY output a ```python code block with the function — NO LaTeX, NO grid literals.
    - Signature: def transform(grid: np.ndarray) -> np.ndarray
    - Return a NEW numpy int32 array. Never modify the input.
    - No imports, no print(), no input(), no I/O of any kind.
""").strip()

# Temperature values used across retry attempts (greedy first, then exploratory).
_TEMPERATURE_SCHEDULE = [0.0, 0.4, 0.8, 1.0]


# ---------------------------------------------------------------------------
# Module-level helper functions (pure, no model I/O)
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks produced by deepseek-r1 and similar models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _grid_to_str(grid: Grid) -> str:
    """Render a grid as a compact Python list literal suitable for prompts."""
    rows = []
    for row in grid:
        rows.append("[" + ", ".join(str(int(v)) for v in row) + "]")
    return "[" + ",\n ".join(rows) + "]"


def _diff_summary(expected: Grid, predicted: Grid, max_show: int = 12) -> str:
    """Return a human-readable summary of cell-level differences between two grids.

    Lists up to max_show individual cell mismatches.  If more exist, appends a
    count of the remaining differences.  Returns a shape-mismatch message when
    the grids have different shapes.
    """
    if expected.shape != predicted.shape:
        return f"Shape mismatch: expected {expected.shape}, got {predicted.shape}"

    diffs = []
    it = np.nditer([expected, predicted], flags=["multi_index"])
    while not it.finished:
        r, c = it.multi_index
        ev, pv = int(it[0]), int(it[1])
        if ev != pv:
            ev_name = COLOR_NAMES.get(ev, "unknown")
            pv_name = COLOR_NAMES.get(pv, "unknown")
            diffs.append(
                f"  ({r},{c}) expected {ev} ({ev_name}) got {pv} ({pv_name})"
            )
        it.iternext()

    if not diffs:
        return "  (no differences)"

    extra = f"\n  ... and {len(diffs) - max_show} more" if len(diffs) > max_show else ""
    return "\n".join(diffs[:max_show]) + extra


# ---------------------------------------------------------------------------
# SingleAgent
# ---------------------------------------------------------------------------

class SingleAgent:
    """LLM-based agent that iteratively writes and self-corrects ARC transforms.

    The agent formats each task as a prompt, asks the model to write a Python
    `transform` function, evaluates it against training pairs, and feeds
    pixel-level error feedback back to the model for self-correction.

    Supports two backends:
        - "ollama"    — local Ollama server (no API cost)
        - "anthropic" — Anthropic cloud API
    """

    # Threshold (total cells across input + output) above which we cap the
    # number of training pairs included in the prompt.
    _LARGE_GRID_CELL_THRESHOLD = 300
    _LARGE_GRID_MAX_PAIRS = 2

    def __init__(
        self,
        backend: str = "ollama",
        model: str | None = None,
        max_retries: int = 3,
        timeout: float = 120.0,
        debug: bool = False,
    ):
        """Initialise the agent.

        Args:
            backend:     "ollama" (local, free) or "anthropic" (API credits required).
            model:       Model name override.  Defaults to deepseek-r1:8b for ollama
                         and claude-sonnet-4-6 for anthropic.
            max_retries: Number of self-correction attempts after the first try.
            timeout:     Seconds to wait for a single model call (ollama only).
            debug:       Print raw model responses to stdout for troubleshooting.
        """
        self.max_retries = max_retries
        # Per-execution wall-clock limit.  Exposed as an instance attribute so
        # tests can lower it without patching the sandbox module constant.
        self._execution_timeout = sandbox.EXECUTION_TIMEOUT

        # All model I/O is routed through LLMClient — SingleAgent stays focused
        # on prompt engineering and the self-correction loop.
        self._client = LLMClient(
            backend=backend,
            model=model,
            timeout=timeout,
            debug=debug,
        )

        # Pass-through attributes so callers/tests can inspect backend & model
        # without reaching into the private _client.
        self.backend = self._client.backend
        self.model   = self._client.model
        self.debug   = self._client.debug

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, task: dict, task_id: str = "") -> dict:
        """Attempt to solve a task using the self-correction loop.

        Runs up to max_retries + 1 model calls.  The temperature rises each
        attempt to encourage more diverse completions on retries.

        The self-correction loop exclusively uses training pairs, which mirrors
        real ARC evaluation (test ground truth is not available at solve time).
        When a test ground truth *is* present in the task (as it is in the ARC
        training split), the best code is additionally evaluated against it so
        that callers can report honest held-out accuracy.

        Returns:
            {
                'success':      bool       — True if all training pairs passed
                'test_correct': bool|None  — True/False if test ground truth is
                                             present; None when it is absent
                'code':         str|None   — best code found (even if not fully correct)
                'n_attempts':   int        — number of model calls made
                'log':          list[dict] — per-attempt details
            }
        """
        log: list[dict] = []
        messages: list[dict] = [
            {"role": "user", "content": self._format_task_prompt(task)}
        ]

        best_code: str | None = None
        best_n_correct: int = -1

        # Determine whether we can honestly evaluate on the test pair.
        # In the ARC training split, test outputs are present; in the real
        # evaluation split they are not.  We never use this during self-correction
        # (that would be data leakage); it is only checked at the very end.
        test_pair = task.get("test", [{}])[0]
        has_test_ground_truth = "output" in test_pair

        for attempt in range(1, self.max_retries + 2):
            temperature = _TEMPERATURE_SCHEDULE[min(attempt - 1, len(_TEMPERATURE_SCHEDULE) - 1)]

            # --- Call the model ---
            try:
                response_text = self._client.generate(_SYSTEM_PROMPT, messages, temperature)
            except TimeoutError as e:
                log.append({"attempt": attempt, "error": f"timeout: {e}", "n_correct": 0})
                break
            except Exception as e:
                log.append({"attempt": attempt, "error": f"model_error: {e}", "n_correct": 0})
                break

            messages.append({"role": "assistant", "content": response_text})

            # Strip chain-of-thought tags before extracting code.
            # Fall back to the raw response in case the model embedded code inside <think>.
            clean_response = _strip_thinking(response_text)
            code = self._extract_code(clean_response) or self._extract_code(response_text)

            if self.debug:
                print(f"\n--- RAW MODEL RESPONSE (attempt {attempt}) ---")
                print(response_text[:3000])
                print(f"--- END (code extracted: {code is not None}) ---\n")

            # --- No code block found ---
            if code is None:
                log.append({"attempt": attempt, "error": "no_code_block", "n_correct": 0})
                messages.append({
                    "role": "user",
                    "content": "Your response didn't contain a ```python``` code block. Please try again.",
                })
                continue

            # --- Evaluate the extracted code ---
            eval_result = self._evaluate_code(code, task)
            n_correct = eval_result["n_correct"]
            n_total = eval_result["n_total"]

            log.append({
                "attempt":     attempt,
                "code":        code,
                "n_correct":   n_correct,
                "n_total":     n_total,
                "all_correct": eval_result["all_correct"],
            })

            # Track the best solution seen so far
            if n_correct > best_n_correct:
                best_n_correct = n_correct
                best_code = code

            # Success: every training pair is correct
            if eval_result["all_correct"]:
                return {
                    "success":      True,
                    "test_correct": self._evaluate_test(best_code, test_pair) if has_test_ground_truth else None,
                    "code":         best_code,
                    "n_attempts":   attempt,
                    "log":          log,
                }

            # Not yet correct — add error feedback for the next attempt
            if attempt <= self.max_retries:
                messages.append({
                    "role": "user",
                    "content": self._format_error_feedback(eval_result, attempt),
                })

        return {
            "success":      False,
            "test_correct": self._evaluate_test(best_code, test_pair) if has_test_ground_truth and best_code else None,
            "code":         best_code,
            "n_attempts":   self.max_retries + 1,
            "log":          log,
        }

    def predict(self, task: dict) -> Grid | None:
        """Solve a task and apply the best code to the first test input.

        Returns the predicted output Grid, or None if no working code was found
        or if the best code raises an exception on the test input.
        """
        result = self.solve(task)
        if result["code"] is None:
            return None
        test_input = task["test"][0]["input"]
        output, error = self._execute(result["code"], test_input)
        if error:
            return None
        return output

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    @staticmethod
    def _block_analysis(inp: Grid, out: Grid) -> str | None:
        """Compute a block-structure analysis to include in the task prompt.

        If the output dimensions are an exact N×M multiple of the input
        dimensions, the output can be divided into input-sized blocks.
        This helper describes each block's content and which input cell it
        corresponds to, helping the model spot tiling-based rules.

        Returns None when the output is not a whole multiple of the input.
        """
        ih, iw = inp.shape
        oh, ow = out.shape
        if oh % ih != 0 or ow % iw != 0:
            return None

        br, bc = oh // ih, ow // iw  # number of block rows/cols
        lines = [f"  (Output divided into {br}×{bc} blocks, each {ih}×{iw}:)"]
        for r in range(br):
            for c in range(bc):
                block = out[r * ih:(r + 1) * ih, c * iw:(c + 1) * iw]
                cell_val = int(inp[r, c]) if r < ih and c < iw else "?"
                if np.all(block == 0):
                    desc = "all zeros"
                else:
                    # Show the actual block content so the model can see values
                    desc = _grid_to_str(block)
                    if np.array_equal(block, inp):
                        desc += "  ← this is the complete input grid (note: contains its own zeros too)"
                lines.append(f"  block({r},{c}): input[{r}][{c}]={cell_val} → {desc}")
        return "\n".join(lines)

    def _format_task_prompt(self, task: dict) -> str:
        """Build the initial user prompt describing the ARC task.

        For tasks with large grids (above _LARGE_GRID_CELL_THRESHOLD cells per
        pair), the number of training pairs shown is capped to avoid overflowing
        the model's context window.
        """
        parts = [
            "Here is an ARC-AGI puzzle. Study the training examples and find the transformation rule.\n"
        ]
        train_pairs = task["train"]

        # Decide how many pairs to show based on the largest pair's cell count
        max_cells = max(
            p["input"].size + p["output"].size
            for p in train_pairs
        )
        if max_cells > self._LARGE_GRID_CELL_THRESHOLD:
            train_pairs = train_pairs[: self._LARGE_GRID_MAX_PAIRS]
            parts.append(
                f"(Note: grids are large; showing {len(train_pairs)} of "
                f"{len(task['train'])} training pairs.)\n"
            )

        for i, pair in enumerate(train_pairs):
            inp = pair["input"]
            out = pair["output"]
            parts.append(f"### Training pair {i + 1}")
            parts.append(f"Input  ({inp.shape[0]}×{inp.shape[1]}):\n{_grid_to_str(inp)}")
            parts.append(f"Output ({out.shape[0]}×{out.shape[1]}):\n{_grid_to_str(out)}")
            analysis = self._block_analysis(inp, out)
            if analysis:
                parts.append(analysis)
            parts.append("")

        test_inp = task["test"][0]["input"]
        parts.append(
            f"### Test input ({test_inp.shape[0]}×{test_inp.shape[1]}):\n{_grid_to_str(test_inp)}\n"
        )
        parts.append(
            "Explain your reasoning, then write the `transform` function that produces "
            "the correct output for ANY input following this rule."
        )
        return "\n".join(parts)

    def _format_error_feedback(self, eval_result: dict, attempt: int) -> str:
        """Build the feedback message sent after a failed attempt.

        Includes per-pair correctness, exception messages, full expected/predicted
        grids, and a cell-level diff summary to help the model identify its mistake.
        """
        n_wrong = eval_result["n_total"] - eval_result["n_correct"]
        lines = [
            f"Your function was incorrect on {n_wrong} of {eval_result['n_total']} "
            f"training pairs. Here are the details:\n"
        ]
        for i, pair in enumerate(eval_result["pairs"]):
            if pair["correct"]:
                lines.append(f"Pair {i + 1}: CORRECT ✓")
                continue
            lines.append(f"Pair {i + 1}: WRONG ✗")
            if pair["error"]:
                lines.append(f"  Exception: {pair['error']}")
            else:
                lines.append(
                    f"  Expected  ({pair['expected'].shape[0]}×{pair['expected'].shape[1]}):\n"
                    f"  {_grid_to_str(pair['expected'])}"
                )
                lines.append(
                    f"  Got       ({pair['predicted'].shape[0]}×{pair['predicted'].shape[1]}):\n"
                    f"  {_grid_to_str(pair['predicted'])}"
                )
                lines.append("  Cell differences:\n" + _diff_summary(pair["expected"], pair["predicted"]))
            lines.append("")
        lines.append("Please revise your reasoning and write a corrected `transform` function.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Code extraction & execution
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_to_valid_function(block: str) -> str | None:
        """Trim trailing prose from a code block until it becomes valid Python.

        Some models append grid literals or natural-language explanation after
        the function body.  This method strips lines from the end one-by-one
        until ast.parse succeeds (and a `def` is present), returning the longest
        valid prefix.  Returns None if no valid prefix exists.
        """
        import ast
        lines = block.split("\n")
        for end in range(len(lines), 0, -1):
            candidate = "\n".join(lines[:end]).strip()
            if "def " not in candidate:
                continue
            try:
                ast.parse(candidate)
                return candidate
            except SyntaxError:
                continue
        return None

    def _extract_code(self, text: str) -> str | None:
        """Extract the best Python function from a model response.

        Strategy:
        1. Collect all fenced ```python … ``` blocks (case-insensitive tag).
        2. Iterate in reverse (last block first), looking for one that:
           - contains a `def` statement, and
           - is syntactically valid Python.
           If a block has prose trailing after the function, attempt to trim it.
        3. Fall back to a bare `def …` found anywhere in the text.

        Returning the *last* valid block matches the behaviour of reasoning
        models that draft multiple versions and refine toward the end.
        """
        matches = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if matches:
            import ast
            for block in reversed(matches):
                block = block.strip()
                if "def " not in block:
                    continue
                try:
                    ast.parse(block)
                    return block  # last syntactically-valid block with a def
                except SyntaxError:
                    # Prose may trail the function — try trimming it off
                    trimmed = self._truncate_to_valid_function(block)
                    if trimmed:
                        return trimmed
            return None  # no valid block found among all fenced blocks

        # No fenced blocks at all — try to find a bare function definition.
        # re.DOTALL is intentionally omitted: without it (.*) stops at the
        # first newline, so we only capture the `def` signature line rather
        # than greedily consuming all subsequent prose into the snippet.
        # _truncate_to_valid_function then trims any trailing non-Python lines.
        match = re.search(r"(def \w+\(.*)", text)
        if match:
            candidate = text[match.start():].strip()
            return self._truncate_to_valid_function(candidate)
        return None

    def _execute(self, code: str, input_grid: Grid) -> tuple[Grid | None, str | None]:
        """Execute generated code via arc.sandbox.execute() with this agent's timeout."""
        return sandbox.execute(code, input_grid, self._execution_timeout)

    def _evaluate_test(self, code: str, test_pair: dict) -> bool:
        """Evaluate the best code against the held-out test pair's ground truth.

        This is the honest accuracy metric.  It is called only after the
        self-correction loop has finished — never during it — so the test output
        is never exposed to the model.

        Returns True if the predicted output exactly matches the ground truth,
        False on any mismatch, wrong shape, or runtime error.
        """
        output, error = self._execute(code, test_pair["input"])
        if error or output is None:
            return False
        return grids_equal(output, test_pair["output"])

    def _evaluate_code(self, code: str, task: dict) -> dict:
        """Run the generated code against all training pairs via arc.sandbox."""
        return sandbox.evaluate_code(code, task, self._execution_timeout)

