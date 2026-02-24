"""Single-agent baseline for ARC-AGI.

Flow per task:
  1. Format all training pairs as a prompt
  2. Ask the model to write `def transform(grid: np.ndarray) -> np.ndarray`
  3. Extract and execute the function
  4. If any training pair is wrong, feed pixel-level errors back and ask to fix
  5. Repeat up to max_retries

Backends:
  - "ollama"     : local Ollama server (default, free, runs deepseek-r1:70b)
  - "anthropic"  : Anthropic API (requires credits)
"""
from __future__ import annotations

import re
import textwrap
import traceback
from typing import Any

import numpy as np

from arc.grid import Grid, grids_equal, COLOR_NAMES
from arc.dsl import (
    crop, rotate, flip, translate, scale, tile,
    recolor, mask, overlay, flood_fill,
    find_objects, bounding_box, crop_to_content,
)

# DSL namespace injected into every generated function's exec scope
_DSL_NAMESPACE: dict[str, Any] = {
    "np": np,
    "numpy": np,
    "crop": crop,
    "rotate": rotate,
    "flip": flip,
    "translate": translate,
    "scale": scale,
    "tile": tile,
    "recolor": recolor,
    "mask": mask,
    "overlay": overlay,
    "flood_fill": flood_fill,
    "find_objects": find_objects,
    "bounding_box": bounding_box,
    "crop_to_content": crop_to_content,
}

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

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "deepseek-r1:8b"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks produced by deepseek-r1 and similar models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _grid_to_str(grid: Grid) -> str:
    """Render a grid as a compact Python list literal (for prompts)."""
    rows = []
    for row in grid:
        rows.append("[" + ", ".join(str(int(v)) for v in row) + "]")
    return "[" + ",\n ".join(rows) + "]"


def _diff_summary(expected: Grid, predicted: Grid, max_show: int = 12) -> str:
    """Summarise cell-level differences between two grids."""
    if expected.shape != predicted.shape:
        return f"Shape mismatch: expected {expected.shape}, got {predicted.shape}"
    diffs = []
    it = np.nditer([expected, predicted], flags=["multi_index"])
    while not it.finished:
        r, c = it.multi_index
        ev, pv = int(it[0]), int(it[1])
        if ev != pv:
            diffs.append(
                f"  ({r},{c}) expected {ev} ({COLOR_NAMES[ev]}) got {pv} ({COLOR_NAMES[pv]})"
            )
        it.iternext()
    if not diffs:
        return "  (no differences)"
    extra = f"\n  ... and {len(diffs) - max_show} more" if len(diffs) > max_show else ""
    return "\n".join(diffs[:max_show]) + extra


class SingleAgent:
    def __init__(
        self,
        backend: str = "ollama",
        model: str | None = None,
        max_retries: int = 3,
        timeout: float = 120.0,
        debug: bool = False,
    ):
        """
        Args:
            backend:     "ollama" (local, free) or "anthropic" (API credits required)
            model:       Model name. Defaults to deepseek-r1:70b for ollama,
                         claude-sonnet-4-6 for anthropic.
            max_retries: Number of self-correction attempts after first try.
            timeout:     Seconds to wait for a single model call (ollama only).
            debug:       Print raw model responses for troubleshooting.
        """
        self.backend = backend
        self.max_retries = max_retries
        self.debug = debug

        self._timeout = timeout

        if backend == "ollama":
            self.model = model or DEFAULT_OLLAMA_MODEL
            self.client = None  # native Ollama API used directly
        elif backend == "anthropic":
            import anthropic
            self.model = model or DEFAULT_ANTHROPIC_MODEL
            self.client = anthropic.Anthropic()
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'ollama' or 'anthropic'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, task: dict, task_id: str = "") -> dict:
        """Attempt to solve a task.

        Returns:
            {
                'success': bool,
                'code': str | None,
                'n_attempts': int,
                'log': list[dict],
            }
        """
        log: list[dict] = []
        messages: list[dict] = []

        messages.append({"role": "user", "content": self._format_task_prompt(task)})

        best_code: str | None = None
        best_n_correct: int = -1

        # Temperature schedule: greedy on first attempt, increasingly exploratory on retries
        _temps = [0.0, 0.4, 0.8, 1.0]

        for attempt in range(1, self.max_retries + 2):
            temperature = _temps[min(attempt - 1, len(_temps) - 1)]
            try:
                response_text = self._call_model(messages, temperature=temperature)
            except TimeoutError as e:
                entry = {"attempt": attempt, "error": f"timeout: {e}", "n_correct": 0}
                log.append(entry)
                break
            except Exception as e:
                entry = {"attempt": attempt, "error": f"model_error: {e}", "n_correct": 0}
                log.append(entry)
                break
            messages.append({"role": "assistant", "content": response_text})

            # Strip chain-of-thought thinking tags before extracting code.
            # Fall back to the raw response in case the model put code inside <think>.
            clean_response = _strip_thinking(response_text)
            code = self._extract_code(clean_response) or self._extract_code(response_text)

            if self.debug:
                print(f"\n--- RAW MODEL RESPONSE (attempt {attempt}) ---")
                print(response_text[:3000])
                print(f"--- END (code extracted: {code is not None}) ---\n")

            if code is None:
                entry = {"attempt": attempt, "error": "no_code_block", "n_correct": 0}
                log.append(entry)
                messages.append({
                    "role": "user",
                    "content": "Your response didn't contain a ```python``` code block. Please try again.",
                })
                continue

            eval_result = self._evaluate_code(code, task)
            n_correct = eval_result["n_correct"]
            n_total = eval_result["n_total"]

            entry = {
                "attempt": attempt,
                "code": code,
                "n_correct": n_correct,
                "n_total": n_total,
                "all_correct": eval_result["all_correct"],
            }
            log.append(entry)

            if n_correct > best_n_correct:
                best_n_correct = n_correct
                best_code = code

            if eval_result["all_correct"]:
                return {
                    "success": True,
                    "code": best_code,
                    "n_attempts": attempt,
                    "log": log,
                }

            if attempt <= self.max_retries:
                messages.append({
                    "role": "user",
                    "content": self._format_error_feedback(eval_result, attempt),
                })

        return {
            "success": False,
            "code": best_code,
            "n_attempts": self.max_retries + 1,
            "log": log,
        }

    def predict(self, task: dict) -> Grid | None:
        """Solve and apply the best code to the test input."""
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
        """If output is an exact NxM tiling of input-sized blocks, return a
        pre-computed block analysis string to help the model spot the rule."""
        ih, iw = inp.shape
        oh, ow = out.shape
        if oh % ih != 0 or ow % iw != 0:
            return None
        br, bc = oh // ih, ow // iw  # block grid dimensions
        lines = [f"  (Output divided into {br}×{bc} blocks, each {ih}×{iw}:)"]
        for r in range(br):
            for c in range(bc):
                block = out[r * ih:(r + 1) * ih, c * iw:(c + 1) * iw]
                cell_val = int(inp[r, c]) if r < ih and c < iw else "?"
                if np.all(block == 0):
                    desc = "all zeros"
                else:
                    # Always show the actual block content so the model can see
                    # what values appear (e.g. the full input pattern, not just the cell value)
                    desc = _grid_to_str(block)
                    if np.array_equal(block, inp):
                        desc += "  ← this is the complete input grid (note: contains its own zeros too)"
                lines.append(f"  block({r},{c}): input[{r}][{c}]={cell_val} → {desc}")
        return "\n".join(lines)

    # Max cells (input + output) per pair before we cap the number of pairs shown
    _LARGE_GRID_CELL_THRESHOLD = 300
    _LARGE_GRID_MAX_PAIRS = 2

    def _format_task_prompt(self, task: dict) -> str:
        parts = [
            "Here is an ARC-AGI puzzle. Study the training examples and find the transformation rule.\n"
        ]
        train_pairs = task["train"]
        # For tasks with large grids, cap the number of pairs to avoid context overflow
        max_cells = max(
            inp.size + out.size
            for p in train_pairs
            for inp, out in [(p["input"], p["output"])]
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
        parts.append(f"### Test input ({test_inp.shape[0]}×{test_inp.shape[1]}):\n{_grid_to_str(test_inp)}\n")
        parts.append(
            "Explain your reasoning, then write the `transform` function that produces "
            "the correct output for ANY input following this rule."
        )
        return "\n".join(parts)

    def _format_error_feedback(self, eval_result: dict, attempt: int) -> str:
        lines = [
            f"Your function was incorrect on {eval_result['n_total'] - eval_result['n_correct']} "
            f"of {eval_result['n_total']} training pairs. Here are the details:\n"
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
        """Given a block that may have prose after the function body, trim trailing
        lines until the block is valid Python containing a def.  Returns the
        longest valid prefix, or None if none exists."""
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
        # Collect all fenced code blocks (case-insensitive lang tag).
        # Reasoning models iterate through many drafts; we want the last block
        # that actually contains a function definition and is valid Python.
        matches = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if matches:
            import ast
            for block in reversed(matches):
                block = block.strip()
                if "def " in block:
                    try:
                        ast.parse(block)
                        return block  # last syntactically-valid block with a def
                    except SyntaxError:
                        # Prose may trail the function — try to trim it off
                        trimmed = self._truncate_to_valid_function(block)
                        if trimmed:
                            return trimmed
            # Nothing valid found
            return None
        match = re.search(r"(def \w+\(.*)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _execute(self, code: str, input_grid: Grid) -> tuple[Grid | None, str | None]:
        import io
        import contextlib

        if "input(" in code or "sys.stdin" in code:
            return None, "Code uses input()/stdin — not allowed; grid is passed as argument."

        namespace = dict(_DSL_NAMESPACE)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, namespace)  # noqa: S102
        except Exception as e:
            return None, f"Compile error: {type(e).__name__}: {e}"

        transform_fn = namespace.get("transform")
        if transform_fn is None:
            # Accept any user-defined callable — models sometimes use a different name
            user_fns = [
                v for k, v in namespace.items()
                if callable(v) and k not in _DSL_NAMESPACE and not k.startswith("_")
                and k != "__builtins__"
            ]
            if user_fns:
                transform_fn = user_fns[-1]
            else:
                return None, "No `transform` function found in generated code."

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = transform_fn(input_grid.copy())
            if not isinstance(result, np.ndarray):
                result = np.array(result, dtype=np.int32)
            return result.astype(np.int32), None
        except Exception as e:
            return None, f"Runtime error: {type(e).__name__}: {e}\n{traceback.format_exc()}"

    def _evaluate_code(self, code: str, task: dict) -> dict:
        pairs = []
        for pair in task["train"]:
            output, error = self._execute(code, pair["input"])
            if error:
                pairs.append({
                    "correct": False,
                    "predicted": None,
                    "expected": pair["output"],
                    "error": error,
                })
            else:
                correct = grids_equal(output, pair["output"])
                pairs.append({
                    "correct": correct,
                    "predicted": output,
                    "expected": pair["output"],
                    "error": None,
                })

        n_correct = sum(p["correct"] for p in pairs)
        return {
            "pairs": pairs,
            "n_correct": n_correct,
            "n_total": len(pairs),
            "all_correct": n_correct == len(pairs),
        }

    # ------------------------------------------------------------------
    # Model API (backend-agnostic)
    # ------------------------------------------------------------------

    def _call_model(self, messages: list[dict], temperature: float = 0.0) -> str:
        if self.backend == "anthropic":
            return self._call_anthropic(messages)
        return self._call_ollama(messages, temperature=temperature)

    def _call_anthropic(self, messages: list[dict]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=_SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text

    def _call_ollama(self, messages: list[dict], temperature: float = 0.0) -> str:
        import json
        import socket
        import time
        import urllib.error
        import urllib.request

        full_messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + messages
        payload = json.dumps({
            "model": self.model,
            "messages": full_messages,
            "stream": True,  # stream so we can exit early & recover partial results
            "options": {
                "temperature": temperature,
                "num_predict": 32000,  # budget-forces model to conclude; at ~40 tok/s ≈ 800s
            },
        }).encode()

        req = urllib.request.Request(
            OLLAMA_CHAT_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        thinking_parts: list[str] = []
        content_parts: list[str] = []
        # Per-read socket timeout: shorter than overall deadline so we can check
        # wall-clock time between chunks.
        _READ_TIMEOUT = min(self._timeout, 60)
        deadline = time.monotonic() + self._timeout

        try:
            with urllib.request.urlopen(req, timeout=_READ_TIMEOUT) as resp:
                for raw_line in resp:
                    try:
                        chunk = json.loads(raw_line.strip())
                    except (json.JSONDecodeError, ValueError):
                        continue

                    msg = chunk.get("message", {})
                    thinking_parts.append(msg.get("thinking") or "")
                    content_parts.append(msg.get("content") or "")

                    if chunk.get("done"):
                        break

                    # Early exit: complete code block found in content
                    content_so_far = "".join(content_parts)
                    if (
                        content_so_far.count("```") >= 2
                        and "def " in content_so_far
                        and re.search(r"```(?:python)?\s.*?```", content_so_far, re.DOTALL | re.IGNORECASE)
                    ):
                        break

                    # Deadline check (wall clock)
                    if time.monotonic() > deadline:
                        break

        except socket.timeout as e:
            # Per-read socket timeout — treat as overall timeout only if we got nothing
            if not thinking_parts and not content_parts:
                raise TimeoutError(f"Ollama call timed out after {self._timeout}s") from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, socket.timeout):
                if not thinking_parts and not content_parts:
                    raise TimeoutError(f"Ollama call timed out after {self._timeout}s") from e
            else:
                raise

        thinking = "".join(thinking_parts)
        content = "".join(content_parts)

        if self.debug:
            print(f"[debug] content length={len(content)}  thinking length={len(thinking)}")

        # Reassemble so _strip_thinking can handle it uniformly
        if thinking and content:
            return f"<think>{thinking}</think>\n{content}"
        if thinking:
            return f"<think>{thinking}</think>"
        return content
