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
    You are an expert at solving ARC-AGI (Abstraction and Reasoning Corpus) puzzles.

    Each puzzle shows several input→output grid pairs that share one hidden transformation
    rule. Your job: discover the rule and write a Python function that implements it exactly.

    Grids are 2-D numpy arrays of integers 0-9 representing colours:
      0=black  1=blue  2=red   3=green  4=yellow
      5=grey   6=magenta  7=orange  8=azure  9=maroon

    DSL helpers available inside your function (already imported, no extra imports needed):
      crop(grid, r1, c1, r2, c2)        flip(grid, axis)          rotate(grid, n)
      translate(grid, dr, dc, fill=0)   scale(grid, factor)       tile(grid, n_rows, n_cols)
      recolor(grid, from_color, to_color)
      flood_fill(grid, row, col, new_color)
      find_objects(grid, background=None) → list of {color, pixels, bbox, subgrid}
      bounding_box(grid, color=None)     crop_to_content(grid)
      mask(grid, mask_grid, fill=0)      overlay(base, top, transparent=0)
      np  (numpy is available as np)

    REQUIRED RESPONSE FORMAT — follow these steps in order:

    STEP 1 — OBSERVE: For each training pair, describe in one sentence what changed
    (e.g. "the 3×3 input is tiled 3 times horizontally and vertically to make a 9×9 output").

    STEP 2 — RULE: State the single generalised rule in plain English.

    STEP 3 — VERIFY: Mentally check the rule holds for every training pair.

    STEP 4 — CODE: Write the function in a ```python ... ``` block.

    Coding rules:
    - Signature must be exactly:  def transform(grid: np.ndarray) -> np.ndarray
    - Return a NEW numpy int32 array — never modify the input in place.
    - No imports, no print statements, no top-level code outside the function.
""").strip()

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "deepseek-r1:70b"
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

    def _format_task_prompt(self, task: dict) -> str:
        parts = [
            "Here is an ARC-AGI puzzle. Study the training examples and find the transformation rule.\n"
        ]
        for i, pair in enumerate(task["train"]):
            inp = pair["input"]
            out = pair["output"]
            parts.append(f"### Training pair {i + 1}")
            parts.append(f"Input  ({inp.shape[0]}×{inp.shape[1]}):\n{_grid_to_str(inp)}")
            parts.append(f"Output ({out.shape[0]}×{out.shape[1]}):\n{_grid_to_str(out)}\n")

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

    def _extract_code(self, text: str) -> str | None:
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"(def transform\(.*)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _execute(self, code: str, input_grid: Grid) -> tuple[Grid | None, str | None]:
        import io
        import contextlib

        namespace = dict(_DSL_NAMESPACE)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, namespace)  # noqa: S102
        except Exception as e:
            return None, f"Compile error: {type(e).__name__}: {e}"

        transform_fn = namespace.get("transform")
        if transform_fn is None:
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
        import urllib.request

        full_messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + messages
        payload = json.dumps({
            "model": self.model,
            "messages": full_messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 16000},
        }).encode()

        import socket
        import urllib.error

        req = urllib.request.Request(
            OLLAMA_CHAT_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                result = json.loads(resp.read())
        except socket.timeout as e:
            raise TimeoutError(f"Ollama call timed out after {self._timeout}s") from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, socket.timeout):
                raise TimeoutError(f"Ollama call timed out after {self._timeout}s") from e
            raise

        msg = result.get("message", {})
        content = msg.get("content", "")
        thinking = msg.get("thinking", "")

        if self.debug:
            print(f"[debug] content length={len(content)}  thinking length={len(thinking)}")

        # Reassemble so _strip_thinking can handle it uniformly
        if thinking and content:
            return f"<think>{thinking}</think>\n{content}"
        if thinking:
            return f"<think>{thinking}</think>"
        return content
