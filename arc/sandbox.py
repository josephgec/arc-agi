"""Sandboxed execution and evaluation of LLM-generated transform functions.

This module is the single source of truth for running untrusted code safely.
It is designed to be backend-agnostic so any future agent (Analyst, Coder,
Critic, …) can share the same hardened execution layer without duplicating
timeout or namespace logic.

Public API
----------
execute(code, input_grid, timeout) -> (Grid | None, str | None)
    Run a transform function in a child process with a hard wall-clock limit.

evaluate_code(code, task, timeout) -> dict
    Apply execute() to every training pair and return correctness statistics.

Constants
---------
EXECUTION_TIMEOUT   Hard wall-clock limit in seconds (default 10 s).
DSL_NAMESPACE       The exec namespace seeded with DSL helpers and numpy.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np

from arc.grid import Grid, grids_equal
from arc.dsl import (
    crop, rotate, flip, translate, scale, tile,
    recolor, mask, overlay, flood_fill,
    find_objects, bounding_box, crop_to_content,
)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Hard wall-clock limit for executing generated code.  Any future agent that
# calls execute() inherits this default and can pass a tighter value if needed.
EXECUTION_TIMEOUT: float = 10.0

# DSL functions and numpy injected into every generated function's exec scope.
# Defined at module level so _subprocess_worker can access it after spawn.
DSL_NAMESPACE: dict[str, Any] = {
    "np":              np,
    "numpy":           np,
    "crop":            crop,
    "rotate":          rotate,
    "flip":            flip,
    "translate":       translate,
    "scale":           scale,
    "tile":            tile,
    "recolor":         recolor,
    "mask":            mask,
    "overlay":         overlay,
    "flood_fill":      flood_fill,
    "find_objects":    find_objects,
    "bounding_box":    bounding_box,
    "crop_to_content": crop_to_content,
}


# ---------------------------------------------------------------------------
# Subprocess worker  (module-level — required for pickle / spawn start method)
# ---------------------------------------------------------------------------

def _subprocess_worker(code: str, grid_list: list, out_queue: mp.Queue) -> None:
    """Execute generated code inside a child process and put the result in out_queue.

    Must be a module-level function so it is picklable when multiprocessing
    uses the 'spawn' start method (default on macOS / Windows).

    Puts exactly one item into out_queue:
        ("ok",    result_list)  — success; result_list is grid.tolist()
        ("error", message: str) — any failure (compile, runtime, missing fn)
    """
    import contextlib
    import io
    import traceback

    import numpy as np  # re-import required in spawned process

    namespace = dict(DSL_NAMESPACE)

    # --- Compile / exec phase ---
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, namespace)  # noqa: S102
    except Exception as e:
        out_queue.put(("error", f"Compile error: {type(e).__name__}: {e}"))
        return

    # --- Resolve the transform function ---
    # Prefer "transform" by name; fall back to the last user-defined callable.
    transform_fn = namespace.get("transform")
    if transform_fn is None:
        user_fns = [
            v for k, v in namespace.items()
            if callable(v)
            and k not in DSL_NAMESPACE
            and not k.startswith("_")
            and k != "__builtins__"
        ]
        if user_fns:
            transform_fn = user_fns[-1]
        else:
            out_queue.put(("error", "No `transform` function found in generated code."))
            return

    # --- Run phase ---
    input_grid = np.array(grid_list, dtype=np.int32)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = transform_fn(input_grid.copy())
        if not isinstance(result, np.ndarray):
            result = np.array(result, dtype=np.int32)
        out_queue.put(("ok", result.astype(np.int32).tolist()))
    except Exception as e:
        out_queue.put(("error", f"Runtime error: {type(e).__name__}: {e}\n{traceback.format_exc()}"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute(
    code: str,
    input_grid: Grid,
    timeout: float = EXECUTION_TIMEOUT,
) -> tuple[Grid | None, str | None]:
    """Execute generated code in a child process with a hard wall-clock timeout.

    Spawns a multiprocessing.Process so that infinite loops, OOM errors, or
    any other runaway computation cannot freeze the caller.  The child is
    forcibly killed if it has not finished within `timeout` seconds.

    The code runs inside an isolated namespace (DSL_NAMESPACE) with stdout
    suppressed.  Code referencing input()/sys.stdin is rejected before the
    subprocess is spawned.

    Args:
        code:       Python source code containing a transform function.
        input_grid: The grid to pass to the function.
        timeout:    Wall-clock seconds before the child is killed.

    Returns:
        (result_grid, None)  on success.
        (None, error_msg)    on any failure (compile, runtime, timeout, stdin).
    """
    if "input(" in code or "sys.stdin" in code:
        return None, "Code uses input()/stdin — not allowed; grid is passed as argument."

    out_queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_subprocess_worker,
        args=(code, input_grid.tolist(), out_queue),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return None, (
            f"Execution timed out after {timeout}s "
            "(infinite loop or excessive computation)"
        )

    if out_queue.empty():
        return None, f"Execution process exited unexpectedly (exit code {proc.exitcode})"

    status, value = out_queue.get()
    if status == "ok":
        return np.array(value, dtype=np.int32), None
    return None, value


def evaluate_code(
    code: str,
    task: dict,
    timeout: float = EXECUTION_TIMEOUT,
) -> dict:
    """Run generated code against every training pair and collect results.

    Args:
        code:    Python source containing a transform function.
        task:    ARC task dict with a "train" list of {input, output} pairs.
        timeout: Per-execution wall-clock limit passed through to execute().

    Returns:
        {
            "pairs":       list[dict]  — per-pair results (correct, predicted,
                                         expected, error)
            "n_correct":   int         — number of pairs that passed
            "n_total":     int         — total training pairs evaluated
            "all_correct": bool        — True iff n_correct == n_total
        }
    """
    pairs = []
    for pair in task["train"]:
        output, error = execute(code, pair["input"], timeout)
        if error:
            pairs.append({
                "correct":   False,
                "predicted": None,
                "expected":  pair["output"],
                "error":     error,
            })
        else:
            correct = grids_equal(output, pair["output"])
            pairs.append({
                "correct":   correct,
                "predicted": output,
                "expected":  pair["output"],
                "error":     None,
            })

    n_correct = sum(p["correct"] for p in pairs)
    return {
        "pairs":       pairs,
        "n_correct":   n_correct,
        "n_total":     len(pairs),
        "all_correct": n_correct == len(pairs),
    }
