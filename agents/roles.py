"""Specialized agent roles for the ARC-AGI multi-agent framework.

Three cognitive roles collaborate to solve ARC puzzles:

  Hypothesizer — observes grid transformations and proposes competing
                 natural-language algorithms without writing any code.

  Coder        — translates one hypothesis into an executable Python DSL
                 function using only the provided primitives.

  Critic       — diagnoses failures and routes feedback to either the
                 Hypothesizer (flawed rule) or the Coder (implementation bug).
"""
from __future__ import annotations

import re

from agents.llm_client import LLMClient


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# DSL reference card given to the Coder
# ---------------------------------------------------------------------------

DSL_DOCS = """\
Available DSL primitives (all pre-imported — do NOT add any import statements):

Geometric transforms:
  crop(grid, r1, c1, r2, c2)          → sub-grid rows [r1:r2], cols [c1:c2] (exclusive end)
  rotate(grid, n=1)                    → rotate 90° counter-clockwise n times (1=90°, 2=180°, 3=270°)
  flip(grid, axis=0)                   → flip along axis: 0=vertical (up/down), 1=horizontal (left/right)
  translate(grid, dr, dc, fill=0)      → shift by (dr rows, dc cols); positive=down/right; no wrap
  scale(grid, factor)                  → enlarge: each cell → factor×factor block
  tile(grid, n_rows, n_cols)           → repeat grid n_rows times vertically, n_cols times horizontally

Color operations:
  recolor(grid, from_color, to_color)  → replace every from_color cell with to_color
  mask(grid, mask_grid, fill=0)        → zero out cells where mask_grid == 0
  overlay(base, top, transparent=0)    → place top onto base, skipping transparent cells

Flood fill:
  flood_fill(grid, row, col, new_color) → 4-connected flood-fill from (row, col)

Object detection:
  find_objects(grid, background=None)  → list of dicts with keys:
      'color'   int          — the object's color
      'pixels'  list[(r,c)]  — all cell coordinates
      'bbox'    (r_min, c_min, r_max, c_max) — inclusive bounding box
      'subgrid' Grid         — minimal bounding-box crop (background=0)
  bounding_box(grid, color=None)       → (r_min, c_min, r_max, c_max) inclusive;
                                         color=None → all non-background cells
  crop_to_content(grid)                → crop to bounding box of non-background cells

Numpy:
  np — numpy is available as `np` (e.g. np.zeros, np.where, np.unique)

Colors: 0=black 1=blue 2=red 3=green 4=yellow 5=grey 6=fuschia 7=orange 8=azure 9=maroon

Rules:
  - Function MUST be named transform(grid) and return a numpy int32 array
  - No imports, no print, no input
  - Compose DSL primitives; avoid raw numpy indexing when a primitive exists\
"""


# ---------------------------------------------------------------------------
# Hypothesizer
# ---------------------------------------------------------------------------

class Hypothesizer:
    """Proposes competing natural-language transformation algorithms.

    Studies the training grid pairs and generates exactly 3 distinct hypotheses
    describing the transformation in plain English.  Does not write code.
    """

    _SYSTEM_PROMPT = """\
You are an abstract logic puzzle solver specializing in ARC (Abstraction and Reasoning Corpus).

Your task: study the input/output grid pairs and propose exactly 3 DISTINCT, COMPETING \
step-by-step natural language algorithms that could explain the transformation.

Rules:
- DO NOT write any code.
- Number each hypothesis: 1., 2., 3.
- Begin each hypothesis with a one-sentence summary on a line by itself.
- Use bullet points for each step.
- Be precise about colors (0=black, 1=blue, 2=red, 3=green, 4=yellow, \
5=grey, 6=fuschia, 7=orange, 8=azure, 9=maroon), object positions, and geometric movements.
- Each hypothesis must be a complete, standalone description.
- Think broadly: consider rotation, reflection, recoloring, pattern extension, \
object movement, tiling, scaling, and conditional rules.
- OUTPUT SHAPE (REQUIRED in every hypothesis): State whether the output grid is the \
SAME size as the input, or a DIFFERENT size. If different, give the exact formula \
(e.g., "output is input_height × 2 rows by input_width × 2 cols", "output is 1×1", \
"output height = number of distinct colors in input").

Output ONLY the 3 numbered hypotheses, nothing else.\
"""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def generate(self, task_description: str, feedback: str | None = None) -> str:
        """Return the raw model response containing 3 numbered hypotheses.

        Args:
            task_description: Formatted grid pairs with block analysis.
            feedback:         Optional Critic feedback asking for new hypotheses.
        """
        if feedback:
            content = (
                f"{task_description}\n\n"
                f"--- Critic feedback from previous attempt ---\n{feedback}\n"
                f"---\nPropose 3 NEW hypotheses that address the above feedback."
            )
        else:
            content = task_description
        messages = [{"role": "user", "content": content}]
        return self._client.generate(self._SYSTEM_PROMPT, messages)


# ---------------------------------------------------------------------------
# Coder
# ---------------------------------------------------------------------------

class Coder:
    """Translates a single hypothesis into executable Python DSL code.

    Takes one natural-language hypothesis from the Hypothesizer and produces
    a syntactically valid ``transform(grid)`` function using only the DSL
    primitives documented in DSL_DOCS.
    """

    _SYSTEM_PROMPT_TEMPLATE = """\
You are a Python code generator for ARC-AGI puzzles.

You will be given a natural-language transformation rule.
Implement it immediately as a Python function. Start your response with ```python.
Do NOT think out loud. Do NOT explain. Do NOT write prose. Output ONLY the code block.

{dsl_docs}

OUTPUT SHAPE — determine this FIRST before writing any code:
  Same size as input  → use np.copy(grid) as your base, or use shape-preserving \
primitives (recolor, mask, overlay, flood_fill).
  Larger than input   → allocate output FIRST: `out = np.zeros((new_h, new_w), dtype=np.int32)` \
then fill it. Or use scale(grid, factor) or tile(grid, n_rows, n_cols) which handle sizing.
  Smaller than input  → use crop(grid, r1, c1, r2, c2) or crop_to_content(grid).

MULTI-STEP COMPOSITION — use named intermediate variables:
  GOOD — each step is readable and debuggable:
    step1  = flip(grid, 0)
    step2  = rotate(step1, 1)
    result = recolor(step2, 1, 2)
    return result

  BAD — deeply nested calls hide shape bugs:
    return recolor(rotate(flip(grid, 0), 1), 1, 2)

CRITICAL RULE: NEVER mutate NumPy grids sequentially when swapping colors or \
applying multiple rules (e.g., `grid[grid==1]=2` then `grid[grid==2]=1`). This \
creates overlapping overwrite bugs. ALWAYS use `np.copy()`, `np.where()`, or \
`np.select()` to apply transformations simultaneously.

  BAD — sequential mutation destroys earlier writes:
    grid[grid == 1] = 2
    grid[grid == 2] = 1  # BUG: also overwrites the 1→2 cells just set above

  GOOD — freeze source values with a copy first:
    new_grid = np.copy(grid)
    new_grid[grid == 1] = 2
    new_grid[grid == 2] = 1

  GOOD — build result in a single vectorised step:
    new_grid = np.select([grid == 1, grid == 2], [2, 1], default=grid)

Output ONLY the ```python code block. No explanation. No commentary. No prose.\
"""

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self._system_prompt = self._SYSTEM_PROMPT_TEMPLATE.format(dsl_docs=DSL_DOCS)

    def generate(
        self,
        hypothesis: str,
        feedback: str | None = None,
        training_context: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Return the raw model response containing a ```python code block.

        Args:
            hypothesis:       Natural-language transformation algorithm to implement.
                              Any <think>…</think> blocks are stripped before the
                              hypothesis is sent so reasoning-model chain-of-thought
                              from the Hypothesizer never reaches the Coder.
            feedback:         Optional Critic feedback from a previous failed attempt.
            training_context: Optional formatted training pairs shown after the
                              hypothesis so the Coder can verify its implementation.
            temperature:      Sampling temperature (0 = greedy; raise for retries).
        """
        # Strip <think>…</think> blocks that a reasoning Hypothesizer may have
        # emitted.  qwen2.5-coder doesn't need the chain-of-thought — it only
        # needs the clean, structured algorithm description.
        hypothesis = _strip_thinking(hypothesis)

        base = f"Hypothesis to implement:\n\n{hypothesis}"
        if training_context:
            base = f"{base}\n\n{training_context}"
        if feedback:
            content = (
                f"{base}\n\n"
                f"--- Critic feedback from previous attempt ---\n{feedback}\n"
                f"---\nFix the implementation based on the feedback above."
            )
        else:
            content = base
        messages = [{"role": "user", "content": content}]
        return self._client.generate(self._system_prompt, messages, temperature=temperature)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

ROUTE_HYPOTHESIZER = "hypothesizer"
ROUTE_CODER        = "coder"


class Critic:
    """Diagnoses failures and routes feedback to the Hypothesizer or Coder.

    Given a failed attempt, determines whether:
      - The logical hypothesis was wrong  → route to Hypothesizer
      - The implementation had a bug      → route to Coder

    ``analyze()`` always returns a dict with keys:
      'route':    'hypothesizer' | 'coder'
      'feedback': actionable feedback string for the next agent
    """

    _SYSTEM_PROMPT = """\
You are a debugging expert for an ARC puzzle-solving system.

You will be given:
1. The natural-language hypothesis that described the transformation
2. The Python code that attempted to implement it
3. The execution result (error message or test failures)
4. A pixel-level diff between predicted and expected outputs

Your task: determine the ROOT CAUSE of the failure.

Option A — FLAWED HYPOTHESIS: The logical rule itself is wrong, incomplete, or misses an edge case.
  → The Hypothesizer must write a better rule.

Option B — CODING ERROR: The hypothesis is sound but was implemented incorrectly.
  → The Coder must fix the implementation bug.

Common coding errors to check:
- SEQUENTIAL OVERWRITE BUG: CRITICAL RULE: NEVER mutate NumPy grids sequentially \
when swapping colors or applying multiple rules (e.g., `grid[grid==1]=2` then \
`grid[grid==2]=1`). This creates overlapping overwrite bugs. ALWAYS use `np.copy()`, \
`np.where()`, or `np.select()` to apply transformations simultaneously.
- SHAPE ERROR: If expected.shape != predicted.shape in the diff, the code computed \
the wrong output dimensions. Check: (a) did the hypothesis say the output is a \
different size? (b) did the code allocate `np.zeros((new_h, new_w), ...)` with \
the correct formula, or did it incorrectly use `np.zeros_like(grid)`? When routing \
to CODER for a shape error, state the exact expected shape formula from the hypothesis.
- Off-by-one errors in bounding box, crop, or loop indices.
- Wrong color constant (0=black,1=blue,2=red,3=green,4=yellow,5=grey,6=fuschia,7=orange,8=azure,9=maroon).
- Nested DSL calls where an intermediate result has the wrong shape fed into the next call.

Analyze the failure, then end your response with EXACTLY one of these two lines:
ROUTE: HYPOTHESIZER
ROUTE: CODER

Immediately after the route line, write specific, actionable feedback for the next agent.\
"""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def analyze(
        self,
        hypothesis: str,
        code: str,
        error_info: str,
        diff_info: str,
    ) -> dict:
        """Diagnose a failure and return routing + feedback.

        Returns:
            {'route': 'hypothesizer'|'coder', 'feedback': str}
        """
        content = (
            f"### Hypothesis\n{hypothesis}\n\n"
            f"### Code\n```python\n{code}\n```\n\n"
            f"### Execution Result\n{error_info}\n\n"
            f"### Pixel Diff\n{diff_info}"
        )
        messages = [{"role": "user", "content": content}]
        response = self._client.generate(self._SYSTEM_PROMPT, messages)
        return self._parse_response(response)

    def _parse_response(self, response: str) -> dict:
        """Extract route and feedback from the Critic's raw response.

        Strips reasoning tokens first so that <think> blocks cannot confuse
        the route detector and are never leaked into the feedback passed to
        the next agent.

        Feedback priority:
          1. Text *after* the ROUTE line (the model's explicit guidance).
          2. Text *before* the ROUTE line (the model's analysis) — used when
             the model puts the ROUTE line last with nothing following it.
          3. The cleaned full response as a last resort.
        """
        clean = _strip_thinking(response)
        route = ROUTE_CODER   # safe default

        lines         = clean.splitlines()
        pre_route     : list[str] = []
        post_route    : list[str] = []
        route_found   = False

        for line in lines:
            if route_found:
                post_route.append(line)
            elif line.strip().upper() == "ROUTE: HYPOTHESIZER":
                route       = ROUTE_HYPOTHESIZER
                route_found = True
            elif line.strip().upper() == "ROUTE: CODER":
                route       = ROUTE_CODER
                route_found = True
            else:
                pre_route.append(line)

        after  = "\n".join(post_route).strip()
        before = "\n".join(pre_route).strip()
        feedback = after or before or clean
        return {"route": route, "feedback": feedback}
