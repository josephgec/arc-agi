"""Test-time compute scaling via ensembling and majority voting.

The Compositional Gap: LLMs frequently hallucinate rules that pass the
training pairs but fail on the hidden test input.  This module mitigates
that by independently deriving multiple candidate programs and submitting
only the output that a majority of them agree on.

The core insight: if k distinct code paths — produced by independent
Orchestrator runs — all produce the same pixel-perfect output grid for the
test input, it is statistically near-certain to be the generalised rule.

Flow
----
1. Run the Orchestrator repeatedly, collecting programs that achieve 100%
   on all training pairs.  Stop when target_candidates unique programs are
   pooled or max_runs is exhausted.
2. Execute every pooled program on the test input.
3. Group identical output grids (pixel-perfect equality).
4. Submit the output grid with the most supporting programs (votes).
   Ties are broken in favour of the first group formed.
"""
from __future__ import annotations

import numpy as np

from arc import sandbox
from arc.grid import Grid, grids_equal
from agents.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Voting primitives
# ---------------------------------------------------------------------------

def _grids_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True when two grids have identical shape and values."""
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _majority_vote(grids: list[np.ndarray]) -> np.ndarray | None:
    """Return the grid that appears most often (pixel-perfect equality).

    Ties are broken in favour of the first group to reach the maximum count.
    Returns None when the list is empty.
    """
    if not grids:
        return None

    representatives: list[np.ndarray] = []
    counts:          list[int]         = []

    for g in grids:
        for i, rep in enumerate(representatives):
            if _grids_equal(g, rep):
                counts[i] += 1
                break
        else:
            representatives.append(g)
            counts.append(1)

    return representatives[counts.index(max(counts))]


def _vote_summary(grids: list[np.ndarray]) -> list[dict]:
    """Group grids by identity and return groups sorted by count descending.

    Each group dict contains:
        'output':            np.ndarray — the representative grid
        'count':             int        — number of programs producing it
        'candidate_indices': list[int]  — indices into the original grids list
    """
    if not grids:
        return []

    representatives:   list[np.ndarray] = []
    indices_per_group: list[list[int]]  = []

    for i, g in enumerate(grids):
        for j, rep in enumerate(representatives):
            if _grids_equal(g, rep):
                indices_per_group[j].append(i)
                break
        else:
            representatives.append(g)
            indices_per_group.append([i])

    groups = [
        {
            "output":            rep,
            "count":             len(idx),
            "candidate_indices": idx,
        }
        for rep, idx in zip(representatives, indices_per_group)
    ]
    return sorted(groups, key=lambda g: -g["count"])


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class Ensemble:
    """Collect candidate programs and select the answer by majority voting.

    Runs the Orchestrator up to max_runs times.  After each run, newly
    discovered candidate programs (those achieving 100% on training pairs)
    are added to the pool, de-duplicated by code text.  Collection stops
    once the pool reaches target_candidates programs or max_runs is exhausted.

    All pooled programs are then executed on the test input.  The output
    grid that the most programs agree on is submitted as the prediction.

    Args:
        backend:           'ollama' or 'anthropic'.
        model:             Model name override.
        timeout:           Seconds per LLM call (Ollama only).
        debug:             Print progress to stdout.
        target_candidates: Stop collecting when this many unique programs are
                           pooled.  Recommended: 3–5.
        max_runs:          Hard cap on Orchestrator calls.
        n_hypotheses:      Hypotheses requested per Orchestrator run.
        max_retries:       Coder retries per hypothesis per Orchestrator run.
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
        target_candidates:        int        = 3,
        max_runs:                 int        = 5,
        n_hypotheses:             int        = 3,
        max_retries:              int        = 2,
    ) -> None:
        self._orchestrator    = Orchestrator(
            backend=backend,
            model=model,
            hypothesizer_model=hypothesizer_model,
            coder_model=coder_model,
            critic_model=critic_model,
            hypothesizer_temperature=hypothesizer_temperature,
            coder_temperature=coder_temperature,
            critic_temperature=critic_temperature,
            hypothesizer_max_tokens=hypothesizer_max_tokens,
            coder_max_tokens=coder_max_tokens,
            critic_max_tokens=critic_max_tokens,
            timeout=timeout,
            debug=debug,
            n_hypotheses=n_hypotheses,
            max_retries=max_retries,
        )
        self.target_candidates        = target_candidates
        self.max_runs                 = max_runs
        self.debug                    = debug
        self.backend                  = self._orchestrator.backend
        self.hypothesizer_model       = self._orchestrator.hypothesizer_model
        self.coder_model              = self._orchestrator.coder_model
        self.critic_model             = self._orchestrator.critic_model
        self.hypothesizer_temperature = self._orchestrator.hypothesizer_temperature
        self.coder_temperature        = self._orchestrator.coder_temperature
        self.critic_temperature       = self._orchestrator.critic_temperature
        self.hypothesizer_max_tokens  = self._orchestrator.hypothesizer_max_tokens
        self.coder_max_tokens         = self._orchestrator.coder_max_tokens
        self.critic_max_tokens        = self._orchestrator.critic_max_tokens
        self.model                    = self._orchestrator.model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, task: dict) -> dict:
        """Run the ensemble loop and return a result dict.

        Returns:
            {
                'success':      bool
                'prediction':   np.ndarray | None  — majority-voted output grid
                'test_correct': bool | None         — None when test output absent
                'candidates':   list[dict]          — unique programs (100% training)
                'vote_summary': list[dict]          — output groups by vote count
                'n_runs':       int                 — Orchestrator calls made
            }
        """
        seen_codes: set[str]   = set()
        candidates: list[dict] = []
        n_runs:     int        = 0

        test_pair             = (task.get("test") or [{}])[0]
        has_test_ground_truth = "output" in test_pair

        # ---- Collection loop ----------------------------------------
        for _ in range(self.max_runs):
            n_runs += 1
            if self.debug:
                print(
                    f"[ensemble] run {n_runs}/{self.max_runs} "
                    f"(pool: {len(candidates)}/{self.target_candidates})"
                )

            result = self._orchestrator.solve(task)

            for cand in result.get("candidates", []):
                code = cand["code"].strip()
                if code not in seen_codes:
                    seen_codes.add(code)
                    candidates.append(cand)

            if len(candidates) >= self.target_candidates:
                break

        if not candidates:
            return {
                "success":      False,
                "prediction":   None,
                "test_correct": None,
                "candidates":   [],
                "vote_summary": [],
                "n_runs":       n_runs,
            }

        # ---- Execute all candidates on the test input ---------------
        test_input = test_pair["input"]
        outputs: list[np.ndarray | None] = []

        for cand in candidates:
            out, _ = sandbox.execute(cand["code"], test_input)
            outputs.append(out)

        valid_outputs = [o for o in outputs if o is not None]
        prediction    = _majority_vote(valid_outputs)
        vote_summary  = _vote_summary(valid_outputs)

        if self.debug:
            top = vote_summary[0]["count"] if vote_summary else 0
            print(
                f"[ensemble] {len(valid_outputs)} valid outputs, "
                f"{len(vote_summary)} distinct group(s), "
                f"top vote: {top}/{len(valid_outputs)}"
            )

        test_correct: bool | None = None
        if has_test_ground_truth and prediction is not None:
            test_correct = grids_equal(prediction, test_pair["output"])

        return {
            "success":      prediction is not None,
            "prediction":   prediction,
            "test_correct": test_correct,
            "candidates":   candidates,
            "vote_summary": vote_summary,
            "n_runs":       n_runs,
        }

    def predict(self, task: dict) -> Grid | None:
        """Solve a task and return the majority-voted output grid, or None."""
        return self.solve(task)["prediction"]
