"""State-machine orchestrator for ARC-AGI multi-agent puzzle solving.

The Orchestrator owns task state and drives the conversational loop across
three specialized agents, advancing through explicit states:

    HYPOTHESIZING  —  Hypothesizer generates N competing natural-language rules
    CODING         —  Coder translates one rule into a Python DSL function
    EVALUATING     —  Sandbox executes the code against all training pairs
    CRITICIZING    —  Critic diagnoses failure and assigns blame
    CANDIDATE      —  A program that passes all training pairs is saved

For each hypothesis the Coder receives up to (max_retries + 1) attempts.
A Critic verdict of "hypothesizer" abandons the current hypothesis immediately;
a verdict of "coder" retries with targeted feedback up to max_retries times.
"""
from __future__ import annotations

from arc import sandbox
from arc.grid import Grid, grids_equal
from agents.llm_client import LLMClient
from agents.roles import Hypothesizer, Coder, Critic, ROUTE_HYPOTHESIZER
from agents.multi_agent import (
    _extract_code,
    _format_diff,
    _format_error_info,
    _format_task_description,
    _parse_hypotheses,
    _strip_thinking,
)


# Log state-name constants (appear as the "state" key in every log entry).
STATE_HYPOTHESIZING = "HYPOTHESIZING"
STATE_CODING        = "CODING"
STATE_EVALUATING    = "EVALUATING"
STATE_CRITICIZING   = "CRITICIZING"
STATE_CANDIDATE     = "CANDIDATE"


class Orchestrator:
    """State-machine orchestrator for multi-agent ARC puzzle solving.

    Outer loop: iterate over each hypothesis produced by the Hypothesizer.
    Inner loop: give the Coder up to (max_retries + 1) attempts per hypothesis.

    Args:
        backend:      'ollama' or 'anthropic'.
        model:        Model name override.
        timeout:      Seconds per LLM call (Ollama only).
        debug:        Print state transitions to stdout.
        n_hypotheses: Number of hypotheses to request from the Hypothesizer.
        max_retries:  Maximum additional Coder attempts per hypothesis after
                      the first (total Coder calls per hypothesis = max_retries + 1).
    """

    def __init__(
        self,
        backend:      str        = "ollama",
        model:        str | None = None,
        timeout:      float      = 120.0,
        debug:        bool       = False,
        n_hypotheses: int        = 3,
        max_retries:  int        = 2,
    ) -> None:
        client             = LLMClient(backend=backend, model=model, timeout=timeout, debug=debug)
        self._hypothesizer = Hypothesizer(client)
        self._coder        = Coder(client)
        self._critic       = Critic(client)
        self.n_hypotheses  = n_hypotheses
        self.max_retries   = max_retries
        self.debug         = debug
        self.backend       = client.backend
        self.model         = client.model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, task: dict) -> dict:
        """Run the state-machine loop and return a result dict.

        Returns:
            {
                'success':      bool
                'test_correct': bool | None  — None when test output absent
                'code':         str  | None  — first candidate's code, or best partial
                'candidates':   list[dict]   — every program that passed all training pairs
                'log':          list[dict]   — per-transition log entries
            }
        """
        log:            list[dict] = []
        candidates:     list[dict] = []
        best_code:      str | None = None
        best_n_correct: int        = -1

        test_pair             = task.get("test", [{}])[0]
        has_test_ground_truth = "output" in test_pair

        # ==============================================================
        # STATE: HYPOTHESIZING
        # ==============================================================
        task_description = _format_task_description(task)
        try:
            hyp_response = self._hypothesizer.generate(task_description)
        except Exception as e:
            log.append({"state": STATE_HYPOTHESIZING, "error": str(e)})
            return self._make_result(
                success=False, code=best_code, candidates=candidates,
                log=log, test_pair=test_pair, has_ground_truth=has_test_ground_truth,
            )

        hypotheses = _parse_hypotheses(hyp_response)
        log.append({"state": STATE_HYPOTHESIZING, "n_hypotheses": len(hypotheses)})
        if self.debug:
            print(f"[orch] HYPOTHESIZING → {len(hypotheses)} hypothesis(es)")

        # ==============================================================
        # Outer loop: one pass per hypothesis
        # ==============================================================
        for hyp_index, hypothesis in enumerate(hypotheses):
            coder_feedback: str | None = None

            # ==============================================================
            # Inner loop: Coder attempts for this hypothesis
            # ==============================================================
            for attempt in range(1, self.max_retries + 2):

                # ----------------------------------------------------------
                # STATE: CODING
                # ----------------------------------------------------------
                try:
                    code_response = self._coder.generate(hypothesis, coder_feedback)
                except Exception as e:
                    log.append({
                        "state": STATE_CODING, "hyp_index": hyp_index,
                        "attempt": attempt, "error": str(e),
                    })
                    break  # abandon this hypothesis

                coder_feedback = None  # consumed; reset before next iteration

                clean = _strip_thinking(code_response)
                code  = _extract_code(clean) or _extract_code(code_response)

                if code is None:
                    log.append({
                        "state": STATE_CODING, "hyp_index": hyp_index,
                        "attempt": attempt, "error": "no_code_block",
                    })
                    break  # no code produced → abandon hypothesis

                log.append({
                    "state": STATE_CODING, "hyp_index": hyp_index, "attempt": attempt,
                })
                if self.debug:
                    print(f"[orch] CODING    hyp={hyp_index} attempt={attempt}")

                # ----------------------------------------------------------
                # STATE: EVALUATING
                # ----------------------------------------------------------
                eval_result = sandbox.evaluate_code(code, task)
                n_correct   = eval_result["n_correct"]
                n_total     = eval_result["n_total"]

                if n_correct > best_n_correct:
                    best_n_correct = n_correct
                    best_code      = code

                log.append({
                    "state":       STATE_EVALUATING,
                    "hyp_index":   hyp_index,
                    "attempt":     attempt,
                    "n_correct":   n_correct,
                    "n_total":     n_total,
                    "all_correct": eval_result["all_correct"],
                })
                if self.debug:
                    print(
                        f"[orch] EVALUATING hyp={hyp_index} attempt={attempt}: "
                        f"{n_correct}/{n_total}"
                    )

                if eval_result["all_correct"]:
                    candidates.append({
                        "code":             code,
                        "hypothesis":       hypothesis,
                        "hypothesis_index": hyp_index,
                        "attempt":          attempt,
                        "n_correct":        n_correct,
                        "n_total":          n_total,
                    })
                    log.append({
                        "state":     STATE_CANDIDATE,
                        "hyp_index": hyp_index,
                        "attempt":   attempt,
                    })
                    if self.debug:
                        print(f"[orch] CANDIDATE  hyp={hyp_index} saved")
                    break  # hypothesis solved → move to next hypothesis

                # ----------------------------------------------------------
                # STATE: CRITICIZING
                # ----------------------------------------------------------
                try:
                    critic_result = self._critic.analyze(
                        hypothesis, code,
                        _format_error_info(eval_result),
                        _format_diff(eval_result),
                    )
                except Exception as e:
                    log.append({
                        "state": STATE_CRITICIZING, "hyp_index": hyp_index,
                        "attempt": attempt, "error": str(e),
                    })
                    break  # Critic unavailable → abandon hypothesis

                route    = critic_result["route"]
                feedback = critic_result["feedback"]

                log.append({
                    "state":     STATE_CRITICIZING,
                    "hyp_index": hyp_index,
                    "attempt":   attempt,
                    "route":     route,
                    "feedback":  feedback,
                })
                if self.debug:
                    print(f"[orch] CRITICIZING hyp={hyp_index} → {route}")

                if route == ROUTE_HYPOTHESIZER:
                    break  # logic flaw → abandon hypothesis immediately
                else:
                    # Coding bug → carry feedback into the next Coder attempt.
                    # If this was the last attempt the loop exits naturally.
                    coder_feedback = feedback

        # ==============================================================
        # Return
        # ==============================================================
        best_candidate_code = candidates[0]["code"] if candidates else best_code
        return self._make_result(
            success=len(candidates) > 0,
            code=best_candidate_code,
            candidates=candidates,
            log=log,
            test_pair=test_pair,
            has_ground_truth=has_test_ground_truth,
        )

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
        """Run code on the held-out test input and compare to ground truth."""
        out, err = sandbox.execute(code, test_pair["input"])
        if err or out is None:
            return False
        return grids_equal(out, test_pair["output"])

    def _make_result(
        self,
        *,
        success:      bool,
        code:         str | None,
        candidates:   list[dict],
        log:          list[dict],
        test_pair:    dict,
        has_ground_truth: bool,
    ) -> dict:
        test_correct = (
            self._evaluate_test(code, test_pair)
            if has_ground_truth and code else None
        )
        return {
            "success":      success,
            "test_correct": test_correct,
            "code":         code,
            "candidates":   candidates,
            "log":          log,
        }
