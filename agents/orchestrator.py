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
    _format_training_examples,
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

    Each agent role uses a dedicated model and temperature, matched to the
    cognitive demands of the task:
      - Hypothesizer: deepseek-r1:32b at temp 0.6 — creative spatial reasoning
      - Coder:        qwen2.5-coder:14b at temp 0.1 — deterministic code output
      - Critic:       deepseek-r1:14b at temp 0.2 — nuanced failure diagnosis

    Args:
        backend:                  'ollama' or 'anthropic'.
        model:                    Fallback model for any role that doesn't specify one.
                                  Defaults to the backend's built-in default.
        hypothesizer_model:       Model for the Hypothesizer role.
        coder_model:              Model for the Coder role.
        critic_model:             Model for the Critic role.
        hypothesizer_temperature: Sampling temperature for the Hypothesizer (default 0.6).
        coder_temperature:        Sampling temperature for the Coder (default 0.1).
                                  Used as the base of the retry temperature ramp.
        critic_temperature:       Sampling temperature for the Critic (default 0.2).
        timeout:                  Seconds per LLM call (Ollama only).
        debug:                    Print state transitions to stdout.
        n_hypotheses:             Number of hypotheses to request from the Hypothesizer.
        max_retries:              Maximum additional Coder attempts per hypothesis after
                                  the first (total = max_retries + 1).
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
        timeout:                  float      = 120.0,
        debug:                    bool       = False,
        n_hypotheses:             int        = 3,
        max_retries:              int        = 2,
    ) -> None:
        def _make_client(role_model: str | None, temperature: float) -> LLMClient:
            return LLMClient(
                backend=backend,
                model=role_model or model,
                temperature=temperature,
                timeout=timeout,
                debug=debug,
            )

        hyp_client = _make_client(hypothesizer_model, hypothesizer_temperature)
        cod_client = _make_client(coder_model,        coder_temperature)
        cri_client = _make_client(critic_model,       critic_temperature)

        self._hypothesizer           = Hypothesizer(hyp_client)
        self._coder                  = Coder(cod_client)
        self._critic                 = Critic(cri_client)
        self.n_hypotheses            = n_hypotheses
        self.max_retries             = max_retries
        self.debug                   = debug
        self.backend                 = backend
        self.hypothesizer_model      = hyp_client.model
        self.coder_model             = cod_client.model
        self.critic_model            = cri_client.model
        self.hypothesizer_temperature = hypothesizer_temperature
        self.coder_temperature        = coder_temperature
        self.critic_temperature       = critic_temperature
        self.model                   = self.hypothesizer_model  # backward-compat alias

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
        task_description   = _format_task_description(task)
        training_examples  = _format_training_examples(task)
        try:
            hyp_response = self._hypothesizer.generate(task_description)
        except Exception as e:
            log.append({"state": STATE_HYPOTHESIZING, "error": str(e)})
            return self._make_result(
                success=False, code=best_code, candidates=candidates,
                log=log, test_pair=test_pair, has_ground_truth=has_test_ground_truth,
            )

        hypotheses = _parse_hypotheses(hyp_response, max_n=self.n_hypotheses)
        log.append({"state": STATE_HYPOTHESIZING, "n_hypotheses": len(hypotheses)})
        if self.debug:
            print(f"[orch] HYPOTHESIZING → {len(hypotheses)} hypothesis(es)")

        # ==============================================================
        # Outer loop: one pass per hypothesis
        # ==============================================================
        for hyp_index, hypothesis in enumerate(hypotheses):
            coder_feedback:  str | None = None
            prev_n_correct:  int        = -1
            no_improve_count: int       = 0

            # ==============================================================
            # Inner loop: Coder attempts for this hypothesis
            # ==============================================================
            for attempt in range(1, self.max_retries + 2):
                is_last_attempt = (attempt == self.max_retries + 1)

                # Temperature diversity: start from role baseline, ramp up on retries.
                temperature = min(self.coder_temperature + (attempt - 1) * 0.3, 0.9)

                # ----------------------------------------------------------
                # STATE: CODING
                # ----------------------------------------------------------
                try:
                    code_response = self._coder.generate(
                        hypothesis, coder_feedback,
                        training_context=training_examples,
                        temperature=temperature,
                    )
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

                # Track consecutive non-improvement to detect a stuck hypothesis.
                if n_correct <= prev_n_correct:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                prev_n_correct = n_correct

                # Skip the Critic when it can't help: last attempt (its feedback
                # would be discarded) or the hypothesis is clearly stuck (0 correct
                # across multiple attempts suggests a logic flaw, not a code bug).
                _hypothesis_stuck = (n_correct == 0 and no_improve_count >= 2)
                if is_last_attempt or _hypothesis_stuck:
                    if self.debug and _hypothesis_stuck:
                        print(f"[orch] STUCK      hyp={hyp_index} — skipping Critic")
                    break

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
