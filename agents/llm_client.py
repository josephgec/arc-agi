"""LLM backend abstraction for the ARC-AGI multi-agent framework.

Any agent (Analyst, Coder, Critic, …) instantiates LLMClient and calls
generate() to get a completion.  All network and SDK details are hidden
behind that single method so agents stay focused on reasoning, not I/O.

Supported backends
------------------
  "ollama"    — local Ollama server via urllib streaming NDJSON (no API cost)
  "anthropic" — Anthropic cloud API via the `anthropic` SDK

Thinking tokens
---------------
deepseek-r1 and similar models emit reasoning via a separate `thinking`
field in Ollama's NDJSON stream.  generate() collects those tokens,
wraps them in <think>…</think>, and prepends them to the content so that
callers can strip them uniformly with a single regex.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Connection constants
# ---------------------------------------------------------------------------

OLLAMA_CHAT_URL       = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL  = "deepseek-r1:32b"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """Thin, backend-agnostic wrapper around LLM completion APIs.

    Agents call generate() with a system prompt, a conversation history,
    and a temperature.  The client routes to the configured backend and
    returns raw response text (including any <think> tags from reasoning
    models).

    Args:
        backend:    "ollama" or "anthropic".
        model:      Model name override.  Defaults to deepseek-r1:32b /
                    claude-sonnet-4-6 depending on backend.
        temperature: Default sampling temperature.  Overridden per-call when needed.
        max_tokens: Maximum tokens to generate.  Set low for the Coder (8192 — it only
                    outputs code, not reasoning) and high for the Hypothesizer/Critic
                    (32768 — they need space for extended thinking chains).
                    This maps to ``num_predict`` in Ollama and ``max_tokens`` in the
                    Anthropic API.
        timeout:    Seconds to wait for a model response (Ollama only).
        debug:      Print response length diagnostics to stdout.
    """

    def __init__(
        self,
        backend: str = "ollama",
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 32768,
        timeout: float = 120.0,
        debug: bool = False,
    ) -> None:
        self.debug       = debug
        self._timeout    = timeout
        self.temperature = temperature  # role-specific default; overridden per-call when needed
        self.max_tokens  = max_tokens   # token budget: low for Coder, high for Hypothesizer/Critic

        if backend == "ollama":
            self.backend = "ollama"
            self.model   = model or DEFAULT_OLLAMA_MODEL
            self._anthropic = None
        elif backend == "anthropic":
            import anthropic
            self.backend    = "anthropic"
            self.model      = model or DEFAULT_ANTHROPIC_MODEL
            self._anthropic = anthropic.Anthropic()
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose 'ollama' or 'anthropic'."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float | None = None,
    ) -> str:
        """Request a completion and return the response text.

        Args:
            system_prompt: The system instruction prepended to every request.
            messages:      Conversation history in OpenAI-style role/content dicts.
            temperature:   Sampling temperature (0 = greedy).

        Returns:
            Raw response text.  Thinking tokens from reasoning models are
            wrapped in <think>…</think> so callers can strip them uniformly.

        Raises:
            TimeoutError: Ollama backend only — server never responded.
        """
        if temperature is None:
            temperature = self.temperature
        if self.backend == "anthropic":
            return self._call_anthropic(system_prompt, messages, temperature)
        return self._call_ollama(system_prompt, messages, temperature)

    # ------------------------------------------------------------------
    # Private backends
    # ------------------------------------------------------------------

    def _call_anthropic(self, system_prompt: str, messages: list[dict], temperature: float) -> str:
        """Send a request to the Anthropic API and return the response text."""
        response = self._anthropic.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=messages,
            temperature=temperature,
        )
        return response.content[0].text

    def _call_ollama(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float,
    ) -> str:
        """Send a streaming chat request to the local Ollama server.

        Streams NDJSON line-by-line, accumulating content and thinking tokens
        until the server signals done=True or the wall-clock deadline fires.
        Partial results are returned on deadline rather than raising an error,
        so the agent can still attempt code extraction from whatever arrived.

        Thinking tokens (deepseek-r1, qwen3, etc.) arrive in the `thinking`
        field of each chunk and are wrapped in <think>…</think> on return so
        callers can strip them with a single regex.

        Note: no early-exit on code-block detection.  Reasoning models write
        draft code inside <think> blocks before producing their final answer;
        stopping early risks capturing the draft instead of the real solution.

        Raises:
            TimeoutError: if the server never responds at all (empty result).
        """
        import json
        import socket
        import time
        import urllib.error
        import urllib.request

        full_messages = [{"role": "system", "content": system_prompt}] + messages
        payload = json.dumps({
            "model":      self.model,
            "messages":   full_messages,
            "stream":     True,
            "keep_alive": -1,   # never unload the model between calls
            "options":    {
                "temperature": temperature,
                "num_predict": self.max_tokens,
            },
        }).encode()

        req = urllib.request.Request(
            OLLAMA_CHAT_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        thinking_parts: list[str] = []
        content_parts:  list[str] = []

        # Use the full user-specified timeout as the per-read socket timeout.
        # A hard cap at 60 s was previously applied here, but deepseek-r1:32b
        # needs >60 s of prefill on large prompts before emitting its first
        # token, causing all such calls to silently fail.
        _READ_TIMEOUT = self._timeout
        deadline      = time.monotonic() + self._timeout

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

                    if time.monotonic() > deadline:
                        break

        except socket.timeout as e:
            if not thinking_parts and not content_parts:
                raise TimeoutError(
                    f"Ollama call timed out after {self._timeout}s"
                ) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, socket.timeout):
                if not thinking_parts and not content_parts:
                    raise TimeoutError(
                        f"Ollama call timed out after {self._timeout}s"
                    ) from e
            else:
                raise

        thinking = "".join(thinking_parts)
        content  = "".join(content_parts)

        if self.debug:
            print(f"[debug] content={len(content)} chars  thinking={len(thinking)} chars")

        if thinking and content:
            return f"<think>{thinking}</think>\n{content}"
        if thinking:
            # Some model configs (e.g. deepseek-r1:32b via Ollama) place the
            # entire output — reasoning AND final answer — inside the thinking
            # field, leaving content empty.  Returning the raw thinking without
            # tags lets callers (code extractors, hypothesis parsers) find
            # numbered lists and ```python blocks within it directly.
            # _strip_thinking won't remove anything since there are no tags,
            # which is the desired behaviour here.
            return thinking
        return content
