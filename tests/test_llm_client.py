"""Tests for agents/llm_client.py.

All network calls (urllib, anthropic SDK) are mocked so no real server
is needed.  These tests form the contract that future agents depend on:
LLMClient.generate() always returns a str and wraps thinking tokens in
<think> tags so callers can strip them uniformly.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.llm_client import (
    LLMClient,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_ANTHROPIC_MODEL,
    OLLAMA_CHAT_URL,
)


# ---------------------------------------------------------------------------
# Streaming mock helpers
# ---------------------------------------------------------------------------

def _ollama_stream(*content_pieces: str, thinking: str = "") -> MagicMock:
    """Build a urllib context-manager mock yielding one NDJSON chunk per piece.

    Args:
        *content_pieces: Content strings emitted as separate stream chunks.
        thinking:        A single thinking string emitted before content chunks.
    """
    lines: list[bytes] = []
    if thinking:
        lines.append(
            json.dumps({"message": {"content": "", "thinking": thinking}, "done": False}).encode()
            + b"\n"
        )
    for piece in content_pieces:
        lines.append(
            json.dumps({"message": {"content": piece, "thinking": ""}, "done": False}).encode()
            + b"\n"
        )
    lines.append(
        json.dumps({"message": {"content": "", "thinking": ""}, "done": True}).encode() + b"\n"
    )
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__  = MagicMock(return_value=False)
    cm.__iter__  = lambda self: iter(lines)
    return cm


# ---------------------------------------------------------------------------
# LLMClient.__init__
# ---------------------------------------------------------------------------

class TestLLMClientInit:
    def test_ollama_backend_default_model(self):
        client = LLMClient(backend="ollama")
        assert client.backend == "ollama"
        assert client.model == DEFAULT_OLLAMA_MODEL

    def test_ollama_model_override(self):
        client = LLMClient(backend="ollama", model="qwen2.5:32b")
        assert client.model == "qwen2.5:32b"

    def test_anthropic_backend_default_model(self):
        with patch("anthropic.Anthropic"):
            client = LLMClient(backend="anthropic")
        assert client.backend == "anthropic"
        assert client.model == DEFAULT_ANTHROPIC_MODEL

    def test_anthropic_model_override(self):
        with patch("anthropic.Anthropic"):
            client = LLMClient(backend="anthropic", model="claude-opus-4-6")
        assert client.model == "claude-opus-4-6"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            LLMClient(backend="groq")

    def test_timeout_stored(self):
        client = LLMClient(backend="ollama", timeout=99.0)
        assert client._timeout == 99.0

    def test_temperature_defaults_zero(self):
        client = LLMClient(backend="ollama")
        assert client.temperature == 0.0

    def test_temperature_stored(self):
        client = LLMClient(backend="ollama", temperature=0.6)
        assert client.temperature == pytest.approx(0.6)

    def test_max_tokens_default(self):
        client = LLMClient(backend="ollama")
        assert client.max_tokens == 32768

    def test_max_tokens_stored(self):
        client = LLMClient(backend="ollama", max_tokens=8192)
        assert client.max_tokens == 8192

    def test_debug_defaults_false(self):
        client = LLMClient(backend="ollama")
        assert client.debug is False

    def test_debug_flag_stored(self):
        client = LLMClient(backend="ollama", debug=True)
        assert client.debug is True

    def test_ollama_has_no_anthropic_client(self):
        client = LLMClient(backend="ollama")
        assert client._anthropic is None


# ---------------------------------------------------------------------------
# generate() — Ollama backend (urllib mocked)
# ---------------------------------------------------------------------------

class TestGenerateOllama:
    def _client(self, timeout: float = 10.0) -> LLMClient:
        return LLMClient(backend="ollama", timeout=timeout)

    def test_content_returned(self):
        client = self._client()
        with patch("urllib.request.urlopen", return_value=_ollama_stream("Hello world")):
            result = client.generate("sys", [{"role": "user", "content": "hi"}])
        assert result == "Hello world"

    def test_thinking_only_returned_raw(self):
        # When content is empty the model put its final answer inside the
        # thinking field (deepseek-r1:32b behaviour).  Return raw thinking
        # without tags so parsers can still find hypotheses and code blocks.
        client = self._client()
        with patch("urllib.request.urlopen", return_value=_ollama_stream(thinking="I reasoned")):
            result = client.generate("sys", [{"role": "user", "content": "hi"}])
        assert result == "I reasoned"

    def test_thinking_and_content_combined(self):
        client = self._client()
        with patch("urllib.request.urlopen",
                   return_value=_ollama_stream("Answer", thinking="reasoning")):
            result = client.generate("sys", [{"role": "user", "content": "hi"}])
        assert "<think>reasoning</think>" in result
        assert "Answer" in result

    def test_empty_response_returns_empty_string(self):
        client = self._client()
        with patch("urllib.request.urlopen", return_value=_ollama_stream()):
            result = client.generate("sys", [{"role": "user", "content": "hi"}])
        assert result == ""

    def test_multiple_content_chunks_concatenated(self):
        """Chunks streaming in separately must be joined into one string."""
        client = self._client()
        with patch("urllib.request.urlopen",
                   return_value=_ollama_stream("Hello", " ", "world")):
            result = client.generate("sys", [{"role": "user", "content": "hi"}])
        assert result == "Hello world"

    def test_timeout_raises_timeout_error(self):
        import socket
        client = self._client()
        with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
            with pytest.raises(TimeoutError):
                client.generate("sys", [{"role": "user", "content": "hi"}])

    def test_partial_result_returned_on_deadline(self):
        """When the wall-clock deadline fires, partial data is returned (not raised)."""
        import time
        client = self._client(timeout=0.001)  # nearly-zero deadline

        thinking_chunk = (
            json.dumps({"message": {"content": "", "thinking": "partial"}, "done": False}).encode()
            + b"\n"
        )
        done_chunk = (
            json.dumps({"message": {"content": "", "thinking": ""}, "done": True}).encode()
            + b"\n"
        )

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__  = MagicMock(return_value=False)

        def slow_iter():
            yield thinking_chunk
            time.sleep(0.05)
            yield done_chunk

        cm.__iter__ = lambda self: slow_iter()

        with patch("urllib.request.urlopen", return_value=cm):
            result = client.generate("sys", [{"role": "user", "content": "hi"}])

        assert "partial" in result

    def test_full_think_block_preserved_in_stream(self):
        """When a model streams both <think>draft</think> and a final answer,
        both must be present in the raw return value so _strip_thinking can work."""
        client = self._client()
        think_chunk = "<think>\ndef transform(g): return g * 0\n</think>\n"
        final_chunk = "```python\ndef transform(g):\n    return g.copy()\n```"

        with patch("urllib.request.urlopen",
                   return_value=_ollama_stream(think_chunk, final_chunk)):
            result = client.generate("sys", [{"role": "user", "content": "hi"}])

        assert "return g * 0"   in result
        assert "return g.copy()" in result

    def test_no_early_exit_on_first_code_block(self):
        """Without early-exit, trailing content after the first code block
        must still be included in the return value."""
        client = self._client()
        first_block = "```python\ndef transform(g):\n    return g.copy()\n```"
        trailing    = " extra content after code"

        with patch("urllib.request.urlopen",
                   return_value=_ollama_stream(first_block, trailing)):
            result = client.generate("sys", [{"role": "user", "content": "hi"}])

        assert "return g.copy()" in result
        assert "extra content after code" in result

    def test_system_prompt_included_in_request(self):
        """The system_prompt parameter must be sent as the first message."""
        client = self._client()
        captured = {}

        def fake_urlopen(req, timeout):
            import json as _json
            body = _json.loads(req.data.decode())
            captured["messages"] = body["messages"]
            return _ollama_stream("ok")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.generate("MY SYSTEM PROMPT", [{"role": "user", "content": "hi"}])

        assert captured["messages"][0]["role"] == "system"
        assert captured["messages"][0]["content"] == "MY SYSTEM PROMPT"

    def test_temperature_included_in_request(self):
        """The temperature value must be forwarded to the Ollama options."""
        client = self._client()
        captured = {}

        def fake_urlopen(req, timeout):
            import json as _json
            body = _json.loads(req.data.decode())
            captured["options"] = body["options"]
            return _ollama_stream("ok")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.generate("sys", [{"role": "user", "content": "hi"}], temperature=0.7)

        assert captured["options"]["temperature"] == pytest.approx(0.7)

    def test_stored_temperature_used_when_not_passed(self):
        """When generate() is called without temperature, client.temperature is used."""
        client = LLMClient(backend="ollama", temperature=0.6)
        captured = {}

        def fake_urlopen(req, timeout):
            import json as _json
            body = _json.loads(req.data.decode())
            captured["options"] = body["options"]
            return _ollama_stream("ok")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.generate("sys", [{"role": "user", "content": "hi"}])

        assert captured["options"]["temperature"] == pytest.approx(0.6)

    def test_max_tokens_forwarded_as_num_predict(self):
        """max_tokens must be sent to Ollama as options.num_predict."""
        client = LLMClient(backend="ollama", max_tokens=8192)
        captured = {}

        def fake_urlopen(req, timeout):
            import json as _json
            body = _json.loads(req.data.decode())
            captured["options"] = body["options"]
            return _ollama_stream("ok")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.generate("sys", [{"role": "user", "content": "hi"}])

        assert captured["options"]["num_predict"] == 8192


# ---------------------------------------------------------------------------
# generate() — Anthropic backend (SDK mocked)
# ---------------------------------------------------------------------------

class TestGenerateAnthropic:
    def _client(self) -> LLMClient:
        with patch("anthropic.Anthropic"):
            client = LLMClient(backend="anthropic")
        client._anthropic = MagicMock()
        return client

    def _mock_api_response(self, text: str) -> MagicMock:
        resp = MagicMock()
        resp.content = [MagicMock(text=text)]
        return resp

    def test_returns_response_text(self):
        client = self._client()
        client._anthropic.messages.create.return_value = self._mock_api_response("hello")
        result = client.generate("sys", [{"role": "user", "content": "hi"}])
        assert result == "hello"

    def test_system_prompt_passed_to_api(self):
        client = self._client()
        client._anthropic.messages.create.return_value = self._mock_api_response("x")
        client.generate("MY SYSTEM", [{"role": "user", "content": "hi"}])
        call_kwargs = client._anthropic.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "MY SYSTEM"

    def test_messages_passed_to_api(self):
        client = self._client()
        client._anthropic.messages.create.return_value = self._mock_api_response("x")
        msgs = [{"role": "user", "content": "solve it"}]
        client.generate("sys", msgs)
        call_kwargs = client._anthropic.messages.create.call_args.kwargs
        assert call_kwargs["messages"] == msgs

    def test_model_passed_to_api(self):
        client = self._client()
        client._anthropic.messages.create.return_value = self._mock_api_response("x")
        client.generate("sys", [{"role": "user", "content": "hi"}])
        call_kwargs = client._anthropic.messages.create.call_args.kwargs
        assert call_kwargs["model"] == client.model

    def test_temperature_passed_to_api(self):
        """Temperature must be forwarded so greedy (0.0) and random (1.0) differ."""
        client = self._client()
        client._anthropic.messages.create.return_value = self._mock_api_response("x")
        client.generate("sys", [{"role": "user", "content": "hi"}], temperature=0.4)
        call_kwargs = client._anthropic.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == pytest.approx(0.4)

    def test_max_tokens_passed_to_api(self):
        """max_tokens must be forwarded to the Anthropic API."""
        with patch("anthropic.Anthropic"):
            client = LLMClient(backend="anthropic", max_tokens=8192)
        client._anthropic = MagicMock()
        client._anthropic.messages.create.return_value = self._mock_api_response("x")
        client.generate("sys", [{"role": "user", "content": "hi"}])
        call_kwargs = client._anthropic.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 8192
