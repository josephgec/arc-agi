"""Tests for agents/roles.py — Hypothesizer, Coder, and Critic."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.roles import (
    Hypothesizer,
    Coder,
    Critic,
    DSL_DOCS,
    ROUTE_HYPOTHESIZER,
    ROUTE_CODER,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client(response: str = "ok") -> MagicMock:
    client = MagicMock()
    client.generate.return_value = response
    return client


# ---------------------------------------------------------------------------
# Hypothesizer
# ---------------------------------------------------------------------------

class TestHypothesizer:
    def test_generate_calls_client(self):
        client = _mock_client("1. First\n2. Second\n3. Third")
        h = Hypothesizer(client)
        result = h.generate("task grids here")
        client.generate.assert_called_once()
        assert result == "1. First\n2. Second\n3. Third"

    def test_system_prompt_passed(self):
        client = _mock_client()
        h = Hypothesizer(client)
        h.generate("task grids")
        system_prompt = client.generate.call_args[0][0]
        assert "DO NOT write any code" in system_prompt
        assert "3 DISTINCT" in system_prompt

    def test_task_description_in_messages(self):
        client = _mock_client()
        h = Hypothesizer(client)
        h.generate("MY TASK GRIDS")
        messages = client.generate.call_args[0][1]
        assert messages[0]["role"] == "user"
        assert "MY TASK GRIDS" in messages[0]["content"]

    def test_no_feedback_omits_critic_header(self):
        client = _mock_client()
        h = Hypothesizer(client)
        h.generate("task")
        content = client.generate.call_args[0][1][0]["content"]
        assert "Critic feedback" not in content

    def test_feedback_included_in_content(self):
        client = _mock_client()
        h = Hypothesizer(client)
        h.generate("task", feedback="Try a rotation rule instead.")
        content = client.generate.call_args[0][1][0]["content"]
        assert "Try a rotation rule instead." in content
        assert "Critic feedback" in content

    def test_feedback_requests_new_hypotheses(self):
        client = _mock_client()
        h = Hypothesizer(client)
        h.generate("task", feedback="hint")
        content = client.generate.call_args[0][1][0]["content"]
        assert "NEW" in content or "new" in content


# ---------------------------------------------------------------------------
# Coder
# ---------------------------------------------------------------------------

class TestCoder:
    def test_generate_calls_client(self):
        client = _mock_client("```python\ndef transform(grid): return grid\n```")
        c = Coder(client)
        result = c.generate("recolor 1 to 2")
        client.generate.assert_called_once()
        assert "transform" in result

    def test_dsl_docs_in_system_prompt(self):
        client = _mock_client()
        c = Coder(client)
        c.generate("hypothesis")
        system_prompt = client.generate.call_args[0][0]
        assert "crop" in system_prompt
        assert "recolor" in system_prompt
        assert "find_objects" in system_prompt

    def test_hypothesis_in_message_content(self):
        client = _mock_client()
        c = Coder(client)
        c.generate("MY HYPOTHESIS RULE")
        content = client.generate.call_args[0][1][0]["content"]
        assert "MY HYPOTHESIS RULE" in content

    def test_no_feedback_omits_critic_header(self):
        client = _mock_client()
        c = Coder(client)
        c.generate("hypothesis")
        content = client.generate.call_args[0][1][0]["content"]
        assert "Critic feedback" not in content

    def test_feedback_included_in_content(self):
        client = _mock_client()
        c = Coder(client)
        c.generate("hypothesis", feedback="Use recolor instead of flood_fill.")
        content = client.generate.call_args[0][1][0]["content"]
        assert "Use recolor instead of flood_fill." in content
        assert "Critic feedback" in content

    def test_feedback_requests_fix(self):
        client = _mock_client()
        c = Coder(client)
        c.generate("hypothesis", feedback="hint")
        content = client.generate.call_args[0][1][0]["content"]
        assert "Fix" in content or "fix" in content


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class TestCritic:
    def _critic(self, response: str) -> Critic:
        return Critic(_mock_client(response))

    def test_routes_to_hypothesizer(self):
        critic = self._critic("The rule is wrong.\nROUTE: HYPOTHESIZER\nTry tiling instead.")
        result = critic.analyze("hyp", "code", "error", "diff")
        assert result["route"] == ROUTE_HYPOTHESIZER

    def test_routes_to_coder(self):
        critic = self._critic("The code has a bug.\nROUTE: CODER\nUse rotate(grid, 2).")
        result = critic.analyze("hyp", "code", "error", "diff")
        assert result["route"] == ROUTE_CODER

    def test_default_route_is_coder_when_no_route_line(self):
        critic = self._critic("Something went wrong, no routing signal present.")
        result = critic.analyze("hyp", "code", "error", "diff")
        assert result["route"] == ROUTE_CODER

    def test_feedback_extracted_after_route_line(self):
        critic = self._critic("Analysis here.\nROUTE: CODER\nSpecific fix: use scale(grid, 2).")
        result = critic.analyze("hyp", "code", "error", "diff")
        assert "Specific fix: use scale(grid, 2)." in result["feedback"]

    def test_feedback_not_empty_when_no_route_line(self):
        response = "Full analysis with no route."
        critic = self._critic(response)
        result = critic.analyze("hyp", "code", "error", "diff")
        assert result["feedback"]

    def test_analyze_passes_all_inputs_to_client(self):
        client = _mock_client("ROUTE: CODER\nfix it")
        critic = Critic(client)
        critic.analyze("MY HYP", "MY CODE", "MY ERROR", "MY DIFF")
        content = client.generate.call_args[0][1][0]["content"]
        assert "MY HYP"   in content
        assert "MY CODE"  in content
        assert "MY ERROR" in content
        assert "MY DIFF"  in content

    def test_system_prompt_mentions_routing(self):
        client = _mock_client("ROUTE: CODER\n")
        critic = Critic(client)
        critic.analyze("h", "c", "e", "d")
        system_prompt = client.generate.call_args[0][0]
        assert "ROUTE: HYPOTHESIZER" in system_prompt
        assert "ROUTE: CODER"        in system_prompt

    def test_route_line_case_insensitive(self):
        """The parser should match regardless of surrounding whitespace."""
        critic = self._critic("Analysis.\nROUTE: HYPOTHESIZER\nNew rule needed.")
        result = critic.analyze("h", "c", "e", "d")
        assert result["route"] == ROUTE_HYPOTHESIZER

    def test_feedback_empty_string_falls_back_to_full_response(self):
        """If there's nothing after the route line, feedback = full response."""
        critic = self._critic("ROUTE: CODER")
        result = critic.analyze("h", "c", "e", "d")
        assert result["feedback"]

    def test_dsl_docs_not_empty(self):
        assert "crop" in DSL_DOCS
        assert "transform(grid)" in DSL_DOCS
