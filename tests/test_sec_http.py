"""Offline tests for quant_sentiment.sec_http's runtime user-agent lookup.

Real defect found by independent review (Hai/Codex, tracker Issue #3,
2026-09-17): SEC_USER_AGENT was a hard-coded personal contact string in
authoritative v2 code. It must now come from the SEC_USER_AGENT
environment variable and fail clearly -- before any network call -- when
absent or blank.
"""

from __future__ import annotations

import pytest

from quant_sentiment.sec_http import get_sec_user_agent


def test_get_sec_user_agent_returns_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "Example Org research contact@example.com")
    assert get_sec_user_agent() == "Example Org research contact@example.com"


def test_get_sec_user_agent_raises_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        get_sec_user_agent()


def test_get_sec_user_agent_raises_when_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "   ")
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        get_sec_user_agent()
