"""Offline tests for the D9-D runtime SEC User-Agent (change D).

The authoritative v2 path takes its SEC contact from the environment and fails
loudly when it is absent, instead of shipping a personal contact string in a
public repo. These tests prove the value is sourced at CALL time and that it
actually reaches the outbound request -- not merely that a formula looks right.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from typing import Literal

import pytest

from quant_sentiment import sec_http

SEC_HTTP_SOURCE = Path(sec_http.__file__).read_text(encoding="utf-8")
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
GOOD = "Fixture Suite research fixtures@example.invalid"


def test_user_agent_reads_environment_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", GOOD)
    assert sec_http.user_agent() == GOOD
    # A later environment change must be picked up by the NEXT call (sourced at
    # call time, not frozen at import time); only outer whitespace is stripped.
    monkeypatch.setenv("SEC_USER_AGENT", f"   {GOOD} second-contact   ")
    assert sec_http.user_agent() == f"{GOOD} second-contact"


def test_user_agent_strips_surrounding_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", f"   {GOOD}\t ")
    assert sec_http.user_agent() == GOOD


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_user_agent_blank_fails_loudly(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", value)
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        sec_http.user_agent()


def test_user_agent_unset_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        sec_http.user_agent()


def test_module_source_has_no_hard_coded_contact() -> None:
    """A surfaced literal address would defeat runtime sourcing (and leak a
    personal contact from a public repo)."""
    code = "\n".join(
        line for line in SEC_HTTP_SOURCE.splitlines() if not line.strip().startswith("#")
    )
    # The one address allowed is the documented placeholder in the failure
    # message ("you@example.com"); any other address would be a real contact.
    found = [address for address in EMAIL.findall(code) if address != "you@example.com"]
    assert not found, found


def test_sec_get_sends_the_environment_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment value must reach the actual outbound request."""
    sent: dict[str, str | None] = {}

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *exc: object) -> Literal[False]:
            return False

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request: urllib.request.Request, timeout: int | None = None) -> _Response:
        sent["user_agent"] = request.get_header("User-agent")
        return _Response()

    monkeypatch.setenv("SEC_USER_AGENT", GOOD)
    monkeypatch.setattr(sec_http.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(sec_http.time, "sleep", lambda _seconds: None)
    sec_http.sec_get_json("https://example.invalid/filings.json")
    assert sent["user_agent"] == GOOD
