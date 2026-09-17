"""Shared SEC EDGAR HTTP plumbing: user-agent, rate-limit pacing, and the
raw GET helpers both acquisition scripts (v1 exploratory and v2
authoritative) use.

Previously defined only in ``scripts/acquire_edgar_8k_yf_megacap_daily.py``
(v1) and imported from there by
``scripts/acquire_sec_filings_12issuer_daily.py`` (v2) -- a real coupling
defect: v1 is classified EXPLORATORY / NON-CONFORMING LEGACY EVIDENCE, so
retiring or refactoring it would silently break v2's acquisition path and
its SEC rate-limit politeness. Moved into the package so both scripts (and
any future one) depend on this module, not on each other.

The User-Agent contact is supplied by the ``SEC_USER_AGENT`` environment
variable at call time (SEC EDGAR requires a descriptive agent carrying a real
contact). No personal contact string is hard-coded here.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any

SEC_SLEEP_SECONDS = 0.25


def user_agent() -> str:
    """Return the SEC User-Agent, sourced from the environment at call time.

    Fails loudly rather than sending a blank User-Agent to SEC (which invites
    throttling/blocking) when the variable is missing or whitespace-only.
    """
    value = os.environ.get("SEC_USER_AGENT", "").strip()
    if not value:
        raise RuntimeError(
            "SEC_USER_AGENT is not set (or is blank). SEC EDGAR requires a "
            "descriptive User-Agent carrying a real contact address, e.g.\n"
            '  export SEC_USER_AGENT="Your Org research you@example.com"'
        )
    return value


# Deprecated: legacy (v1) callers import this name for their manifest record.
# It is no longer a hard-coded contact string, and authoritative v2 code must
# call user_agent() instead so a missing environment value fails loudly.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "").strip()


def sec_get(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent(),
            "Accept-Encoding": "identity",
            "Host": urllib.parse.urlparse(url).netloc,
        },
    )
    # urllib.parse.urlparse is fine; Host override helps some CDNs.
    with urllib.request.urlopen(request, timeout=60) as response:
        return bytes(response.read())


def sec_get_json(url: str) -> dict[str, Any]:
    time.sleep(SEC_SLEEP_SECONDS)
    raw = sec_get(url)
    parsed: dict[str, Any] = json.loads(raw.decode("utf-8"))
    return parsed


def sec_get_text(url: str) -> str:
    time.sleep(SEC_SLEEP_SECONDS)
    raw = sec_get(url)
    return raw.decode("utf-8", errors="replace")
