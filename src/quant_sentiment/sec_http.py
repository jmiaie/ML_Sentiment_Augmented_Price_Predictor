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
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any

SEC_USER_AGENT = "jmiaie ML_Sentiment_D9 research jmilam.emba@gmail.com"
SEC_SLEEP_SECONDS = 0.25


def sec_get(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": SEC_USER_AGENT,
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
