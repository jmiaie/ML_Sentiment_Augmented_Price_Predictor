"""SEC EDGAR HTML/XBRL text extraction helpers (offline-friendly)."""

from __future__ import annotations

import re
import warnings
from html import unescape

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

_OPEN = chr(60)
_CLOSE = chr(62)
_TAG_RE = re.compile(_OPEN + "[^" + _CLOSE + "]+" + _CLOSE)
_WS_RE = re.compile(r"\s+")


def html_to_plain_text(document: str, *, max_chars: int = 200_000) -> str:
    """Extract plain text from an EDGAR primary document.

    Uses BeautifulSoup when available; falls back to naive tag stripping.
    Truncates to ``max_chars`` for reproducible hashing / storage bounds.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        try:
            soup = BeautifulSoup(document, "lxml")
        except Exception:
            soup = BeautifulSoup(document, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(" ")
    text = unescape(text)
    text = _WS_RE.sub(" ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def naive_strip_tags(document: str, *, max_chars: int = 200_000) -> str:
    text = _TAG_RE.sub(" ", document)
    text = unescape(text)
    text = _WS_RE.sub(" ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text
