"""Offline tests for Directive #9 acquisition helpers (no network)."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "acquire_edgar_8k_yf_megacap_daily.py"
)


def _load_acquire_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "acquire_edgar_8k_yf_megacap_daily", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def acq() -> ModuleType:
    return _load_acquire_module()


def _sample_frame(n: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2015-01-02", periods=n, freq="B")
    rng = np.random.default_rng(0)
    prices = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "Open": prices - 0.1,
            "High": prices + 0.2,
            "Low": prices - 0.2,
            "Close": prices,
            "Volume": rng.integers(1_000_000, 2_000_000, n),
        },
        index=dates,
    )


def test_validate_prices_ok(acq: ModuleType) -> None:
    stats = acq.validate_prices("AAPL", _sample_frame())
    assert stats["row_count"] == 10
    assert stats["missing_ohlcv_cells"] == 0


def test_canonical_hash_stable(acq: ModuleType) -> None:
    h1 = acq.canonical_dataset_hash({"A.csv": "aaa", "B.csv": "bbb"})
    h2 = acq.canonical_dataset_hash({"B.csv": "bbb", "A.csv": "aaa"})
    assert h1 == h2
    assert len(h1) == 64


def test_default_symbols_and_ciks(acq: ModuleType) -> None:
    assert acq.DEFAULT_SYMBOLS == ["AAPL", "MSFT", "AMZN"]
    for symbol in acq.DEFAULT_SYMBOLS:
        assert symbol in acq.SYMBOL_CIK


def test_filter_filings_date_and_form(acq: ModuleType) -> None:
    rows: list[dict[str, Any]] = [
        {
            "form": "8-K",
            "filingDate": "2020-01-02",
            "acceptanceDateTime": "2020-01-02T21:00:00.000Z",
            "accessionNumber": "0001",
            "primaryDocument": "a.htm",
            "reportDate": "2020-01-02",
        },
        {
            "form": "10-K",
            "filingDate": "2020-01-02",
            "acceptanceDateTime": "2020-01-02T21:00:00.000Z",
            "accessionNumber": "0002",
            "primaryDocument": "b.htm",
            "reportDate": "2020-01-02",
        },
        {
            "form": "8-K",
            "filingDate": "2014-12-31",
            "acceptanceDateTime": "2014-12-31T21:00:00.000Z",
            "accessionNumber": "0003",
            "primaryDocument": "c.htm",
            "reportDate": "2014-12-31",
        },
        {
            "form": "8-K",
            "filingDate": "2025-06-01",
            "acceptanceDateTime": "2025-06-01T21:00:00.000Z",
            "accessionNumber": "0004",
            "primaryDocument": "d.htm",
            "reportDate": "2025-06-01",
        },
    ]
    filtered = acq.filter_filings(
        rows, start="2015-01-01", end_exclusive="2026-01-01", forms=("8-K",)
    )
    assert [r["accessionNumber"] for r in filtered] == ["0001", "0004"]


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PLACEHOLDER_DOMAINS = ("example.com", "example.invalid", "example.org")


def test_no_plaintext_contact_in_committed_data() -> None:
    """A plaintext personal contact address was once written into a frozen, published
    manifest (the SEC User-Agent); the acquisition writers now record a SHA-256 of it
    instead. This fails if any committed artifact under data/ regresses to a real one."""
    data = Path(__file__).resolve().parents[1] / "data"
    offenders = [
        f"{path.relative_to(data)}: {email}"
        for path in data.rglob("*")
        if path.is_file()
        for email in _EMAIL_RE.findall(path.read_text(encoding="utf-8", errors="ignore"))
        if not email.lower().split("@")[1].endswith(_PLACEHOLDER_DOMAINS)
    ]
    assert offenders == [], f"plaintext contact address in committed artifact: {offenders}"
