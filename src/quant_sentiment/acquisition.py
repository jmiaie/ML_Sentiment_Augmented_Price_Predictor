"""Shared acquisition helpers used by the authoritative v2 SEC/price
acquisition path (``scripts/acquire_sec_filings_12issuer_daily.py``).

Previously these lived only in ``scripts/acquire_edgar_8k_yf_megacap_daily.py``
(the v1 exploratory script) and were imported from there by the v2 script -- a
real coupling defect (D-4/C): v1 is classified EXPLORATORY / NON-CONFORMING
LEGACY EVIDENCE, so retiring or refactoring it would silently break the
authoritative acquisition path and its SEC/Yahoo politeness.

Both scripts now depend on this module, not on each other.

Deliberate duplication note: the v1 script retains its own copies, because v1
source is treated as frozen legacy evidence and must stay byte-identical. The
package copy here is authoritative for every v2 run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from .sec_http import sec_get_json as _sec_get_json

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_dataset_hash(file_hashes: dict[str, str]) -> str:
    payload = "\n".join(f"{k}:{v}" for k, v in sorted(file_hashes.items())) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fetch_company_filings(cik: str) -> list[dict[str, Any]]:
    """Return filing metadata rows from submissions recent + historical shards."""
    padded = cik.zfill(10)
    submissions = _sec_get_json(f"https://data.sec.gov/submissions/CIK{padded}.json")
    rows = _filings_from_block(submissions.get("filings", {}).get("recent", {}))
    for shard in submissions.get("filings", {}).get("files", []) or []:
        name = shard.get("name")
        if not name:
            continue
        shard_url = f"https://data.sec.gov/submissions/{name}"
        block = _sec_get_json(shard_url)
        rows.extend(_filings_from_block(block))
    return rows


def _filings_from_block(block: dict[str, Any]) -> list[dict[str, Any]]:
    if not block:
        return []
    forms = block.get("form") or []
    n = len(forms)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append(
            {
                "form": forms[i],
                "filingDate": (block.get("filingDate") or [None] * n)[i],
                "acceptanceDateTime": (block.get("acceptanceDateTime") or [None] * n)[i],
                "accessionNumber": (block.get("accessionNumber") or [None] * n)[i],
                "primaryDocument": (block.get("primaryDocument") or [None] * n)[i],
                "reportDate": (block.get("reportDate") or [None] * n)[i],
            }
        )
    return rows


def filter_filings(
    rows: list[dict[str, Any]],
    *,
    start: str,
    end_exclusive: str,
    forms: tuple[str, ...],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        form = row.get("form")
        filing_date = row.get("filingDate") or ""
        if form not in forms:
            continue
        if not filing_date:
            continue
        if not (filing_date >= start and end_exclusive > filing_date):
            continue
        if not row.get("accessionNumber") or not row.get("primaryDocument"):
            continue
        if not row.get("acceptanceDateTime"):
            continue
        out.append(row)
    # Stable order
    out.sort(key=lambda r: (r["filingDate"], r["accessionNumber"]))
    return out


def filing_archive_url(cik: str, accession: str, primary_document: str) -> str:
    cik_int = int(cik)
    acc_nodash = accession.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
        f"{acc_nodash}/{primary_document}"
    )


def _flatten_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        level0 = out.columns.get_level_values(0)
        level1 = out.columns.get_level_values(1)
        if symbol in set(level1.astype(str)):
            out.columns = [
                str(a) if str(b) == symbol else f"{a}_{b}"
                for a, b in zip(level0, level1, strict=True)
            ]
        else:
            out.columns = [str(c[0]) for c in out.columns]
    out.columns = [str(c).strip() for c in out.columns]
    rename = {}
    for c in out.columns:
        cl = c.lower().replace(" ", "_")
        if cl == "adj_close":
            rename[c] = "Adj Close"
        elif cl == "stock_splits":
            rename[c] = "Stock Splits"
        elif cl == "capital_gains":
            rename[c] = "Capital Gains"
    if rename:
        out = out.rename(columns=rename)
    return out


def download_prices(
    symbol: str,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        actions=True,
        repair=False,
        keepna=True,
        progress=False,
        threads=False,
        group_by="column",
    )
    if raw is None or raw.empty:
        raise ValueError(f"No price data for {symbol}")
    df = _flatten_columns(raw, symbol)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    df = df.loc[(df.index >= start_ts) & (end_ts > df.index)]
    missing = [c for c in OHLCV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol} missing columns: {missing}")
    return df


def validate_prices(symbol: str, df: pd.DataFrame) -> dict[str, Any]:
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{symbol}: index not monotonic")
    if df.index.has_duplicates:
        raise ValueError(f"{symbol}: duplicate timestamps")
    missing = int(df[OHLCV_COLS].isna().sum().sum())
    return {
        "row_count": len(df),
        "missing_ohlcv_cells": missing,
        "actual_start": df.index.min().strftime("%Y-%m-%d"),
        "actual_end": df.index.max().strftime("%Y-%m-%d"),
        "columns": list(df.columns),
    }


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = "Date"
    out.to_csv(path, float_format="%.8f")
