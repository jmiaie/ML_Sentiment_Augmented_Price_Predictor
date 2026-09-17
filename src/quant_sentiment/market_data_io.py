"""yfinance OHLCV download, validation, and CSV write helpers.

Shared by both acquisition scripts (v1 exploratory and v2 authoritative) so
neither depends on the other for this plumbing -- see ``hashing.py``'s
docstring for why that coupling was a real defect, not just style.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


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
