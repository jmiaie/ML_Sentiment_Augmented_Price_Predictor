#!/usr/bin/env python3
"""Acquire and freeze the D9-D AUTHORITATIVE (v2) datasets for Directive #9:

  * ``sec_filings_12issuer_2015_2025_v1`` -- SEC EDGAR 10-K/10-Q/8-K primary
    document text + full event metadata (CIK, ticker, company, accession,
    form, filing date, acceptance/availability timestamp, retrieval
    timestamp, source document URL, SHA-256) for the authoritative
    12-issuer universe.
  * ``yf_sentiment_equities_daily_2015_2025_v1`` -- yfinance daily OHLCV for
    the same 12 issuers plus SPY (market context).

THIS SCRIPT REQUIRES LIVE NETWORK ACCESS TO data.sec.gov / www.sec.gov AND
Yahoo Finance, WHICH THIS SESSION'S EGRESS PROXY BLOCKS (confirmed: both
SEC hosts return a 403 CONNECT-tunnel rejection here). It is therefore
prepared as a HANDOFF: run it from an environment with real SEC EDGAR
access, then hand back data/manifests/*.json plus data/raw/ (gitignored,
do not commit) so this repo's v2 study can run against real filings.

Deliberately does NOT score filing text at acquisition time (no lexicon,
no LM dictionary call here) -- acquisition only fetches and freezes raw
inputs; feature extraction (real Loughran-McDonald categories via
quant_sentiment.lm_dictionary) happens downstream in the v2 study, reading
these frozen files. This is a cleaner separation than v1's acquisition
script, which scored with the (now-superseded) embedded lexicon inline.

Reuses acquire_edgar_8k_yf_megacap_daily.py's already-tested SEC/yfinance
plumbing (submission fetch, filing filter, price download/validation,
hashing) via direct import rather than duplicating it.

IMPORTANT -- CIK VERIFICATION: AAPL/MSFT/AMZN's CIKs below match this
repo's already-frozen v1 acquisition exactly. The other 9 issuers' CIKs
were NOT independently verified against a live SEC source in the session
that wrote this script (network to SEC was blocked there too) -- verify
each against https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany
or data.sec.gov/submissions/CIK##########.json (confirm the returned
``name``/``tickers`` match) BEFORE running this script for real.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from acquire_edgar_8k_yf_megacap_daily import (  # noqa: E402
    SEC_USER_AGENT,
    canonical_dataset_hash,
    download_prices,
    fetch_company_filings,
    filing_archive_url,
    filter_filings,
    sha256_file,
    validate_prices,
    write_csv,
)

from quant_sentiment.edgar_text import html_to_plain_text  # noqa: E402

FILINGS_DATASET_ID = "sec_filings_12issuer_2015_2025_v1"
PRICES_DATASET_ID = "yf_sentiment_equities_daily_2015_2025_v1"

# Exactly the 12 issuers named in Directive #9's D9-D spec, plus SPY for
# market context (SPY has no filing acquisition -- prices only).
SYMBOL_CIK: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "NVDA": "0001045810",
    "JPM": "0000019617",
    "XOM": "0000034088",
    "JNJ": "0000200406",
    "PG": "0000080424",
    "WMT": "0000104169",
    "HD": "0000354950",
    "KO": "0000021344",
}
SYMBOL_COMPANY: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc.",
    "NVDA": "NVIDIA Corporation",
    "JPM": "JPMorgan Chase & Co.",
    "XOM": "Exxon Mobil Corporation",
    "JNJ": "Johnson & Johnson",
    "PG": "The Procter & Gamble Company",
    "WMT": "Walmart Inc.",
    "HD": "The Home Depot, Inc.",
    "KO": "The Coca-Cola Company",
}
MARKET_CONTEXT_SYMBOL = "SPY"
FILING_SYMBOLS = list(SYMBOL_CIK)
PRICE_SYMBOLS = [*FILING_SYMBOLS, MARKET_CONTEXT_SYMBOL]
FORM_TYPES = ("10-K", "10-Q", "8-K")
REQUESTED_START = "2015-01-01"
REQUESTED_END_EXCLUSIVE = "2026-01-01"
TEXT_MAX_CHARS = 200_000


def _repo_root() -> Path:
    return ROOT


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def acquire_symbol_filings_raw(
    symbol: str,
    cik: str,
    *,
    start: str,
    end: str,
    text_dir: Path,
    max_filings: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch + freeze raw filing text and full event metadata for one
    issuer -- no sentiment/text scoring here (see module docstring)."""
    rows = filter_filings(
        fetch_company_filings(cik),
        start=start,
        end_exclusive=end,
        forms=FORM_TYPES,
    )
    if max_filings is not None:
        rows = rows[:max_filings]

    text_dir.mkdir(parents=True, exist_ok=True)
    event_records: list[dict[str, Any]] = []
    failures = 0
    retrieval_ts = _utc_now()

    for row in rows:
        accession = str(row["accessionNumber"])
        primary = str(row["primaryDocument"])
        url = filing_archive_url(cik, accession, primary)
        safe_name = accession.replace("-", "") + ".txt"
        text_path = text_dir / safe_name
        try:
            import urllib.error

            if text_path.exists():
                plain = text_path.read_text(encoding="utf-8")
            else:
                from acquire_edgar_8k_yf_megacap_daily import _sec_get_text

                html = _sec_get_text(url)
                plain = html_to_plain_text(html, max_chars=TEXT_MAX_CHARS)
                text_path.write_text(plain, encoding="utf-8")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            failures += 1
            event_records.append(
                {
                    "cik": cik,
                    "ticker": symbol,
                    "company": SYMBOL_COMPANY[symbol],
                    "accession": accession,
                    "form": row["form"],
                    "filing_date": row["filingDate"],
                    "acceptance_datetime": row["acceptanceDateTime"],
                    "retrieval_timestamp_utc": retrieval_ts,
                    "source_document_url": url,
                    "sha256": None,
                    "text_chars": 0,
                    "status": f"FAILED:{type(exc).__name__}",
                }
            )
            continue

        event_records.append(
            {
                "cik": cik,
                "ticker": symbol,
                "company": SYMBOL_COMPANY[symbol],
                "accession": accession,
                "form": row["form"],
                "filing_date": row["filingDate"],
                "acceptance_datetime": row["acceptanceDateTime"],
                "retrieval_timestamp_utc": retrieval_ts,
                "source_document_url": url,
                "sha256": sha256_file(text_path),
                "text_chars": len(plain),
                "text_relpath": f"filings/text/{symbol}/{safe_name}",
                "status": "OK",
            }
        )

    index = pd.DataFrame.from_records(event_records)
    stats = {
        "filings_listed": len(rows),
        "filings_ok": int((index["status"] == "OK").sum()) if not index.empty else 0,
        "filings_failed": failures,
        "by_form": (
            index.loc[index["status"] == "OK", "form"].value_counts().to_dict()
            if not index.empty
            else {}
        ),
        "first_filing": index["filing_date"].min() if not index.empty else None,
        "last_filing": index["filing_date"].max() if not index.empty else None,
    }
    return index, {"stats": stats}


def build_filings_manifest(
    *,
    symbols: list[str],
    retrieval_ts: str,
    freeze_ts: str | None,
    status: str,
    filing_stats: dict[str, dict[str, Any]],
    parameters: dict[str, Any],
    sha256: dict[str, str] | None,
) -> dict[str, Any]:
    return {
        "dataset_id": FILINGS_DATASET_ID,
        "source": "sec_edgar",
        "source_version": {"sec_data_api": "data.sec.gov/submissions"},
        "universe_description": (
            "Directive #9 D9-D authoritative 12-issuer universe: AAPL, MSFT, AMZN, GOOGL, "
            "NVDA, JPM, XOM, JNJ, PG, WMT, HD, KO. This is a static, present-day-selected "
            "universe of large, currently-listed issuers -- it therefore carries "
            "survivorship/selection bias (no delisted, merged, or historically-relevant-"
            "but-no-longer-prominent issuers are included), which is disclosed here "
            "explicitly, not hidden."
        ),
        "symbols": symbols,
        "ciks": {s: SYMBOL_CIK[s] for s in symbols},
        "companies": {s: SYMBOL_COMPANY[s] for s in symbols},
        "form_types": list(FORM_TYPES),
        "requested_start": parameters["start"],
        "requested_end_exclusive": parameters["end"],
        "filing_stats": filing_stats,
        "retrieval_timestamp_utc": retrieval_ts,
        "freeze_timestamp_utc": freeze_ts,
        "status": status,
        "sha256": sha256,
        "parameters": parameters,
        "event_record_fields": [
            "cik",
            "ticker",
            "company",
            "accession",
            "form",
            "filing_date",
            "acceptance_datetime",
            "retrieval_timestamp_utc",
            "source_document_url",
            "sha256",
        ],
        "point_in_time_field": (
            "acceptance_datetime (EDGAR acceptanceDateTime) -- the public availability/"
            "acceptance timestamp is used as the information timestamp, never "
            "filing_date/reportDate alone."
        ),
        "limitations": [
            "Static present-day-selected 12-issuer universe: survivorship/selection bias.",
            "Primary document text truncated at 200k chars.",
            "No sentiment/text scoring performed at acquisition time -- raw text and "
            "metadata only; features are computed downstream from these frozen inputs.",
        ],
        "notes": (
            "Acquired via a network-capable handoff environment (this session's own "
            "egress proxy blocks data.sec.gov/www.sec.gov). Raw filing text under "
            "data/raw/ is gitignored, never committed to this public repository."
        ),
    }


def build_prices_manifest(
    *,
    symbols: list[str],
    yf_version: str,
    retrieval_ts: str,
    freeze_ts: str | None,
    status: str,
    price_stats: dict[str, dict[str, Any]],
    parameters: dict[str, Any],
    sha256: dict[str, str] | None,
) -> dict[str, Any]:
    return {
        "dataset_id": PRICES_DATASET_ID,
        "source": "yfinance",
        "source_version": {"yfinance": yf_version},
        "universe_description": (
            "Daily yfinance OHLCV for the D9-D 12-issuer universe plus SPY "
            "(market context for excess-return labeling)."
        ),
        "symbols": symbols,
        "interval": "1d",
        "requested_start": parameters["start"],
        "requested_end_exclusive": parameters["end"],
        "actual_start": {s: price_stats[s]["actual_start"] for s in symbols},
        "actual_end": {s: price_stats[s]["actual_end"] for s in symbols},
        "row_counts": {s: price_stats[s]["row_count"] for s in symbols},
        "missing_counts": {s: price_stats[s]["missing_ohlcv_cells"] for s in symbols},
        "retrieval_timestamp_utc": retrieval_ts,
        "freeze_timestamp_utc": freeze_ts,
        "status": status,
        "sha256": sha256,
        "parameters": parameters,
        "limitations": ["yfinance prices are not exchange-official tapes."],
        "notes": "Raw CSVs under data/raw/ are gitignored, never committed.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=REQUESTED_START)
    parser.add_argument("--end", default=REQUESTED_END_EXCLUSIVE)
    parser.add_argument("--no-freeze", action="store_true")
    parser.add_argument("--max-filings-per-symbol", type=int, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance required", file=sys.stderr)
        return 2

    gitignore = _repo_root() / ".gitignore"
    if gitignore.exists() and "data/raw/" not in gitignore.read_text(encoding="utf-8"):
        print("ERROR: data/raw/ must be gitignored before acquire", file=sys.stderr)
        return 2

    root = _repo_root()
    raw_dir = args.raw_dir or (root / "data" / "raw")
    filings_raw_dir = raw_dir / FILINGS_DATASET_ID
    prices_raw_dir = raw_dir / PRICES_DATASET_ID

    print(f"Acquiring {FILINGS_DATASET_ID} + {PRICES_DATASET_ID}")
    print(f"yfinance={yf.__version__} issuers={FILING_SYMBOLS} forms={FORM_TYPES}")
    print(f"Requested [{args.start}, {args.end})")

    # --- Filings ---
    filing_parameters = {
        "start": args.start,
        "end": args.end,
        "forms": list(FORM_TYPES),
        "sec_user_agent": SEC_USER_AGENT,
        "max_filings_per_symbol": args.max_filings_per_symbol,
        "text_max_chars": TEXT_MAX_CHARS,
    }
    retrieval_ts = _utc_now()
    filing_stats: dict[str, dict[str, Any]] = {}
    filing_file_paths: dict[str, Path] = {}

    for symbol in FILING_SYMBOLS:
        print(f"== {symbol} SEC EDGAR (10-K/10-Q/8-K) ==")
        text_dir = filings_raw_dir / "filings" / "text" / symbol
        index, payload = acquire_symbol_filings_raw(
            symbol,
            SYMBOL_CIK[symbol],
            start=args.start,
            end=args.end,
            text_dir=text_dir,
            max_filings=args.max_filings_per_symbol,
        )
        index_path = filings_raw_dir / "filings" / f"{symbol}_index.csv"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index.to_csv(index_path, index=False)
        filing_file_paths[f"filings/{symbol}_index.csv"] = index_path
        for text_file in sorted(text_dir.glob("*.txt")):
            filing_file_paths[f"filings/text/{symbol}/{text_file.name}"] = text_file
        filing_stats[symbol] = payload["stats"]
        print(
            f"  listed={payload['stats']['filings_listed']} "
            f"ok={payload['stats']['filings_ok']} failed={payload['stats']['filings_failed']}"
        )

    if args.no_freeze:
        filings_status = "VALIDATED"
        filings_freeze_ts = None
        filings_sha256 = None
    else:
        filings_status = "DATA FROZEN"
        filings_freeze_ts = _utc_now()
        file_hashes = {name: sha256_file(path) for name, path in sorted(filing_file_paths.items())}
        filings_sha256 = {**file_hashes, "dataset_canonical": canonical_dataset_hash(file_hashes)}

    filings_manifest = build_filings_manifest(
        symbols=FILING_SYMBOLS,
        retrieval_ts=retrieval_ts,
        freeze_ts=filings_freeze_ts,
        status=filings_status,
        filing_stats=filing_stats,
        parameters=filing_parameters,
        sha256=filings_sha256,
    )
    filings_manifest_path = root / "data" / "manifests" / f"{FILINGS_DATASET_ID}.json"
    filings_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    filings_manifest_path.write_text(
        json.dumps(filings_manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote manifest {filings_manifest_path} status={filings_status}")

    # --- Prices (12 issuers + SPY) ---
    price_parameters = {
        "start": args.start,
        "end": args.end,
        "interval": "1d",
        "auto_adjust": True,
        "actions": True,
    }
    price_stats: dict[str, dict[str, Any]] = {}
    price_file_paths: dict[str, Path] = {}
    for symbol in PRICE_SYMBOLS:
        print(f"== {symbol} prices ==")
        prices = download_prices(symbol, start=args.start, end=args.end)
        stats = validate_prices(symbol, prices)
        price_path = prices_raw_dir / f"{symbol}.csv"
        write_csv(price_path, prices)
        price_file_paths[f"{symbol}.csv"] = price_path
        price_stats[symbol] = stats
        print(
            f"  rows={stats['row_count']} {stats['actual_start']}..{stats['actual_end']} "
            f"missing={stats['missing_ohlcv_cells']}"
        )

    if args.no_freeze:
        prices_status = "VALIDATED"
        prices_freeze_ts = None
        prices_sha256 = None
    else:
        prices_status = "DATA FROZEN"
        prices_freeze_ts = _utc_now()
        file_hashes = {name: sha256_file(path) for name, path in sorted(price_file_paths.items())}
        prices_sha256 = {**file_hashes, "dataset_canonical": canonical_dataset_hash(file_hashes)}

    prices_manifest = build_prices_manifest(
        symbols=PRICE_SYMBOLS,
        yf_version=yf.__version__,
        retrieval_ts=retrieval_ts,
        freeze_ts=prices_freeze_ts,
        status=prices_status,
        price_stats=price_stats,
        parameters=price_parameters,
        sha256=prices_sha256,
    )
    prices_manifest_path = root / "data" / "manifests" / f"{PRICES_DATASET_ID}.json"
    prices_manifest_path.write_text(
        json.dumps(prices_manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote manifest {prices_manifest_path} status={prices_status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
