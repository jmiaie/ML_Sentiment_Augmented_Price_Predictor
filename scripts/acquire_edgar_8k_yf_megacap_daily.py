#!/usr/bin/env python3
"""Acquire and freeze edgar_8k_yf_megacap_daily_2015_2025_v1 for Directive #9.

Reproducible *subset* (not full EDGAR corpus):
- Prices: yfinance daily OHLCV for AAPL / MSFT / AMZN
- Text: SEC EDGAR 8-K primary documents for the same CIKs
- Sentiment: deterministic lexicon polarity on extracted filing text
- Point-in-time: availability = EDGAR acceptanceDateTime

Raw under data/raw/ (gitignored). Manifest under data/manifests/ committed.
Network acquisition is local/agent only — never from CI.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DATASET_ID = "edgar_8k_yf_megacap_daily_2015_2025_v1"
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "AMZN"]
# SEC CIK mapping for the frozen megacap subset (zero-padded 10 digits in API paths).
SYMBOL_CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "AMZN": "0001018724",
}
REQUESTED_START = "2015-01-01"
REQUESTED_END_EXCLUSIVE = "2026-01-01"
FORM_TYPES = ("8-K",)

# Import lexicon/scorer from package when available; script also works after editable install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quant_sentiment.edgar_filings import (  # noqa: E402
    fetch_company_filings,
    filing_archive_url,
    filter_filings,
)
from quant_sentiment.edgar_text import html_to_plain_text  # noqa: E402
from quant_sentiment.hashing import canonical_dataset_hash, sha256_file  # noqa: E402
from quant_sentiment.lexicon import LEXICON_ID, LEXICON_VERSION, score_text  # noqa: E402
from quant_sentiment.market_data_io import (  # noqa: E402
    download_prices,
    validate_prices,
    write_csv,
)
from quant_sentiment.sec_http import get_sec_user_agent  # noqa: E402
from quant_sentiment.sec_http import sec_get_text as _sec_get_text  # noqa: E402


def _repo_root() -> Path:
    return ROOT


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def acquire_symbol_filings(
    symbol: str,
    cik: str,
    *,
    start: str,
    end: str,
    text_dir: Path,
    max_filings: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    index_records: list[dict[str, Any]] = []
    failures = 0

    for row in rows:
        accession = str(row["accessionNumber"])
        primary = str(row["primaryDocument"])
        url = filing_archive_url(cik, accession, primary)
        safe_name = accession.replace("-", "") + ".txt"
        text_path = text_dir / safe_name
        try:
            if text_path.exists():
                plain = text_path.read_text(encoding="utf-8")
            else:
                html = _sec_get_text(url)
                plain = html_to_plain_text(html)
                text_path.write_text(plain, encoding="utf-8")
            scored = score_text(plain)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            failures += 1
            index_records.append(
                {
                    "symbol": symbol,
                    "cik": cik,
                    "form": row["form"],
                    "filingDate": row["filingDate"],
                    "acceptanceDateTime": row["acceptanceDateTime"],
                    "accessionNumber": accession,
                    "primaryDocument": primary,
                    "url": url,
                    "status": f"FAILED:{type(exc).__name__}",
                    "text_chars": 0,
                }
            )
            continue

        index_records.append(
            {
                "symbol": symbol,
                "cik": cik,
                "form": row["form"],
                "filingDate": row["filingDate"],
                "acceptanceDateTime": row["acceptanceDateTime"],
                "accessionNumber": accession,
                "primaryDocument": primary,
                "url": url,
                "status": "OK",
                "text_chars": len(plain),
                "positive_hits": scored.positive_hits,
                "negative_hits": scored.negative_hits,
                "token_count": scored.token_count,
                "text_relpath": f"filings/text/{symbol}/{safe_name}",
            }
        )
        event_records.append(
            {
                "source": "sec_edgar_8k",
                "event_id": accession,
                "entity": symbol,
                "symbol": symbol,
                "publication_timestamp": row["acceptanceDateTime"],
                "availability_timestamp": row["acceptanceDateTime"],
                "sentiment_score": scored.sentiment_score,
                "confidence": scored.confidence,
                "model_version": f"{LEXICON_ID}:{LEXICON_VERSION}",
                "positive_hits": scored.positive_hits,
                "negative_hits": scored.negative_hits,
                "token_count": scored.token_count,
                "filingDate": row["filingDate"],
                "form": row["form"],
            }
        )

    events = pd.DataFrame.from_records(event_records)
    index = pd.DataFrame.from_records(index_records)
    stats = {
        "filings_listed": len(rows),
        "filings_ok": int((index["status"] == "OK").sum()) if not index.empty else 0,
        "filings_failed": failures,
        "events": int(len(events)),
        "first_filing": index["filingDate"].min() if not index.empty else None,
        "last_filing": index["filingDate"].max() if not index.empty else None,
    }
    return events, {"index": index, "stats": stats}


def build_manifest(
    *,
    dataset_id: str,
    symbols: list[str],
    yf_version: str,
    retrieval_ts: str,
    freeze_ts: str | None,
    status: str,
    price_stats: dict[str, dict[str, Any]],
    filing_stats: dict[str, dict[str, Any]],
    parameters: dict[str, Any],
    sha256: dict[str, str] | None,
    notes: str,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "source": "sec_edgar+yfinance",
        "source_version": {
            "yfinance": yf_version,
            "sec_data_api": "data.sec.gov/submissions",
            "lexicon_id": LEXICON_ID,
            "lexicon_version": LEXICON_VERSION,
        },
        "universe_description": (
            "Reproducible megacap subset: AAPL/MSFT/AMZN daily yfinance prices + "
            "SEC EDGAR 8-K primary-document text scored with embedded financial "
            "polarity lexicon v1. Not the full EDGAR corpus; not LM-complete."
        ),
        "symbols": symbols,
        "ciks": {s: SYMBOL_CIK[s] for s in symbols},
        "form_types": list(FORM_TYPES),
        "interval": "1d",
        "requested_start": parameters["start"],
        "requested_end_exclusive": parameters["end"],
        "actual_start": {s: price_stats[s]["actual_start"] for s in symbols},
        "actual_end": {s: price_stats[s]["actual_end"] for s in symbols},
        "row_counts": {s: price_stats[s]["row_count"] for s in symbols},
        "missing_counts": {s: price_stats[s]["missing_ohlcv_cells"] for s in symbols},
        "filing_stats": {s: filing_stats[s] for s in symbols},
        "retrieval_timestamp_utc": retrieval_ts,
        "freeze_timestamp_utc": freeze_ts,
        "status": status,
        "sha256": sha256,
        "parameters": parameters,
        "limitations": [
            "8-K subset only (not 10-K/10-Q full corpus)",
            "Three megacap issuers only",
            "Simplified embedded polarity lexicon (not full Loughran-McDonald)",
            "Primary document text truncated at 200k chars",
            "yfinance prices are not exchange-official tapes",
        ],
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--start", default=REQUESTED_START)
    parser.add_argument("--end", default=REQUESTED_END_EXCLUSIVE)
    parser.add_argument("--no-freeze", action="store_true")
    parser.add_argument("--max-filings-per-symbol", type=int, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    for symbol in args.symbols:
        if symbol not in SYMBOL_CIK:
            print(f"ERROR: unknown CIK mapping for {symbol}", file=sys.stderr)
            return 2

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
    raw_dir = args.raw_dir or (root / "data" / "raw" / args.dataset_id)
    manifest_path = root / "data" / "manifests" / f"{args.dataset_id}.json"
    raw_dir.mkdir(parents=True, exist_ok=True)

    parameters = {
        "start": args.start,
        "end": args.end,
        "interval": "1d",
        "auto_adjust": True,
        "actions": True,
        "forms": list(FORM_TYPES),
        "sec_user_agent": get_sec_user_agent(),
        "lexicon_id": LEXICON_ID,
        "lexicon_version": LEXICON_VERSION,
        "max_filings_per_symbol": args.max_filings_per_symbol,
        "text_max_chars": 200_000,
    }

    retrieval_ts = _utc_now()
    price_stats: dict[str, dict[str, Any]] = {}
    filing_stats: dict[str, dict[str, Any]] = {}
    file_paths: dict[str, Path] = {}

    print(f"Acquiring {args.dataset_id}")
    print(f"yfinance={yf.__version__} symbols={args.symbols} forms={FORM_TYPES}")
    print(f"Requested [{args.start}, {args.end})")

    for symbol in args.symbols:
        print(f"== {symbol} prices ==")
        prices = download_prices(symbol, start=args.start, end=args.end)
        stats = validate_prices(symbol, prices)
        price_path = raw_dir / "prices" / f"{symbol}.csv"
        write_csv(price_path, prices)
        file_paths[f"prices/{symbol}.csv"] = price_path
        price_stats[symbol] = stats
        print(
            f"  prices rows={stats['row_count']} "
            f"{stats['actual_start']}..{stats['actual_end']} "
            f"missing={stats['missing_ohlcv_cells']}"
        )

        print(f"== {symbol} EDGAR 8-K ==")
        text_dir = raw_dir / "filings" / "text" / symbol
        events, filing_payload = acquire_symbol_filings(
            symbol,
            SYMBOL_CIK[symbol],
            start=args.start,
            end=args.end,
            text_dir=text_dir,
            max_filings=args.max_filings_per_symbol,
        )
        index = filing_payload["index"]
        index_path = raw_dir / "filings" / f"{symbol}_8k_index.csv"
        events_path = raw_dir / "events" / f"{symbol}_sentiment_events.csv"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.parent.mkdir(parents=True, exist_ok=True)
        index.to_csv(index_path, index=False)
        events.to_csv(events_path, index=False)
        file_paths[f"filings/{symbol}_8k_index.csv"] = index_path
        file_paths[f"events/{symbol}_sentiment_events.csv"] = events_path
        # Hash text files too (content-addressed freeze).
        for text_file in sorted(text_dir.glob("*.txt")):
            rel = f"filings/text/{symbol}/{text_file.name}"
            file_paths[rel] = text_file
        filing_stats[symbol] = filing_payload["stats"]
        print(
            f"  filings listed={filing_payload['stats']['filings_listed']} "
            f"ok={filing_payload['stats']['filings_ok']} "
            f"failed={filing_payload['stats']['filings_failed']}"
        )

    if args.no_freeze:
        status = "VALIDATED"
        freeze_ts = None
        sha256: dict[str, str] | None = None
        notes = "Acquired/validated; NOT frozen (sha256 null)."
    else:
        status = "DATA FROZEN"
        freeze_ts = _utc_now()
        file_hashes = {name: sha256_file(path) for name, path in sorted(file_paths.items())}
        sha256 = {
            **file_hashes,
            "dataset_canonical": canonical_dataset_hash(file_hashes),
        }
        notes = (
            "Frozen after local SEC EDGAR + yfinance acquisition. Raw files remain "
            "gitignored under data/raw/. Re-acquire with same parameters and compare "
            "dataset_canonical if regenerating. Subset limitations recorded in manifest."
        )

    manifest = build_manifest(
        dataset_id=args.dataset_id,
        symbols=list(args.symbols),
        yf_version=yf.__version__,
        retrieval_ts=retrieval_ts,
        freeze_ts=freeze_ts,
        status=status,
        price_stats=price_stats,
        filing_stats=filing_stats,
        parameters=parameters,
        sha256=sha256,
        notes=notes,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote manifest {manifest_path}")
    print(f"status={status}")
    if sha256 is not None:
        print(f"dataset_canonical sha256={sha256['dataset_canonical']}")
    else:
        print("sha256=null (not frozen)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
