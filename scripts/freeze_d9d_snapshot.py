#!/usr/bin/env python3
"""Freeze the EXISTING D9-D raw snapshot in place -- zero network access.

``scripts/acquire_sec_filings_12issuer_daily.py`` is the only other writer of
these two manifests, and it can only reach ``DATA FROZEN`` by re-downloading
the whole corpus first: it re-enters SEC acquisition and re-fetches every
Yahoo price series before flipping the status. Using that path to freeze would
make the freeze a side effect of a *fresh* acquisition rather than a guarantee
about the bytes already on disk -- and it would silently rewrite the
authoritative snapshot.

This is the separate, deliberately network-free freeze operation. It consumes
the EXISTING local raw trees only, verifies them against the per-file hashes
and counts the acquisition already recorded, and flips BOTH manifests from
VALIDATED -> DATA FROZEN while preserving the exact existing bytes.

No HTTP client is imported (no requests / urllib / yfinance).
``tests/test_freeze_d9d_snapshot.py`` proves it completes with sockets
disabled, and that a refusal leaves both manifests untouched.

Verified, fail closed (exit 2 on any mismatch):
  * 12 issuers, each with an index CSV and text files present on disk
  * every filing index row is ``status == OK`` and the total equals the
    manifest's own recorded ``parameters.text_n_files``
  * ``parameters.truncation_count == 0``
  * every per-file sha256 recorded in the filings manifest still matches the
    bytes on disk, and the recomputed canonical dataset hash matches
  * the text paths referenced by the index rows are exactly the hashed ones
  * all 13 price CSVs exist with their recorded row counts -- then computes
    and records the per-file + canonical hashes that manifest is missing

Usage:
    python scripts/freeze_d9d_snapshot.py --dry-run   # verify only, write nothing
    python scripts/freeze_d9d_snapshot.py             # verify, then freeze
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from quant_sentiment.acquisition import (  # noqa: E402
    canonical_dataset_hash,
    sha256_file,
)

FILINGS_DATASET_ID = "sec_filings_12issuer_2015_2025_v1"
PRICES_DATASET_ID = "yf_sentiment_equities_daily_2015_2025_v1"
STATUS_VALIDATED = "VALIDATED"
STATUS_FROZEN = "DATA FROZEN"
EXPECTED_ISSUERS = 12


class FreezeRefused(RuntimeError):
    """A verification failed. Both manifests are left exactly as they were."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load(path: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Serialise exactly as the acquisition script does -- verified to
    round-trip the committed manifests byte-for-byte, so the only diff this
    script produces is the fields it deliberately changes."""
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _count_data_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def verify_filings(manifest: dict[str, Any], dataset_dir: Path) -> dict[str, Any]:
    """Verify the existing filing bytes. Returns the measurements to report."""
    symbols = [str(s) for s in manifest.get("symbols") or []]
    if len(symbols) != EXPECTED_ISSUERS:
        raise FreezeRefused(f"manifest lists {len(symbols)} issuers, expected {EXPECTED_ISSUERS}")

    parameters = manifest.get("parameters") or {}
    expected_rows = int(parameters["text_n_files"])
    if int(parameters["truncation_count"]) != 0:
        raise FreezeRefused(
            f"truncation_count is {parameters['truncation_count']}, expected 0 -- "
            "the corpus is not uncapped, do not freeze it"
        )

    listings = dataset_dir / "filings"
    index_rows = 0
    referenced_text: set[str] = set()
    for symbol in symbols:
        index_path = listings / f"{symbol}_index.csv"
        if not index_path.is_file():
            raise FreezeRefused(f"missing filing index CSV for {symbol}: {index_path}")
        with index_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise FreezeRefused(f"{symbol}: index CSV has no rows")
        not_ok = [r for r in rows if r.get("status") != "OK"]
        if not_ok:
            raise FreezeRefused(
                f"{symbol}: {len(not_ok)} index row(s) not status OK "
                f"(e.g. accession {not_ok[0].get('accession')!r} "
                f"status {not_ok[0].get('status')!r})"
            )
        for row in rows:
            rel = str(row.get("text_relpath") or "")
            if not rel:
                raise FreezeRefused(
                    f"{symbol}: index row {row.get('accession')!r} has no text_relpath"
                )
            referenced_text.add(rel)
        index_rows += len(rows)

    if index_rows != expected_rows:
        raise FreezeRefused(
            f"index rows on disk = {index_rows}, manifest parameters.text_n_files = {expected_rows}"
        )

    recorded = {k: v for k, v in (manifest.get("sha256") or {}).items() if k != "dataset_canonical"}
    if not recorded:
        raise FreezeRefused("filings manifest records no per-file sha256 entries")

    measured: dict[str, str] = {}
    for rel, expected in sorted(recorded.items()):
        target = dataset_dir / rel
        if not target.is_file():
            raise FreezeRefused(f"recorded file missing on disk: {rel}")
        actual = sha256_file(target)
        if actual != expected:
            raise FreezeRefused(
                f"sha256 mismatch for {rel}: on-disk {actual} != recorded {expected} "
                "(the snapshot has been modified; do not freeze)"
            )
        measured[rel] = actual

    hashed_text = {k for k in measured if k.startswith("filings/text/")}
    if referenced_text != hashed_text:
        only_rows = sorted(referenced_text - hashed_text)[:3]
        only_files = sorted(hashed_text - referenced_text)[:3]
        raise FreezeRefused(
            "index-referenced text files do not match the hashed text files "
            f"(referenced-only {only_rows}, hashed-only {only_files})"
        )

    canonical = canonical_dataset_hash(measured)
    recorded_canonical = (manifest.get("sha256") or {}).get("dataset_canonical")
    if canonical != recorded_canonical:
        raise FreezeRefused(
            f"recomputed canonical dataset hash {canonical} != recorded {recorded_canonical}"
        )

    return {
        "issuers": len(symbols),
        "index_rows_ok": index_rows,
        "files_verified": len(measured),
        "text_files": len(hashed_text),
        "canonical": canonical,
        "file_hashes": measured,
    }


def verify_prices(manifest: dict[str, Any], dataset_dir: Path) -> dict[str, Any]:
    """Verify the existing price bytes and compute the hashes it is missing."""
    symbols = [str(s) for s in manifest.get("symbols") or []]
    if not symbols:
        raise FreezeRefused("prices manifest lists no symbols")
    row_counts = dict(manifest.get("row_counts") or {})

    file_hashes: dict[str, str] = {}
    for symbol in symbols:
        path = dataset_dir / f"{symbol}.csv"
        if not path.is_file():
            raise FreezeRefused(f"missing price CSV for {symbol}: {path}")
        expected_rows = row_counts.get(symbol)
        if expected_rows is None:
            raise FreezeRefused(f"prices manifest records no row_count for {symbol}")
        actual_rows = _count_data_rows(path)
        if actual_rows != int(expected_rows):
            raise FreezeRefused(
                f"{symbol}.csv has {actual_rows} data rows, manifest records {expected_rows}"
            )
        file_hashes[f"{symbol}.csv"] = sha256_file(path)

    return {
        "symbols": len(symbols),
        "csvs_hashed": len(file_hashes),
        "canonical": canonical_dataset_hash(file_hashes),
        "file_hashes": file_hashes,
    }


def freeze(root: Path, *, now: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Verify the existing snapshot, then flip both manifests to DATA FROZEN."""
    manifests_dir = root / "data" / "manifests"
    raw_dir = root / "data" / "raw"
    filings_path = manifests_dir / f"{FILINGS_DATASET_ID}.json"
    prices_path = manifests_dir / f"{PRICES_DATASET_ID}.json"
    for path in (filings_path, prices_path):
        if not path.is_file():
            raise FreezeRefused(f"missing manifest: {path}")

    filings = _load(filings_path)
    prices = _load(prices_path)

    for label, manifest in (("filings", filings), ("prices", prices)):
        status = str(manifest.get("status", ""))
        if status == STATUS_FROZEN:
            raise FreezeRefused(
                f"{label} manifest is already {STATUS_FROZEN}; refusing to re-freeze "
                "(it would move freeze_timestamp_utc and break the byte-preserving claim)"
            )
        if status != STATUS_VALIDATED:
            raise FreezeRefused(
                f"{label} manifest status is {status!r}, expected {STATUS_VALIDATED!r}"
            )

    filings_result = verify_filings(filings, raw_dir / FILINGS_DATASET_ID)
    prices_result = verify_prices(prices, raw_dir / PRICES_DATASET_ID)

    timestamp = now or _utc_now()
    if not dry_run:
        # Order matters: everything above is read-only, so a refusal cannot
        # leave one manifest frozen and the other not.
        filings["status"] = STATUS_FROZEN
        filings["freeze_timestamp_utc"] = timestamp
        prices["status"] = STATUS_FROZEN
        prices["freeze_timestamp_utc"] = timestamp
        prices["sha256"] = {
            **prices_result["file_hashes"],
            "dataset_canonical": prices_result["canonical"],
        }
        write_manifest(filings_path, filings)
        write_manifest(prices_path, prices)

    return {
        "dry_run": dry_run,
        "freeze_timestamp_utc": timestamp,
        "filings": filings_result,
        "prices": prices_result,
    }


def _report(result: dict[str, Any], root: Path) -> None:
    mode = "DRY RUN (nothing written)" if result["dry_run"] else "FROZEN"
    print(f"[freeze_d9d_snapshot] {mode}  root={root}")
    filings = result["filings"]
    prices = result["prices"]
    print(
        f"  filings: {filings['issuers']} issuers, {filings['index_rows_ok']} index rows OK, "
        f"{filings['files_verified']} files verified ({filings['text_files']} text)"
    )
    print(f"           canonical {filings['canonical']}")
    print(f"  prices:  {prices['csvs_hashed']} CSVs hashed, canonical {prices['canonical']}")
    print(f"  freeze_timestamp_utc {result['freeze_timestamp_utc']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true", help="verify only; write nothing")
    args = parser.parse_args(argv)

    try:
        result = freeze(args.root, dry_run=args.dry_run)
    except FreezeRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    _report(result, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
