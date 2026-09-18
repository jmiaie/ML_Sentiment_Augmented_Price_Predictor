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

The verifiers themselves live in ``quant_sentiment.frozen_inputs``, which the
authoritative runner also calls before any empirical execution -- one
verification path, so the freeze and a run cannot drift apart.

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
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from quant_sentiment.frozen_inputs import (  # noqa: E402
    InputIntegrityError,
    verify_filings,
    verify_prices,
)

FILINGS_DATASET_ID = "sec_filings_12issuer_2015_2025_v1"
PRICES_DATASET_ID = "yf_sentiment_equities_daily_2015_2025_v1"
STATUS_VALIDATED = "VALIDATED"
STATUS_FROZEN = "DATA FROZEN"

# The freeze path and the run path share ONE refusal type and ONE set of
# verifiers (quant_sentiment.frozen_inputs); the freeze keeps its historical
# name for its own callers.
FreezeRefused = InputIntegrityError


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
