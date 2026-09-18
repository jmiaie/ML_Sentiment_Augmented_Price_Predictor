"""Zero-network, fail-closed verification of the frozen D9-D input bytes.

The freeze records per-file and canonical SHA-256 hashes in the two manifests.
Because ``data/raw`` is deliberately gitignored, those recorded values prove
nothing on their own: a raw file could change after the freeze and an empirical
artifact would still attribute itself to the old frozen canonical hash. This
module re-derives every hash from the bytes that are actually about to be
consumed, and refuses unless they are byte-identical to the frozen record.

ONE verification path, shared source -- not a second copy of the freeze
algorithm. ``scripts/freeze_d9d_snapshot.py`` imports these same functions for
its pre-freeze check, and the runner calls :func:`verify_frozen_inputs` BEFORE
it builds the event frame, fits a model, writes an artifact or appends a ledger
row. DEV, 2024, ``--select-final-c`` and the 2025 historical evaluation all pass
through that single call: there are no per-period integrity semantics.

No HTTP client is imported here (no requests / urllib / yfinance).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .acquisition import canonical_dataset_hash, sha256_file

FROZEN_MANIFEST_STATUS = "DATA FROZEN"
EXPECTED_ISSUERS = 12


class InputIntegrityError(RuntimeError):
    """The bytes on disk are not the frozen bytes. Refuse and write nothing."""


def _count_data_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def verify_filings(
    manifest: dict[str, Any],
    dataset_dir: Path,
    *,
    expected_issuers: int = EXPECTED_ISSUERS,
    expected_rows: int | None = None,
    expected_files: int | None = None,
    expected_text_files: int | None = None,
    expected_canonical: str | None = None,
) -> dict[str, Any]:
    """Verify the filing bytes on disk. Returns the measurements to report.

    Every expectation defaults to what the manifest itself claims, so the
    pre-freeze caller passes nothing extra; the run caller passes the frozen
    record, which turns "the manifest agrees with itself" into "the manifest
    agrees with the record measured at freeze time".
    """
    symbols = [str(s) for s in manifest.get("symbols") or []]
    if len(symbols) != expected_issuers:
        raise InputIntegrityError(
            f"manifest lists {len(symbols)} issuers, expected {expected_issuers}"
        )

    parameters = manifest.get("parameters") or {}
    expected_rows_manifest = int(parameters["text_n_files"])
    if expected_rows is not None and expected_rows_manifest != expected_rows:
        raise InputIntegrityError(
            f"manifest parameters.text_n_files = {expected_rows_manifest}, frozen record "
            f"expects {expected_rows}"
        )
    if int(parameters["truncation_count"]) != 0:
        raise InputIntegrityError(
            f"truncation_count is {parameters['truncation_count']}, expected 0 -- "
            "the corpus is not uncapped, do not freeze it"
        )
    if (
        parameters.get("extraction_policy") != "uncapped"
        or parameters.get("text_max_chars") is not None
    ):
        raise InputIntegrityError(
            f"extraction_policy is {parameters.get('extraction_policy')!r} with text_max_chars "
            f"{parameters.get('text_max_chars')!r}, expected 'uncapped' and null -- the corpus "
            "is not uncapped, do not freeze it"
        )
    if int(parameters.get("text_files_at_legacy_cap_200000", 0)) != 0:
        raise InputIntegrityError(
            f"text_files_at_legacy_cap_200000 is "
            f"{parameters['text_files_at_legacy_cap_200000']}, expected 0 -- the corpus is not "
            "uncapped, do not freeze it"
        )

    listings = dataset_dir / "filings"
    index_rows = 0
    referenced_text: set[str] = set()
    for symbol in symbols:
        index_path = listings / f"{symbol}_index.csv"
        if not index_path.is_file():
            raise InputIntegrityError(f"missing filing index CSV for {symbol}: {index_path}")
        with index_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise InputIntegrityError(f"{symbol}: index CSV has no rows")
        not_ok = [r for r in rows if r.get("status") != "OK"]
        if not_ok:
            raise InputIntegrityError(
                f"{symbol}: {len(not_ok)} index row(s) not status OK "
                f"(e.g. accession {not_ok[0].get('accession')!r} "
                f"status {not_ok[0].get('status')!r})"
            )
        for row in rows:
            rel = str(row.get("text_relpath") or "")
            if not rel:
                raise InputIntegrityError(
                    f"{symbol}: index row {row.get('accession')!r} has no text_relpath"
                )
            referenced_text.add(rel)
        index_rows += len(rows)

    if index_rows != expected_rows_manifest:
        raise InputIntegrityError(
            f"index rows on disk = {index_rows}, manifest parameters.text_n_files = "
            f"{expected_rows_manifest}"
        )

    recorded = {k: v for k, v in (manifest.get("sha256") or {}).items() if k != "dataset_canonical"}
    if not recorded:
        raise InputIntegrityError("filings manifest records no per-file sha256 entries")

    measured: dict[str, str] = {}
    for rel, expected in sorted(recorded.items()):
        target = dataset_dir / rel
        if not target.is_file():
            raise InputIntegrityError(f"recorded file missing on disk: {rel}")
        actual = sha256_file(target)
        if actual != expected:
            raise InputIntegrityError(
                f"sha256 mismatch for {rel}: on-disk {actual} != recorded {expected} "
                "(the snapshot has been modified)"
            )
        measured[rel] = actual

    hashed_text = {k for k in measured if k.startswith("filings/text/")}
    if referenced_text != hashed_text:
        only_rows = sorted(referenced_text - hashed_text)[:3]
        only_files = sorted(hashed_text - referenced_text)[:3]
        raise InputIntegrityError(
            "index-referenced text files do not match the hashed text files "
            f"(referenced-only {only_rows}, hashed-only {only_files})"
        )

    if expected_text_files is not None and len(hashed_text) != expected_text_files:
        raise InputIntegrityError(
            f"{len(hashed_text)} text files on disk, frozen record expects {expected_text_files}"
        )
    if expected_files is not None and len(measured) != expected_files:
        raise InputIntegrityError(
            f"{len(measured)} filing files verified, frozen record expects {expected_files}"
        )

    canonical = canonical_dataset_hash(measured)
    recorded_canonical = (manifest.get("sha256") or {}).get("dataset_canonical")
    if canonical != recorded_canonical:
        raise InputIntegrityError(
            f"recomputed canonical dataset hash {canonical} != recorded {recorded_canonical}"
        )
    if expected_canonical is not None and canonical != expected_canonical:
        raise InputIntegrityError(
            f"filing bytes recompute to canonical {canonical}, frozen record expects "
            f"{expected_canonical}"
        )

    return {
        "issuers": len(symbols),
        "index_rows_ok": index_rows,
        "files_verified": len(measured),
        "text_files": len(hashed_text),
        "canonical": canonical,
        "file_hashes": measured,
    }


def verify_prices(
    manifest: dict[str, Any],
    dataset_dir: Path,
    *,
    expected_csvs: int | None = None,
    expected_canonical: str | None = None,
) -> dict[str, Any]:
    """Verify the price bytes on disk. Returns the measurements to report.

    The pre-freeze caller's manifest records no hashes yet, so comparison is
    skipped and the computed hashes are handed back to be recorded. The run
    caller's manifest does record them, so every per-file hash is compared.
    """
    symbols = [str(s) for s in manifest.get("symbols") or []]
    if not symbols:
        raise InputIntegrityError("prices manifest lists no symbols")
    if expected_csvs is not None and len(symbols) != expected_csvs:
        raise InputIntegrityError(
            f"prices manifest lists {len(symbols)} symbols, frozen record expects {expected_csvs}"
        )
    row_counts = dict(manifest.get("row_counts") or {})
    recorded = {k: v for k, v in (manifest.get("sha256") or {}).items() if k != "dataset_canonical"}
    if recorded and set(recorded) != {f"{symbol}.csv" for symbol in symbols}:
        extra = sorted(set(recorded) - {f"{symbol}.csv" for symbol in symbols})[:3]
        raise InputIntegrityError(
            "recorded price files do not match the manifest symbols "
            f"(e.g. {extra} is recorded but not a listed symbol)"
        )

    file_hashes: dict[str, str] = {}
    for symbol in symbols:
        path = dataset_dir / f"{symbol}.csv"
        if not path.is_file():
            raise InputIntegrityError(f"missing price CSV for {symbol}: {path}")
        expected_rows = row_counts.get(symbol)
        if expected_rows is None:
            raise InputIntegrityError(f"prices manifest records no row_count for {symbol}")
        actual_rows = _count_data_rows(path)
        if actual_rows != int(expected_rows):
            raise InputIntegrityError(
                f"{symbol}.csv has {actual_rows} data rows, manifest records {expected_rows}"
            )
        actual = sha256_file(path)
        recorded_hash = recorded.get(f"{symbol}.csv")
        if recorded_hash is not None and recorded_hash != actual:
            raise InputIntegrityError(
                f"sha256 mismatch for {symbol}.csv: on-disk {actual} != recorded {recorded_hash} "
                "(the snapshot has been modified)"
            )
        file_hashes[f"{symbol}.csv"] = actual

    canonical = canonical_dataset_hash(file_hashes)
    recorded_canonical = (manifest.get("sha256") or {}).get("dataset_canonical")
    if recorded_canonical and canonical != recorded_canonical:
        raise InputIntegrityError(
            f"recomputed canonical dataset hash {canonical} != recorded {recorded_canonical}"
        )
    if expected_canonical is not None and canonical != expected_canonical:
        raise InputIntegrityError(
            f"price bytes recompute to canonical {canonical}, frozen record expects "
            f"{expected_canonical}"
        )

    return {
        "symbols": len(symbols),
        "csvs_hashed": len(file_hashes),
        "canonical": canonical,
        "file_hashes": file_hashes,
    }


def verify_frozen_inputs(
    raw_dir: Path,
    *,
    filings_manifest_path: Path,
    prices_manifest_path: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Re-hash the whole frozen snapshot from disk; raise unless byte-exact.

    ``record`` is the freeze-time record (canonical hashes, file/row/issuer
    counts, freeze timestamp). Every value in it is re-measured here, so the
    artifact carries measured evidence rather than values copied out of the
    manifests it is trying to prove.
    """
    for path in (filings_manifest_path, prices_manifest_path):
        if not path.is_file():
            raise InputIntegrityError(f"missing manifest: {path}")

    filings: dict[str, Any] = json.loads(filings_manifest_path.read_text(encoding="utf-8"))
    prices: dict[str, Any] = json.loads(prices_manifest_path.read_text(encoding="utf-8"))

    for label, manifest in (("filings", filings), ("prices", prices)):
        current_status = str(manifest.get("status", ""))
        if current_status != FROZEN_MANIFEST_STATUS:
            raise InputIntegrityError(
                f"{label} manifest status {current_status!r} is not {FROZEN_MANIFEST_STATUS!r}"
            )
        if not (manifest.get("sha256") or {}).get("dataset_canonical"):
            raise InputIntegrityError(
                f"{label} manifest has no sha256.dataset_canonical; a frozen snapshot must "
                "carry its canonical dataset hash"
            )
        frozen_at = str(manifest.get("freeze_timestamp_utc") or "")
        if frozen_at != str(record["freeze_timestamp_utc"]):
            raise InputIntegrityError(
                f"{label} manifest freeze_timestamp_utc is {frozen_at!r}, frozen record expects "
                f"{record['freeze_timestamp_utc']!r} -- the snapshot was re-frozen or rewritten"
            )

    filings_dir = raw_dir / str(filings.get("dataset_id", ""))
    prices_dir = raw_dir / str(prices.get("dataset_id", ""))
    for label, dataset_dir in (("filings", filings_dir), ("prices", prices_dir)):
        if not dataset_dir.is_dir():
            raise InputIntegrityError(f"missing {label} dataset directory: {dataset_dir}")

    filings_result = verify_filings(
        filings,
        filings_dir,
        expected_issuers=int(record["issuers"]),
        expected_rows=int(record["filings_rows"]),
        expected_files=int(record["filings_files"]),
        expected_text_files=int(record["filings_text_files"]),
        expected_canonical=str(record["filings_canonical"]),
    )
    prices_result = verify_prices(
        prices,
        prices_dir,
        expected_csvs=int(record["price_csvs"]),
        expected_canonical=str(record["prices_canonical"]),
    )

    return {
        "verified": True,
        "verification_mode": "frozen_bytes_sha256",
        "frozen_record_freeze_timestamp_utc": str(record["freeze_timestamp_utc"]),
        "filings_issuers_verified": filings_result["issuers"],
        "filings_files_verified": filings_result["files_verified"],
        "filings_text_files_verified": filings_result["text_files"],
        "filings_index_rows_ok": filings_result["index_rows_ok"],
        "filings_canonical_recomputed": filings_result["canonical"],
        "prices_files_verified": prices_result["csvs_hashed"],
        "prices_canonical_recomputed": prices_result["canonical"],
        "filings_manifest_sha256": sha256_file(filings_manifest_path),
        "prices_manifest_sha256": sha256_file(prices_manifest_path),
    }
