"""Offline tests for the zero-network D9-D snapshot freeze.

Two guarantees are on trial here: the freeze verifies the EXISTING bytes, and
it makes no network call while doing so. The second is only meaningful if the
guard can actually fire, so the first test proves the socket blocker blocks --
otherwise every later test would pass vacuously.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import socket
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from quant_sentiment.acquisition import canonical_dataset_hash

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SCRIPT = ROOT / "scripts" / "freeze_d9d_snapshot.py"

ISSUERS = [f"Z{i:02d}" for i in range(12)]
MARKET = "SPY"
MARKET_SYMBOLS = [*ISSUERS, MARKET]
FILINGS_DS = "sec_filings_12issuer_2015_2025_v1"
PRICES_DS = "yf_sentiment_equities_daily_2015_2025_v1"
INDEX_FIELDS = [
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
    "text_chars",
    "text_relpath",
    "status",
]
TEXT_PREFIX = "filings/text/"


def _load_freeze() -> ModuleType:
    spec = importlib.util.spec_from_file_location("freeze_d9d_snapshot", FREEZE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def freezer() -> ModuleType:
    return _load_freeze()


def _assert_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("network access attempted during freeze")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)


def test_socket_guard_is_not_vacuous(monkeypatch: pytest.MonkeyPatch) -> None:
    """If this guard ever stops firing, every other test here is worthless."""
    _assert_no_network(monkeypatch)
    with pytest.raises(AssertionError):
        socket.create_connection(("example.invalid", 80))
    with pytest.raises(AssertionError):
        socket.getaddrinfo("example.invalid", 80)


def _write_price_csv(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Date": pd.bdate_range("2015-01-01", periods=rows).strftime("%Y-%m-%d"),
            "Close": [100.0 + i for i in range(rows)],
            "Volume": [1_000_000 + i for i in range(rows)],
        }
    ).to_csv(path, index=False)


def _build_tree(
    tmp_path: Path,
    *,
    filings_per_issuer: int = 2,
    price_rows: int = 5,
    truncation_count: int = 0,
    row_count_delta: int = 0,
) -> dict[str, Any]:
    raw = tmp_path / "data" / "raw"
    manifests = tmp_path / "data" / "manifests"
    filings_ds = raw / FILINGS_DS
    prices_ds = raw / PRICES_DS
    (filings_ds / "filings").mkdir(parents=True, exist_ok=True)
    prices_ds.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    file_hashes: dict[str, str] = {}
    for offset, symbol in enumerate(ISSUERS):
        rows: list[dict[str, Any]] = []
        for k in range(filings_per_issuer):
            accession = f"{offset:010d}-25-{k:06d}"
            rel = f"{TEXT_PREFIX}{symbol}/{accession}.txt"
            text = f"{symbol} periodic filing {k}. " + " ".join(["risk"] * 20)
            target = filings_ds / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            file_hashes[rel] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            rows.append(
                {
                    "cik": f"000000{offset:04d}",
                    "ticker": symbol,
                    "company": f"{symbol} Corp",
                    "accession": accession,
                    "form": ("10-K", "10-Q", "8-K")[k % 3],
                    "filing_date": f"20{15 + k:02d}-01-05",
                    "acceptance_datetime": f"20{15 + k:02d}-01-05T16:00:00Z",
                    "retrieval_timestamp_utc": "2026-09-17T00:00:00Z",
                    "source_document_url": f"https://example.invalid/{accession}.htm",
                    "sha256": file_hashes[rel],
                    "text_chars": len(text),
                    "text_relpath": rel,
                    "status": "OK",
                }
            )
        index_rel = f"filings/{symbol}_index.csv"
        index_path = filings_ds / index_rel
        with index_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        file_hashes[index_rel] = hashlib.sha256(index_path.read_bytes()).hexdigest()

    n_text = sum(1 for key in file_hashes if key.startswith(TEXT_PREFIX))
    n_rows = sum(filings_per_issuer for _ in ISSUERS)
    filings_path = manifests / f"{FILINGS_DS}.json"
    filings_path.write_text(
        json.dumps(
            {
                "dataset_id": FILINGS_DS,
                "symbols": ISSUERS,
                "status": "VALIDATED",
                "freeze_timestamp_utc": None,
                "sha256": {**file_hashes, "dataset_canonical": canonical_dataset_hash(file_hashes)},
                "parameters": {
                    "text_n_files": n_rows,
                    "truncation_count": truncation_count,
                    "extraction_policy": "uncapped",
                    "text_max_chars": None,
                    "text_files_at_legacy_cap_200000": 0,
                },
            },
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )

    price_row_counts: dict[str, int] = {}
    for symbol in MARKET_SYMBOLS:
        _write_price_csv(prices_ds / f"{symbol}.csv", price_rows)
        price_row_counts[symbol] = price_rows + (row_count_delta if symbol == ISSUERS[0] else 0)
    prices_path = manifests / f"{PRICES_DS}.json"
    prices_path.write_text(
        json.dumps(
            {
                "dataset_id": PRICES_DS,
                "symbols": MARKET_SYMBOLS,
                "status": "VALIDATED",
                "freeze_timestamp_utc": None,
                "sha256": None,
                "row_counts": price_row_counts,
            },
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "root": tmp_path,
        "raw": raw,
        "filings_manifest": filings_path,
        "prices_manifest": prices_path,
        "filings_canonical": canonical_dataset_hash(file_hashes),
        "n_text": n_text,
        "n_rows": n_rows,
    }


def _manifest(tree: dict[str, Any], key: str) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(Path(tree[key]).read_text(encoding="utf-8"))
    return manifest


def _hash_tree(raw: Path) -> dict[str, str]:
    return {
        str(path.relative_to(raw)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(raw.rglob("*"))
        if path.is_file()
    }


def test_freeze_flips_both_manifests_and_makes_no_network_call(
    freezer: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = _build_tree(tmp_path)
    _assert_no_network(monkeypatch)
    before = _hash_tree(tree["raw"])

    result = freezer.freeze(tmp_path, now="2026-09-17T12:00:00Z")

    filings = _manifest(tree, "filings_manifest")
    prices = _manifest(tree, "prices_manifest")
    assert filings["status"] == "DATA FROZEN"
    assert prices["status"] == "DATA FROZEN"
    assert filings["freeze_timestamp_utc"] == "2026-09-17T12:00:00Z"
    assert prices["freeze_timestamp_utc"] == "2026-09-17T12:00:00Z"

    # The filings per-file hashes and canonical hash are verified, never rewritten.
    recorded = {k: v for k, v in filings["sha256"].items() if k != "dataset_canonical"}
    assert len(recorded) == tree["n_text"] + len(ISSUERS)
    assert filings["sha256"]["dataset_canonical"] == tree["filings_canonical"]
    assert result["filings"]["canonical"] == tree["filings_canonical"]
    assert result["filings"]["index_rows_ok"] == tree["n_rows"]

    # The prices manifest had no hashes at all; freeze computes them.
    assert set(prices["sha256"]) == {f"{s}.csv" for s in MARKET_SYMBOLS} | {"dataset_canonical"}
    assert len(prices["sha256"]["dataset_canonical"]) == 64
    assert prices["sha256"]["dataset_canonical"] == result["prices"]["canonical"]

    # Every raw byte is untouched -- freezing never rewrites data.
    assert _hash_tree(tree["raw"]) == before


def test_dry_run_verifies_without_writing(
    freezer: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = _build_tree(tmp_path)
    _assert_no_network(monkeypatch)
    before = _hash_tree(tree["raw"])

    result = freezer.freeze(tmp_path, now="2026-09-17T12:00:00Z", dry_run=True)

    assert result["dry_run"] is True
    assert len(result["prices"]["canonical"]) == 64
    assert _manifest(tree, "filings_manifest")["status"] == "VALIDATED"
    assert _manifest(tree, "prices_manifest")["status"] == "VALIDATED"
    assert _manifest(tree, "prices_manifest")["sha256"] is None
    assert _manifest(tree, "filings_manifest")["freeze_timestamp_utc"] is None
    assert _hash_tree(tree["raw"]) == before


def test_refuses_when_already_frozen(freezer: ModuleType, tmp_path: Path) -> None:
    tree = _build_tree(tmp_path)
    freezer.freeze(tmp_path, now="2026-09-17T12:00:00Z")
    first = _manifest(tree, "filings_manifest")["freeze_timestamp_utc"]
    with pytest.raises(freezer.FreezeRefused):
        freezer.freeze(tmp_path, now="2026-09-18T09:00:00Z")
    assert _manifest(tree, "filings_manifest")["freeze_timestamp_utc"] == first


def test_refuses_when_truncation_count_is_nonzero(freezer: ModuleType, tmp_path: Path) -> None:
    tree = _build_tree(tmp_path, truncation_count=7)
    with pytest.raises(freezer.FreezeRefused, match="truncation_count"):
        freezer.freeze(tmp_path)
    assert _manifest(tree, "filings_manifest")["status"] == "VALIDATED"


def test_refuses_when_a_recorded_file_no_longer_matches(
    freezer: ModuleType, tmp_path: Path
) -> None:
    tree = _build_tree(tmp_path)
    tampered = tree["raw"] / FILINGS_DS / TEXT_PREFIX / ISSUERS[0] / f"{0:010d}-25-000000.txt"
    tampered.write_text("rewritten after acquisition", encoding="utf-8")

    with pytest.raises(freezer.FreezeRefused, match="sha256 mismatch"):
        freezer.freeze(tmp_path)

    filings = _manifest(tree, "filings_manifest")
    prices = _manifest(tree, "prices_manifest")
    assert filings["status"] == "VALIDATED"
    assert prices["status"] == "VALIDATED"
    assert prices["sha256"] is None


def test_refuses_when_an_index_row_is_not_ok(freezer: ModuleType, tmp_path: Path) -> None:
    tree = _build_tree(tmp_path)
    index_path = tree["raw"] / FILINGS_DS / "filings" / f"{ISSUERS[0]}_index.csv"
    text = index_path.read_text(encoding="utf-8").replace(",OK\n", ",FAILED\n", 1)
    index_path.write_text(text, encoding="utf-8")

    with pytest.raises(freezer.FreezeRefused, match="not status OK"):
        freezer.freeze(tmp_path)


def test_refuses_when_a_price_csv_is_missing(freezer: ModuleType, tmp_path: Path) -> None:
    tree = _build_tree(tmp_path)
    (tree["raw"] / PRICES_DS / f"{MARKET}.csv").unlink()
    with pytest.raises(freezer.FreezeRefused, match="missing price CSV"):
        freezer.freeze(tmp_path)
    assert _manifest(tree, "prices_manifest")["status"] == "VALIDATED"


def test_refuses_when_a_price_row_count_disagrees(freezer: ModuleType, tmp_path: Path) -> None:
    _build_tree(tmp_path, row_count_delta=3)
    with pytest.raises(freezer.FreezeRefused, match="data rows"):
        freezer.freeze(tmp_path)


def test_refuses_when_the_issuer_count_is_wrong(freezer: ModuleType, tmp_path: Path) -> None:
    tree = _build_tree(tmp_path)
    manifest = _manifest(tree, "filings_manifest")
    manifest["symbols"] = manifest["symbols"][:11]
    Path(tree["filings_manifest"]).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(freezer.FreezeRefused, match="issuers"):
        freezer.freeze(tmp_path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("extraction_policy", "capped_at_200000"),
        ("text_max_chars", 200_000),
        ("text_files_at_legacy_cap_200000", 3),
    ],
)
def test_refuses_a_corpus_that_is_not_uncapped(
    freezer: ModuleType, tmp_path: Path, key: str, value: Any
) -> None:
    """The freeze's whole claim is that the frozen text is the UNtruncated text,
    so a manifest that no longer declares uncapped extraction must not freeze."""
    tree = _build_tree(tmp_path)
    manifest = _manifest(tree, "filings_manifest")
    manifest["parameters"][key] = value
    Path(tree["filings_manifest"]).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(freezer.FreezeRefused, match="uncapped"):
        freezer.freeze(tmp_path)
    assert _manifest(tree, "filings_manifest")["status"] == "VALIDATED"
