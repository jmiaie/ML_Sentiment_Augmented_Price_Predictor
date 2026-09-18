"""Offline tests for the D9-D v2 acquisition script's freeze gate (no network).

Directly exercises the real, unmodified ``main()`` -- every network-touching
function (``fetch_company_filings``, ``_sec_get_text``, ``download_prices``)
is monkeypatched, and ``_repo_root`` is redirected to an isolated tmp
directory so the run never touches this repo's real ``data/`` tree. This
proves the freeze-refusal guard added for a verified defect (a
partially-failed acquisition was previously freezable and labeled plain
"DATA FROZEN" with no signal that some filings were missing) actually
works end to end, not just that its formula looks right in isolation.
"""

from __future__ import annotations

import importlib.util
import json
import urllib.error
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "acquire_sec_filings_12issuer_daily.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("acquire_sec_filings_12issuer_daily", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def acq() -> ModuleType:
    return _load_module()


def _synthetic_price_frame(n: int = 5) -> pd.DataFrame:
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


def _patch_network(
    monkeypatch: pytest.MonkeyPatch, acq: ModuleType, fake_root: Path, *, fail_first: bool
) -> None:
    """Two filings per symbol (one OK, and -- only for the first symbol
    queried when fail_first=True -- one that raises like a real fetch
    failure). Deterministic regardless of which CIK is passed. Also
    isolates the run to `fake_root` so it never touches this repo's real
    data/ tree."""
    call_state = {"symbol_index": 0}

    def fake_fetch_company_filings(cik: str) -> list[dict[str, Any]]:
        return [
            {
                "form": "8-K",
                "filingDate": "2016-03-01",
                "acceptanceDateTime": "2016-03-01T18:30:00-05:00",
                "accessionNumber": f"0000000000-16-{cik[-2:]}0001",
                "primaryDocument": "primary1.htm",
            },
            {
                "form": "8-K",
                "filingDate": "2016-04-01",
                "acceptanceDateTime": "2016-04-01T18:30:00-05:00",
                "accessionNumber": f"0000000000-16-{cik[-2:]}0002",
                "primaryDocument": "primary2.htm",
            },
        ]

    def fake_sec_get_text(url: str) -> str:
        symbol_position = call_state["symbol_index"]
        # The second filing of the very first symbol acquired is the failure.
        if fail_first and symbol_position == 0 and "0002" in url:
            raise urllib.error.URLError("synthetic network failure")
        return "<html><body><p>Synthetic filing text.</p></body></html>"

    def fake_download_prices(symbol: str, *, start: str, end: str) -> pd.DataFrame:
        return _synthetic_price_frame()

    real_acquire = acq.acquire_symbol_filings_raw

    def wrapped_acquire(*args: Any, **kwargs: Any) -> Any:
        result = real_acquire(*args, **kwargs)
        call_state["symbol_index"] += 1
        return result

    monkeypatch.setattr(acq, "fetch_company_filings", fake_fetch_company_filings)
    monkeypatch.setattr(acq, "_sec_get_text", fake_sec_get_text)
    monkeypatch.setattr(acq, "download_prices", fake_download_prices)
    monkeypatch.setattr(acq, "acquire_symbol_filings_raw", wrapped_acquire)
    monkeypatch.setattr(acq, "_repo_root", lambda: fake_root)
    monkeypatch.setenv("SEC_USER_AGENT", "test-suite contact test@example.invalid")


def test_freeze_refused_by_default_when_a_filing_fails(
    acq: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_network(monkeypatch, acq, tmp_path, fail_first=True)
    exit_code = acq.main([])
    assert exit_code == 2
    manifest_path = tmp_path / "data" / "manifests" / f"{acq.FILINGS_DATASET_ID}.json"
    # Refused before writing DATA FROZEN -- either no manifest at all, or
    # (if a future refactor moves the write earlier) never status "DATA FROZEN".
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["status"] != "DATA FROZEN"


def test_freeze_succeeds_with_no_failures(
    acq: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_network(monkeypatch, acq, tmp_path, fail_first=False)
    exit_code = acq.main([])
    assert exit_code == 0
    manifest_path = tmp_path / "data" / "manifests" / f"{acq.FILINGS_DATASET_ID}.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "DATA FROZEN"
    total_failed = sum(s["filings_failed"] for s in manifest["filing_stats"].values())
    assert total_failed == 0


def test_freeze_with_allow_partial_labels_status_explicitly(
    acq: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_network(monkeypatch, acq, tmp_path, fail_first=True)
    exit_code = acq.main(["--allow-partial"])
    assert exit_code == 0
    manifest_path = tmp_path / "data" / "manifests" / f"{acq.FILINGS_DATASET_ID}.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "DATA FROZEN (PARTIAL: 1 failed)"
    total_failed = sum(s["filings_failed"] for s in manifest["filing_stats"].values())
    assert total_failed == 1


def test_no_sec_user_agent_env_var_fails_clearly(
    acq: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Real defect found by Hai/Codex review: SEC_USER_AGENT was hard-coded
    with a personal contact string. It must now come from the environment
    and fail with a clear error (not a network error, not a silent
    fallback) when unset."""
    _patch_network(monkeypatch, acq, tmp_path, fail_first=False)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        acq.main([])


def test_manifest_records_acquisition_script_provenance(
    acq: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both manifests must record the acquisition script's own hash, so a
    result can be traced back to the exact code that produced it."""
    _patch_network(monkeypatch, acq, tmp_path, fail_first=False)
    exit_code = acq.main([])
    assert exit_code == 0

    filings_manifest_path = tmp_path / "data" / "manifests" / f"{acq.FILINGS_DATASET_ID}.json"
    filings_manifest = json.loads(filings_manifest_path.read_text())
    assert filings_manifest["acquisition_script_sha256"]

    prices_manifest_path = tmp_path / "data" / "manifests" / f"{acq.PRICES_DATASET_ID}.json"
    prices_manifest = json.loads(prices_manifest_path.read_text())
    assert prices_manifest["acquisition_script_sha256"]
