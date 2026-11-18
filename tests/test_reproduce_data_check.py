"""Offline tests for scripts/reproduce_data_check.py (no network, tiny synthetic CSVs)."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reproduce_data_check.py"

HEADER = "Date,Capital Gains,Close,Dividends,High,Low,Open,Stock Splits,Volume\n"
ROW1 = "2020-01-02,0.0,10.0,0.0,11.0,9.0,10.0,0.0,100\n"
ROW2 = "2020-01-03,0.0,10.5,0.2,11.0,9.0,10.0,0.0,120\n"
GOOD = HEADER + ROW1 + ROW2
DRIFTED = HEADER + ROW1.replace(",10.0,", ",9.9,", 1) + ROW2.replace(",10.5,", ",10.4,", 1)


@pytest.fixture(scope="module")
def chk() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reproduce_data_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(chk: ModuleType, text: str, tmp_path: Path) -> str:
    p = tmp_path / "_h.csv"
    p.write_bytes(text.encode())
    return str(chk.sha256_file(p))


def _repo(chk: ModuleType, tmp_path: Path, frozen: dict[str, str]) -> tuple[Path, Path]:
    """A fake repo with one frozen manifest whose hashes are those of ``frozen``."""
    root = tmp_path / "repo"
    manifests = root / "data" / "manifests"
    manifests.mkdir(parents=True)
    hashes = {f"{s}.csv": _sha(chk, text, tmp_path) for s, text in frozen.items()}
    manifest = {
        "dataset_id": "fake_v1",
        "source": "yfinance",
        "source_version": "1.7.0",
        "symbols": list(frozen),
        "interval": "1d",
        "requested_start": "2020-01-01",
        "requested_end_exclusive": "2020-02-01",
        "actual_start": {s: "2020-01-02" for s in frozen},
        "actual_end": {s: "2020-01-03" for s in frozen},
        "row_counts": {s: 2 for s in frozen},
        "missing_counts": {s: 0 for s in frozen},
        "actions": {
            s: {
                "dividends": {"present": True, "nonzero_count": 1},
                "stock_splits": {"present": True, "nonzero_count": 0},
                "capital_gains": {"present": True, "nonzero_count": 0},
            }
            for s in frozen
        },
        "retrieval_timestamp_utc": "2026-01-01T00:00:00Z",
        "sha256": {**hashes, "dataset_canonical": chk.canonical_dataset_hash(hashes)},
        "parameters": {"start": "2020-01-01", "end": "2020-02-01", "interval": "1d"},
    }
    path = manifests / "fake_v1.json"
    path.write_text(json.dumps(manifest, indent=2))
    return root, path


def _fetcher(files: dict[str, str]) -> Callable[[dict[str, Any], Path], dict[str, Any]]:
    def fetch(spec: dict[str, Any], out_dir: Path) -> dict[str, Any]:
        for sym, text in files.items():
            (out_dir / f"{sym}.csv").write_bytes(text.encode())
        return {"retrieval_timestamp_utc": "2026-11-17T00:00:00Z", "excluded": []}

    return fetch


CONFIGS = ({"manifest": "fake_v1.json", "kind": "yfinance_inprocess", "via": "fake"},)


def _run(
    chk: ModuleType,
    root: Path,
    fetch: Callable[[dict[str, Any], Path], dict[str, Any]],
    tmp_path: Path,
) -> tuple[int, dict[str, Any], str]:
    json_out = tmp_path / "out" / "report.json"
    json_out.parent.mkdir(exist_ok=True)
    md_out = tmp_path / "out" / "report.md"
    code = chk.main(
        ["--json-out", str(json_out), "--md-out", str(md_out)],
        fetch=fetch,
        repo_root=root,
        configs=CONFIGS,
    )
    return code, json.loads(json_out.read_text()), md_out.read_text()


def test_match(chk: ModuleType, tmp_path: Path) -> None:
    root, _ = _repo(chk, tmp_path, {"AAA": GOOD, "BBB": GOOD})
    code, report, md = _run(chk, root, _fetcher({"AAA": GOOD, "BBB": GOOD}), tmp_path)
    ds = report["datasets"][0]
    assert code == 0
    assert ds["classification"] == "MATCH"
    assert ds["files_matched"] == 2
    assert all(f["match"] is True for f in ds["files"])
    assert ds["dataset_canonical"]["match"] is True
    assert "| fake_v1 |" in md and "MATCH" in md


def test_drift_is_described_and_exit_zero(chk: ModuleType, tmp_path: Path) -> None:
    root, _ = _repo(chk, tmp_path, {"AAA": GOOD, "BBB": GOOD})
    code, report, md = _run(chk, root, _fetcher({"AAA": GOOD, "BBB": DRIFTED}), tmp_path)
    ds = report["datasets"][0]
    assert code == 0
    assert ds["classification"] == "DATA DRIFT (not a reproduction failure of the analysis)"
    assert [f["match"] for f in ds["files"]] == [True, False]
    assert ds["dataset_canonical"]["match"] is False
    assert ds["files"][1]["rows"]["match"] is True  # metadata unchanged, values drifted
    assert any("bytes differ in values only" in s for s in ds["drift_summary"])
    assert "DATA DRIFT" in md


def test_metadata_drift_is_reported(chk: ModuleType, tmp_path: Path) -> None:
    root, _ = _repo(chk, tmp_path, {"AAA": GOOD})
    longer = GOOD + "2020-01-06,0.0,10.6,0.0,11.0,9.0,10.0,0.0,90\n"
    _, report, _ = _run(chk, root, _fetcher({"AAA": longer}), tmp_path)
    f = report["datasets"][0]["files"][0]
    assert f["rows"] == {"expected": 2, "fresh": 3, "match": False}
    assert f["last_date"]["fresh"] == "2020-01-06"


def test_missing_values_counted(chk: ModuleType, tmp_path: Path) -> None:
    text = HEADER + ROW1.replace("0.0,10.0,", "0.0,,", 1) + ROW2
    p = tmp_path / "m.csv"
    p.write_text(text)
    s = chk.summarize_csv(p)
    assert s["missing_values"] == 1
    assert s["actions"]["dividends"]["nonzero_count"] == 1


def test_missing_fresh_file_is_a_drift_not_a_crash(chk: ModuleType, tmp_path: Path) -> None:
    root, _ = _repo(chk, tmp_path, {"AAA": GOOD, "BBB": GOOD})
    code, report, _ = _run(chk, root, _fetcher({"AAA": GOOD}), tmp_path)
    ds = report["datasets"][0]
    assert code == 0
    assert ds["files"][1]["fresh_file_present"] is False
    assert ds["files"][1]["match"] is False
    assert ds["classification"].startswith("DATA DRIFT")


def test_fetch_failure_exits_two(chk: ModuleType, tmp_path: Path) -> None:
    root, _ = _repo(chk, tmp_path, {"AAA": GOOD})

    def boom(spec: dict[str, Any], out_dir: Path) -> dict[str, Any]:
        raise chk.FetchError("network down")

    code, report, md = _run(chk, root, boom, tmp_path)
    assert code == 2
    assert report["datasets"][0]["classification"] == "FETCH FAILED"
    assert "network down" in report["datasets"][0]["error"]
    assert report["exit_code"] == 2 and "FETCH FAILED" in md


def test_refuses_to_write_report_into_manifest_dir(chk: ModuleType, tmp_path: Path) -> None:
    root, path = _repo(chk, tmp_path, {"AAA": GOOD})
    before = path.read_text()
    code = chk.main(
        ["--json-out", str(path)], fetch=_fetcher({"AAA": GOOD}), repo_root=root, configs=CONFIGS
    )
    assert code == 2
    assert path.read_text() == before


def test_frozen_manifest_untouched_after_run(chk: ModuleType, tmp_path: Path) -> None:
    root, path = _repo(chk, tmp_path, {"AAA": GOOD})
    before = path.read_bytes()
    _run(chk, root, _fetcher({"AAA": DRIFTED}), tmp_path)
    assert path.read_bytes() == before


def test_not_checked_sec_datasets_are_listed(chk: ModuleType, tmp_path: Path) -> None:
    root, _ = _repo(chk, tmp_path, {"AAA": GOOD})
    code, report, md = _run(chk, root, _fetcher({"AAA": GOOD}), tmp_path)
    assert code == 0
    listed = {r["dataset"]: r["reason"] for r in report["not_checked"]}
    assert set(listed) == {
        "edgar_8k_yf_megacap_daily_2015_2025_v1",
        "sec_filings_12issuer_2015_2025_v1",
    }
    assert "not checked (needs SEC_USER_AGENT secret)" in md


def _install_fake_acquisition(
    monkeypatch: pytest.MonkeyPatch, frames: dict[str, str], seen: list[str]
) -> None:
    fake = ModuleType("quant_sentiment.acquisition")

    def download_prices(symbol: str, *, start: str, end: str) -> str:
        seen.append(f"{symbol}:{start}:{end}")
        if symbol not in frames:
            raise ValueError(f"No price data for {symbol}")
        return frames[symbol]

    def validate_prices(symbol: str, df: str) -> dict[str, Any]:
        return {}

    def write_csv(path: Path, df: str) -> None:
        path.write_bytes(df.encode())

    fake.download_prices = download_prices  # type: ignore[attr-defined]
    fake.validate_prices = validate_prices  # type: ignore[attr-defined]
    fake.write_csv = write_csv  # type: ignore[attr-defined]
    pkg = ModuleType("quant_sentiment")
    monkeypatch.setitem(sys.modules, "quant_sentiment", pkg)
    monkeypatch.setitem(sys.modules, "quant_sentiment.acquisition", fake)


def test_fetch_prices_uses_manifest_parameters_and_leaves_manifests_alone(
    chk: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, path = _repo(chk, tmp_path, {"AAA": GOOD})
    before = path.read_bytes()
    seen: list[str] = []
    _install_fake_acquisition(monkeypatch, {"AAA": GOOD}, seen)
    out = tmp_path / "raw"
    out.mkdir()
    spec = {"symbols": ["AAA"], "start": "2020-01-01", "end": "2020-02-01"}
    meta = chk.fetch_prices(spec, out, root)
    assert seen == ["AAA:2020-01-01:2020-02-01"]
    assert (out / "AAA.csv").read_bytes() == GOOD.encode()
    assert meta["retrieval_timestamp_utc"].endswith("Z")
    assert path.read_bytes() == before


def test_fetch_prices_failure_and_inside_repo_are_fetch_errors(
    chk: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _repo(chk, tmp_path, {"AAA": GOOD})
    _install_fake_acquisition(monkeypatch, {"AAA": GOOD}, [])
    out = tmp_path / "raw"
    out.mkdir()
    spec = {"symbols": ["AAA", "ZZZ"], "start": "2020-01-01", "end": "2020-02-01"}
    with pytest.raises(chk.FetchError, match="ZZZ"):
        chk.fetch_prices(spec, out, root)
    with pytest.raises(chk.FetchError, match="inside the repository"):
        chk.fetch_prices(spec, root / "data" / "raw", root)
