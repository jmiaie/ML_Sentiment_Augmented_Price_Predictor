"""Offline tests for the authoritative D9-D v2 runner (no network).

The runner is exercised end to end against a synthetic 2-issuer fixture corpus
built in tmp_path -- real NYSE session index, the real index-CSV schema, real
text files on disk, real manifests -- so the 2025 gates and the C-fixing path
are proven through the same code path the data takes, not by re-asserting the
gate formula in isolation.

Covered (change E):
  * DEV + 2024 validation write artifacts and a ledger, and never 2025
  * 2025 refused without --allow-2025
  * 2025 refused while config status is not frozen-final
  * 2025 refused while walk_forward.final_selected_c is unpopulated
  * 2025 refused while the corpus is not DATA FROZEN
  * a "DATA FROZEN (PARTIAL: ...)" corpus is refused for every run
  * with all gates satisfied, 2025 runs ONCE and uses the frozen C verbatim
    (c_selection == "fixed_from_frozen_config"), never re-selecting on 2025
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest
import yaml

from quant_sentiment.nyse_calendar import NyseCalendar

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_historical_text_study_v2.py"
REAL_CONFIG = ROOT / "configs" / "experiments" / "sentiment_historical_text_study_v2.yaml"

SYMBOLS = ["AAA", "BBB"]
MARKET = "SPY"
FILINGS_DS = "fx_filings_2015_2025_v1"
PRICES_DS = "fx_prices_2015_2025_v1"
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
NEGATIVE = "loss decline litigation uncertainty risk breach impairment weakness default".split()
POSITIVE = "growth improvement strong record profitable dividend expansion gain momentum".split()
FROZEN_C = {
    "model0_majority_baseline": None,
    "model1_market_only": 0.1,
    "model2_text_only": 1.0,
    "model3_market_text_combined": 10.0,
}


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_historical_text_study_v2", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    return _load_runner()


@pytest.fixture(scope="module")
def sessions() -> pd.DatetimeIndex:
    schedule = mcal.get_calendar("NYSE").schedule(start_date="2014-06-01", end_date="2025-12-31")
    return pd.DatetimeIndex(schedule.index).tz_localize(None)


@pytest.fixture(scope="module")
def calendar() -> NyseCalendar:
    return NyseCalendar()


def _write_prices(path: Path, sessions: pd.DatetimeIndex, seed: int) -> None:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.011, len(sessions))))
    pd.DataFrame(
        {
            "Date": sessions.strftime("%Y-%m-%d"),
            "Close": close,
            "Volume": rng.integers(500_000, 4_000_000, len(sessions)),
        }
    ).to_csv(path, index=False)


def _write_filings(
    filings_ds: Path,
    sessions: pd.DatetimeIndex,
    calendar: NyseCalendar,
    symbol: str,
    offset: int,
) -> int:
    rng = np.random.default_rng(offset + 7)
    rows: list[dict[str, Any]] = []
    usable = sessions[130:-6][offset::37]
    for k, ts in enumerate(usable):
        accession = f"{offset:010d}-25-{k:06d}"
        words = NEGATIVE if k % 3 else POSITIVE
        text = f"{symbol} periodic filing {k}. " + " ".join(
            str(w) for w in rng.choice(words, size=40 + 5 * (k % 7))
        )
        rel = f"filings/text/{symbol}/{accession}.txt"
        target = filings_ds / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        # Accepted one hour before that session's real close -> available as of
        # that same session's close, so the event lands on a session present in
        # the price frame above (no accidental holiday routing).
        acceptance = calendar.session_close(ts.date()) - pd.Timedelta(hours=1)
        rows.append(
            {
                "cik": f"000000{offset:04d}",
                "ticker": symbol,
                "company": f"{symbol} Corp",
                "accession": accession,
                "form": ("10-K", "10-Q", "8-K")[k % 3],
                "filing_date": str(ts.date()),
                "acceptance_datetime": acceptance.isoformat(),
                "retrieval_timestamp_utc": "2026-09-17T00:00:00Z",
                "source_document_url": f"https://example.invalid/{accession}.htm",
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text_chars": len(text),
                "text_relpath": rel,
                "status": "OK",
            }
        )
    index_path = filings_ds / "filings" / f"{symbol}_index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _build_fixture(
    tmp_path: Path,
    sessions: pd.DatetimeIndex,
    calendar: NyseCalendar,
    *,
    filings_status: str = "VALIDATED",
    config_status: str = "pre-registered",
    final_c: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    raw = tmp_path / "data" / "raw"
    filings_ds = raw / FILINGS_DS
    prices_ds = raw / PRICES_DS
    filings_ds.mkdir(parents=True, exist_ok=True)
    prices_ds.mkdir(parents=True, exist_ok=True)

    for seed, symbol in enumerate([*SYMBOLS, MARKET]):
        _write_prices(prices_ds / f"{symbol}.csv", sessions, seed)
    n_filings = {
        symbol: _write_filings(filings_ds, sessions, calendar, symbol, offset)
        for offset, symbol in enumerate(SYMBOLS)
    }

    manifests = tmp_path / "data" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / f"{FILINGS_DS}.json").write_text(
        json.dumps(
            {
                "dataset_id": FILINGS_DS,
                "status": filings_status,
                "sha256": {"dataset_canonical": "a" * 64},
                "freeze_timestamp_utc": None,
            }
        ),
        encoding="utf-8",
    )
    (manifests / f"{PRICES_DS}.json").write_text(
        json.dumps(
            {
                "dataset_id": PRICES_DS,
                "status": "VALIDATED",
                "sha256": {"dataset_canonical": "b" * 64},
                "freeze_timestamp_utc": None,
            }
        ),
        encoding="utf-8",
    )

    # Derived from the REAL config so the runner is tested against the schema it
    # actually consumes: a dropped/renamed key in the config fails these tests.
    config = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    config["experiment_id"] = "fx_text_study"
    config["status"] = config_status
    config["dataset_ids"] = {
        "filings": FILINGS_DS,
        "prices": PRICES_DS,
        "dictionary": config["dataset_ids"]["dictionary"],
    }
    config["universe"] = {**config["universe"], "symbols": SYMBOLS, "market_context_symbol": MARKET}
    config["walk_forward"] = {
        **config["walk_forward"],
        "initial_train_size": 6,
        "validation_size": 3,
        "formation_internal_test_size": 3,
        "step_size": 3,
        "final_selected_c": final_c,
    }
    config_path = tmp_path / "fx_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    return {
        "root": tmp_path,
        "raw": raw,
        "config": config_path,
        "results": tmp_path / "results",
        "ledger": tmp_path / "ledger.csv",
        "n_filings": n_filings,
    }


def _argv(fixture: dict[str, Any], *extra: str) -> list[str]:
    return [
        "--config",
        str(fixture["config"]),
        "--raw-dir",
        str(fixture["raw"]),
        "--results-dir",
        str(fixture["results"]),
        "--ledger",
        str(fixture["ledger"]),
        "--branch",
        "test-branch",
        *extra,
    ]


def _artifacts(fixture: dict[str, Any]) -> list[str]:
    results = fixture["results"]
    if not results.exists():
        return []
    return sorted(p.name for p in results.glob("*.json"))


def test_dev_and_validation_run_writes_artifacts_and_never_2025(
    runner: ModuleType, tmp_path: Path, sessions: pd.DatetimeIndex, calendar: NyseCalendar
) -> None:
    fixture = _build_fixture(tmp_path, sessions, calendar)
    assert runner.main(_argv(fixture)) == 0

    artifacts = _artifacts(fixture)
    assert artifacts == [
        "fx_text_study_primary_formation_dev.json",
        "fx_text_study_primary_validation.json",
        "fx_text_study_secondary_formation_dev.json",
        "fx_text_study_secondary_validation.json",
    ]

    payload = json.loads(
        (fixture["results"] / "fx_text_study_primary_validation.json").read_text()
    )
    key_metrics = payload["result"]["key_metrics"]
    assert key_metrics["n_eval_rows"] > 0
    assert key_metrics["n_formation_rows"] > 0
    assert key_metrics["c_selection"] == "tuned_on_formation"
    assert key_metrics["walk_forward_fold_count"] > 0
    assert key_metrics["embargo_sessions"] == 1
    assert "headline_delta_log_loss_model3_minus_model1" in key_metrics
    assert payload["config_status"] == "pre-registered"
    assert payload["corpus"]["filings_manifest_status"] == "VALIDATED"
    assert payload["corpus"]["n_frame_rows"] > 0
    assert payload["corpus"]["n_index_rows_not_ok"] == {}
    assert payload["corpus"]["dictionary_dataset_id_matches"] is True
    assert payload["corpus"]["counts_by_form"]
    assert payload["result"]["eval_period"]["name"] == "validation"
    assert len(payload["config_sha256"]) == 64
    assert "code_git_head" in payload

    secondary = json.loads(
        (fixture["results"] / "fx_text_study_secondary_validation.json").read_text()
    )
    assert secondary["result"]["key_metrics"]["embargo_sessions"] == 5

    ledger_rows = fixture["ledger"].read_text(encoding="utf-8").strip().splitlines()
    assert len(ledger_rows) == 5  # header + 4 runs
    assert "historical_evaluation" not in fixture["ledger"].read_text(encoding="utf-8")


def test_2025_requires_the_allow_flag(
    runner: ModuleType, tmp_path: Path, sessions: pd.DatetimeIndex, calendar: NyseCalendar
) -> None:
    fixture = _build_fixture(tmp_path, sessions, calendar)
    assert runner.main(_argv(fixture, "--periods", "historical_evaluation")) == 2
    assert _artifacts(fixture) == []


def test_2025_refused_while_config_is_not_frozen(
    runner: ModuleType, tmp_path: Path, sessions: pd.DatetimeIndex, calendar: NyseCalendar
) -> None:
    fixture = _build_fixture(tmp_path, sessions, calendar, config_status="pre-registered")
    argv = _argv(fixture, "--periods", "historical_evaluation", "--allow-2025")
    assert runner.main(argv) == 2
    assert _artifacts(fixture) == []


def test_2025_refused_without_final_selected_c(
    runner: ModuleType, tmp_path: Path, sessions: pd.DatetimeIndex, calendar: NyseCalendar
) -> None:
    fixture = _build_fixture(
        tmp_path, sessions, calendar, config_status="frozen-final", final_c=None
    )
    argv = _argv(fixture, "--periods", "historical_evaluation", "--allow-2025")
    assert runner.main(argv) == 2
    assert _artifacts(fixture) == []


def test_2025_refused_while_corpus_is_not_data_frozen(
    runner: ModuleType, tmp_path: Path, sessions: pd.DatetimeIndex, calendar: NyseCalendar
) -> None:
    fixture = _build_fixture(
        tmp_path,
        sessions,
        calendar,
        config_status="frozen-final",
        final_c=FROZEN_C,
        filings_status="VALIDATED",
    )
    argv = _argv(fixture, "--periods", "historical_evaluation", "--allow-2025")
    assert runner.main(argv) == 2
    assert _artifacts(fixture) == []


def test_partial_freeze_corpus_is_refused_for_ordinary_runs(
    runner: ModuleType, tmp_path: Path, sessions: pd.DatetimeIndex, calendar: NyseCalendar
) -> None:
    fixture = _build_fixture(
        tmp_path, sessions, calendar, filings_status="DATA FROZEN (PARTIAL: 3 failed)"
    )
    assert runner.main(_argv(fixture)) == 2
    assert _artifacts(fixture) == []


def test_2025_runs_once_with_all_gates_satisfied_and_uses_the_frozen_c(
    runner: ModuleType, tmp_path: Path, sessions: pd.DatetimeIndex, calendar: NyseCalendar
) -> None:
    fixture = _build_fixture(
        tmp_path,
        sessions,
        calendar,
        filings_status="DATA FROZEN",
        config_status="frozen-final",
        final_c=FROZEN_C,
    )
    argv = _argv(fixture, "--periods", "historical_evaluation", "--allow-2025")
    assert runner.main(argv) == 0

    assert _artifacts(fixture) == [
        "fx_text_study_primary_historical_evaluation.json",
        "fx_text_study_secondary_historical_evaluation.json",
    ]
    payload = json.loads(
        (fixture["results"] / "fx_text_study_primary_historical_evaluation.json").read_text()
    )
    key_metrics = payload["result"]["key_metrics"]
    assert key_metrics["n_eval_rows"] > 0
    assert key_metrics["c_selection"] == "fixed_from_frozen_config"
    assert key_metrics["selected_regularization"] == FROZEN_C
    assert payload["c_source"] == "frozen_config_final_selected_c"
    assert payload["historical_evaluation_label"] == "PREVIOUSLY INSPECTED / HISTORICAL EVALUATION"
    assert payload["corpus"]["filings_manifest_status"] == "DATA FROZEN"


def test_pure_pre_2025_run_cannot_produce_2025_numbers(
    runner: ModuleType, tmp_path: Path, sessions: pd.DatetimeIndex, calendar: NyseCalendar
) -> None:
    """Even with --allow-2025 supplied, a pre-2025-only --periods list must not
    evaluate 2025: the flag permits, it does not select."""
    fixture = _build_fixture(
        tmp_path,
        sessions,
        calendar,
        filings_status="DATA FROZEN",
        config_status="frozen-final",
        final_c=FROZEN_C,
    )
    assert runner.main(_argv(fixture, "--periods", "validation", "--allow-2025")) == 0
    assert _artifacts(fixture) == [
        "fx_text_study_primary_validation.json",
        "fx_text_study_secondary_validation.json",
    ]
