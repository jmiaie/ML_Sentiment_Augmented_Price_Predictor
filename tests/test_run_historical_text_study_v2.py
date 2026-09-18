"""Offline tests for the authoritative D9-D v2 runner (no network).

The runner is exercised end to end against a synthetic 2-issuer fixture corpus
built in tmp_path -- real NYSE session index, the real index-CSV schema, real
text files on disk, real manifests -- so the 2025 gates and the C-fixing path
are proven through the same code path the data takes, not by re-asserting the
gate formula in isolation.

Covered (change E, extended by the D9-D review's corrections):
  * DEV + 2024 validation write artifacts and a ledger, and never 2025
  * 2025 refused without --allow-2025
  * 2025 refused while config status is not frozen-final
  * 2025 refused unless BOTH targets have a complete pre-2025-selected C
    (correction 3: final_selected_c is per target, not one shared dictionary)
  * ANY run, 2025 or not, refused while EITHER manifest is not exactly
    DATA FROZEN, or is missing sha256.dataset_canonical (correction 2)
  * a "DATA FROZEN (PARTIAL: ...)" corpus is refused for every run
  * with all gates satisfied, 2025 runs ONCE and uses the frozen per-target C
    verbatim (c_selection == "fixed_from_frozen_config"), never re-selecting
  * --select-final-c writes only the selection report, never a study artifact,
    and never reaches 2025
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import socket
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest
import yaml

from quant_sentiment.acquisition import canonical_dataset_hash
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
# Distinct values per target: a target mix-up (the secondary target silently
# receiving the primary target's C) fails loudly instead of passing quietly.
FROZEN_C_SECONDARY = {
    "model0_majority_baseline": None,
    "model1_market_only": 1.0,
    "model2_text_only": 0.01,
    "model3_market_text_combined": 0.1,
}
FROZEN_C_BY_TARGET: dict[str, dict[str, float | None]] = {
    "primary": FROZEN_C,
    "secondary": FROZEN_C_SECONDARY,
}
# Shapes the 2025 gate must reject: a target that was never selected (all null)
# and one that is only half selected.
_MALFORMED_FINAL_C: dict[str, dict[str, Any]] = {
    # The pre-selection schema the config ships with.
    "both_targets_literal_null": {"primary": None, "secondary": None},
    "primary_target_never_selected": {
        **FROZEN_C_BY_TARGET,
        "primary": None,
    },
    "primary_target_half_selected": {
        **FROZEN_C_BY_TARGET,
        "primary": {"model0_majority_baseline": None, "model1_market_only": 0.1},
    },
    "secondary_target_never_selected": {
        **FROZEN_C_BY_TARGET,
        "secondary": None,
    },
}

# Sentinel for _build_fixture: use the canonical these fixture bytes actually
# hash to. A literal string overrides it (negative tests); None omits the field.
_RECOMPUTED: Any = object()
FROZEN_AT = "2026-09-17T20:46:08Z"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_historical_text_study_v2", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# One module instance, shared by the `runner` fixture and by _build_fixture,
# which installs the fixture-scale frozen-input record into it.
_RUNNER = _load_runner()


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    return _RUNNER


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
    filings_status: str = "DATA FROZEN",
    prices_status: str = "DATA FROZEN",
    filings_canonical: Any = _RECOMPUTED,
    prices_canonical: Any = _RECOMPUTED,
    config_status: str = "pre-registered",
    final_c: dict[str, Any] | None = None,
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

    # The fixture corpus is frozen for real: every recorded per-file hash and
    # canonical hash is the one these bytes actually have, exactly as the real
    # freeze leaves them. So the runner's byte-level gate is satisfied by
    # construction, and a single mutated byte refuses (the integrity tests).
    filings_file_hashes: dict[str, str] = {}
    filing_rows = 0
    for symbol in SYMBOLS:
        index_path = filings_ds / "filings" / f"{symbol}_index.csv"
        with index_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rel = str(row["text_relpath"])
                filings_file_hashes[rel] = hashlib.sha256(
                    (filings_ds / rel).read_bytes()
                ).hexdigest()
                filing_rows += 1
    for symbol in SYMBOLS:
        index_rel = f"filings/{symbol}_index.csv"
        filings_file_hashes[index_rel] = hashlib.sha256(
            (filings_ds / index_rel).read_bytes()
        ).hexdigest()
    filings_real_canonical = canonical_dataset_hash(filings_file_hashes)

    price_row_counts: dict[str, int] = {}
    prices_file_hashes: dict[str, str] = {}
    for symbol in [*SYMBOLS, MARKET]:
        price_path = prices_ds / f"{symbol}.csv"
        with price_path.open(encoding="utf-8", newline="") as handle:
            price_row_counts[symbol] = sum(1 for _ in csv.DictReader(handle))
        prices_file_hashes[f"{symbol}.csv"] = hashlib.sha256(price_path.read_bytes()).hexdigest()
    prices_real_canonical = canonical_dataset_hash(prices_file_hashes)

    def _declared(chosen: Any, recomputed: str) -> str | None:
        """Sentinel -> the recomputed hash; None -> record no canonical at all."""
        if chosen is _RECOMPUTED:
            return recomputed
        return None if chosen is None else str(chosen)

    filings_declared = _declared(filings_canonical, filings_real_canonical)
    prices_declared = _declared(prices_canonical, prices_real_canonical)

    manifests = tmp_path / "data" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    filings_manifest: dict[str, Any] = {
        "dataset_id": FILINGS_DS,
        "symbols": SYMBOLS,
        "status": filings_status,
        "freeze_timestamp_utc": FROZEN_AT,
        "sha256": dict(filings_file_hashes),
        "parameters": {
            "text_n_files": filing_rows,
            "truncation_count": 0,
            "extraction_policy": "uncapped",
            "text_max_chars": None,
            "text_files_at_legacy_cap_200000": 0,
        },
    }
    if filings_declared is not None:
        filings_manifest["sha256"]["dataset_canonical"] = filings_declared
    (manifests / f"{FILINGS_DS}.json").write_text(
        json.dumps(filings_manifest), encoding="utf-8"
    )

    prices_manifest: dict[str, Any] = {
        "dataset_id": PRICES_DS,
        "symbols": [*SYMBOLS, MARKET],
        "status": prices_status,
        "freeze_timestamp_utc": FROZEN_AT,
        "sha256": dict(prices_file_hashes),
        "row_counts": price_row_counts,
    }
    if prices_declared is not None:
        prices_manifest["sha256"]["dataset_canonical"] = prices_declared
    (manifests / f"{PRICES_DS}.json").write_text(json.dumps(prices_manifest), encoding="utf-8")

    # The production record pins the real 12-issuer / 2,349-row snapshot, which a
    # 2-issuer fixture cannot match; install the fixture-scale record instead.
    fixture_record = {
        "freeze_timestamp_utc": FROZEN_AT,
        "filings_canonical": filings_real_canonical,
        "filings_files": filing_rows + len(SYMBOLS),
        "filings_text_files": filing_rows,
        "filings_rows": filing_rows,
        "issuers": len(SYMBOLS),
        "prices_canonical": prices_real_canonical,
        "price_csvs": len(SYMBOLS) + 1,
    }
    _RUNNER.__dict__["FROZEN_INPUTS"] = fixture_record

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
        "manifests": manifests,
        "frozen_inputs": fixture_record,
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
    assert payload["corpus"]["filings_dataset_id"] == FILINGS_DS
    assert payload["corpus"]["prices_dataset_id"] == PRICES_DS
    assert payload["corpus"]["filings_manifest_status"] == "DATA FROZEN"
    assert payload["corpus"]["prices_manifest_status"] == "DATA FROZEN"
    # The corpus block reports the canonical RECOMPUTED from the bytes on disk by
    # the frozen-input gate, not the value echoed out of the manifest.
    integrity = payload["input_integrity"]
    assert integrity["verified"] is True
    assert integrity["verification_mode"] == "frozen_bytes_sha256"
    assert payload["corpus"]["filings_dataset_canonical_sha256"] == (
        integrity["filings_canonical_recomputed"]
    )
    assert payload["corpus"]["prices_dataset_canonical_sha256"] == (
        integrity["prices_canonical_recomputed"]
    )
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


@pytest.mark.parametrize("manifest", ["filings", "prices"])
def test_2025_refused_while_corpus_is_not_data_frozen(
    runner: ModuleType,
    tmp_path: Path,
    sessions: pd.DatetimeIndex,
    calendar: NyseCalendar,
    manifest: str,
) -> None:
    fixture = _build_fixture(
        tmp_path,
        sessions,
        calendar,
        config_status="frozen-final",
        final_c=FROZEN_C_BY_TARGET,
        filings_status="VALIDATED" if manifest == "filings" else "DATA FROZEN",
        prices_status="VALIDATED" if manifest == "prices" else "DATA FROZEN",
    )
    argv = _argv(fixture, "--periods", "historical_evaluation", "--allow-2025")
    assert runner.main(argv) == 2
    assert _artifacts(fixture) == []


@pytest.mark.parametrize("manifest", ["filings", "prices"])
def test_ordinary_run_refused_while_either_manifest_is_not_data_frozen(
    runner: ModuleType,
    tmp_path: Path,
    sessions: pd.DatetimeIndex,
    calendar: NyseCalendar,
    manifest: str,
) -> None:
    """Correction 2: DEV/validation are gated too -- a frozen filings manifest
    over a merely VALIDATED prices manifest is a half-frozen corpus, and prices
    feed every feature and every target."""
    fixture = _build_fixture(
        tmp_path,
        sessions,
        calendar,
        filings_status="VALIDATED" if manifest == "filings" else "DATA FROZEN",
        prices_status="VALIDATED" if manifest == "prices" else "DATA FROZEN",
    )
    assert runner.main(_argv(fixture)) == 2
    assert _artifacts(fixture) == []


@pytest.mark.parametrize("manifest", ["filings", "prices"])
def test_partial_freeze_corpus_is_refused_for_ordinary_runs(
    runner: ModuleType,
    tmp_path: Path,
    sessions: pd.DatetimeIndex,
    calendar: NyseCalendar,
    manifest: str,
) -> None:
    fixture = _build_fixture(
        tmp_path,
        sessions,
        calendar,
        filings_status=(
            "DATA FROZEN (PARTIAL: 3 failed)" if manifest == "filings" else "DATA FROZEN"
        ),
        prices_status=(
            "DATA FROZEN (PARTIAL: 3 failed)" if manifest == "prices" else "DATA FROZEN"
        ),
    )
    assert runner.main(_argv(fixture)) == 2
    assert _artifacts(fixture) == []


@pytest.mark.parametrize("manifest", ["filings", "prices"])
def test_run_refused_when_a_manifest_lacks_its_canonical_dataset_hash(
    runner: ModuleType,
    tmp_path: Path,
    sessions: pd.DatetimeIndex,
    calendar: NyseCalendar,
    manifest: str,
) -> None:
    """A DATA FROZEN label without a canonical hash is an unverifiable snapshot."""
    fixture = _build_fixture(
        tmp_path,
        sessions,
        calendar,
        filings_canonical=None if manifest == "filings" else "a" * 64,
        prices_canonical=None if manifest == "prices" else "b" * 64,
    )
    assert runner.main(_argv(fixture)) == 2
    assert _artifacts(fixture) == []


@pytest.mark.parametrize("label", sorted(_MALFORMED_FINAL_C))
def test_2025_refused_unless_both_targets_have_a_complete_frozen_c(
    runner: ModuleType,
    tmp_path: Path,
    sessions: pd.DatetimeIndex,
    calendar: NyseCalendar,
    label: str,
) -> None:
    fixture = _build_fixture(
        tmp_path,
        sessions,
        calendar,
        config_status="frozen-final",
        final_c=_MALFORMED_FINAL_C[label],
    )
    argv = _argv(fixture, "--periods", "historical_evaluation", "--allow-2025")
    assert runner.main(argv) == 2
    assert _artifacts(fixture) == []


def test_select_final_c_cannot_be_combined_with_allow_2025(
    runner: ModuleType, tmp_path: Path, sessions: pd.DatetimeIndex, calendar: NyseCalendar
) -> None:
    fixture = _build_fixture(
        tmp_path,
        sessions,
        calendar,
        config_status="frozen-final",
        final_c=FROZEN_C_BY_TARGET,
    )
    argv = _argv(
        fixture, "--select-final-c", "--allow-2025", "--periods", "historical_evaluation"
    )
    assert runner.main(argv) == 2
    assert _artifacts(fixture) == []


def test_select_final_c_writes_only_the_selection_report_and_never_2025(
    runner: ModuleType, tmp_path: Path, sessions: pd.DatetimeIndex, calendar: NyseCalendar
) -> None:
    """The selection mode exists so the reviewed 2024-only selection can be
    pasted into walk_forward.final_selected_c. It runs no period study, writes
    no ledger row, and rewrites neither manifest."""
    fixture = _build_fixture(tmp_path, sessions, calendar)
    manifests = sorted((fixture["raw"].parent / "manifests").glob("*.json"))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in manifests}

    assert runner.main(_argv(fixture, "--select-final-c")) == 0

    assert _artifacts(fixture) == ["fx_text_study_final_c_selection.json"]
    assert not fixture["ledger"].exists()

    report = json.loads(
        (fixture["results"] / "fx_text_study_final_c_selection.json").read_text()
    )
    assert report["selection_max_session"] == "2024-12-31"
    assert set(report["targets"]) == {"primary", "secondary"}
    for target in report["targets"].values():
        assert target["validation_period"]["end_inclusive"] == "2024-12-31"
        assert target["n_validation_rows"] > 0
        assert target["n_formation_rows"] > 0
        assert target["c_grid"] == [0.01, 0.1, 1.0, 10.0]
        assert target["selected_c"]["model0_majority_baseline"] is None
        for model, value in target["selected_c"].items():
            if model == "model0_majority_baseline":
                continue
            assert value in (0.01, 0.1, 1.0, 10.0)
            assert set(target["validation_log_loss_by_model_and_c"][model]) == {
                "0.01",
                "0.1",
                "1.0",
                "10.0",
            }

    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in manifests}
    assert after == before  # selection reads the corpus, it never rewrites it


def test_2025_runs_once_with_all_gates_satisfied_and_uses_the_frozen_c(
    runner: ModuleType, tmp_path: Path, sessions: pd.DatetimeIndex, calendar: NyseCalendar
) -> None:
    fixture = _build_fixture(
        tmp_path,
        sessions,
        calendar,
        filings_status="DATA FROZEN",
        config_status="frozen-final",
        final_c=FROZEN_C_BY_TARGET,
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


# ---------------------------------------------------------------------------
# FINAL PRE-EMPIRICAL P1 GATE: the runner verifies the ACTUAL frozen bytes
# ---------------------------------------------------------------------------


def _assert_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("network access attempted during input verification")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)


def _mutate_one_byte(path: Path) -> None:
    """Change one byte in place: same length, same row count, different bytes."""
    raw = bytearray(path.read_bytes())
    assert raw, f"nothing to mutate in {path}"
    raw[0] = ord("Y") if raw[0] != ord("Y") else ord("Z")
    path.write_bytes(bytes(raw))


# The three kinds of frozen input a real run consumes.
_FROZEN_TARGETS: dict[str, Callable[[dict[str, Any]], Path]] = {
    "filing_text": lambda fx: sorted(
        (fx["raw"] / FILINGS_DS / "filings" / "text").rglob("*.txt")
    )[0],
    "filing_index_csv": lambda fx: fx["raw"] / FILINGS_DS / "filings" / f"{SYMBOLS[0]}_index.csv",
    "price_csv": lambda fx: fx["raw"] / PRICES_DS / f"{SYMBOLS[0]}.csv",
}


def test_production_frozen_inputs_record_pins_the_real_snapshot() -> None:
    """The shipped record must describe the real frozen snapshot.

    Every other test here runs a fixture-scale record, so this is the one place
    the production pin itself is held to account: a loosened count or a stale
    hash fails here instead of quietly weakening every run.
    """
    record = _load_runner().FROZEN_INPUTS
    assert record == {
        "freeze_timestamp_utc": "2026-09-17T20:46:08Z",
        "filings_canonical": "3b2941870391b8584afaad46417cf3e6a0fe03509238fc6ac21958bb4c90f7f4",
        "filings_files": 2361,
        "filings_text_files": 2349,
        "filings_rows": 2349,
        "issuers": 12,
        "prices_canonical": "6ca6e433983fb5f629d084d0964229d5b9c91cbee0756f686598bea6c90d4cb2",
        "price_csvs": 13,
    }


def test_valid_frozen_fixture_records_measured_input_integrity(
    runner: ModuleType, tmp_path: Path, sessions: pd.DatetimeIndex, calendar: NyseCalendar
) -> None:
    """Test 1: a valid frozen fixture passes, and the artifact carries MEASURED
    evidence -- counts and a canonical recomputed from the bytes on disk."""
    fixture = _build_fixture(tmp_path, sessions, calendar)
    assert runner.main(_argv(fixture)) == 0

    payload = json.loads(
        (fixture["results"] / "fx_text_study_primary_validation.json").read_text()
    )
    integrity = payload["input_integrity"]
    record = fixture["frozen_inputs"]
    assert integrity["verified"] is True
    assert integrity["verification_mode"] == "frozen_bytes_sha256"
    assert integrity["filings_issuers_verified"] == record["issuers"]
    assert integrity["filings_files_verified"] == record["filings_files"]
    assert integrity["filings_text_files_verified"] == record["filings_text_files"]
    assert integrity["filings_index_rows_ok"] == record["filings_rows"]
    assert integrity["prices_files_verified"] == record["price_csvs"]
    assert integrity["filings_canonical_recomputed"] == record["filings_canonical"]
    assert integrity["prices_canonical_recomputed"] == record["prices_canonical"]
    # Recomputed independently here from the raw files, so the artifact provably
    # carries a measured value rather than one copied out of the manifest.
    dataset_dir = fixture["raw"] / FILINGS_DS
    independently_hashed = {
        str(path.relative_to(dataset_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(dataset_dir.rglob("*"))
        if path.is_file()
    }
    assert integrity["filings_canonical_recomputed"] == canonical_dataset_hash(
        independently_hashed
    )
    assert integrity["filings_manifest_sha256"] == hashlib.sha256(
        (fixture["manifests"] / f"{FILINGS_DS}.json").read_bytes()
    ).hexdigest()
    assert integrity["prices_manifest_sha256"] == hashlib.sha256(
        (fixture["manifests"] / f"{PRICES_DS}.json").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("target", sorted(_FROZEN_TARGETS))
def test_one_mutated_frozen_byte_refuses_before_any_empirical_work(
    runner: ModuleType,
    tmp_path: Path,
    sessions: pd.DatetimeIndex,
    calendar: NyseCalendar,
    capsys: pytest.CaptureFixture[str],
    target: str,
) -> None:
    """Tests 2-4 and 8: one flipped byte in a filing text, a filing index CSV or
    a price CSV refuses, for the integrity reason, creating no result artifact
    and no ledger row."""
    fixture = _build_fixture(tmp_path, sessions, calendar)
    _mutate_one_byte(_FROZEN_TARGETS[target](fixture))

    assert runner.main(_argv(fixture)) == 2
    assert "frozen input integrity check failed" in capsys.readouterr().err
    assert _artifacts(fixture) == []
    assert not fixture["ledger"].exists()


@pytest.mark.parametrize("target", sorted(_FROZEN_TARGETS))
def test_missing_frozen_file_refuses_before_any_empirical_work(
    runner: ModuleType,
    tmp_path: Path,
    sessions: pd.DatetimeIndex,
    calendar: NyseCalendar,
    capsys: pytest.CaptureFixture[str],
    target: str,
) -> None:
    """Test 5: a frozen input no longer on disk refuses."""
    fixture = _build_fixture(tmp_path, sessions, calendar)
    _FROZEN_TARGETS[target](fixture).unlink()

    assert runner.main(_argv(fixture)) == 2
    assert "frozen input integrity check failed" in capsys.readouterr().err
    assert _artifacts(fixture) == []
    assert not fixture["ledger"].exists()


@pytest.mark.parametrize("dataset_id", [FILINGS_DS, PRICES_DS])
def test_manifest_recorded_per_file_hash_mismatch_refuses(
    runner: ModuleType,
    tmp_path: Path,
    sessions: pd.DatetimeIndex,
    calendar: NyseCalendar,
    capsys: pytest.CaptureFixture[str],
    dataset_id: str,
) -> None:
    """Test 6: a manifest per-file hash that disagrees with the bytes refuses."""
    fixture = _build_fixture(tmp_path, sessions, calendar)
    manifest_path = fixture["manifests"] / f"{dataset_id}.json"
    manifest = json.loads(manifest_path.read_text())
    key = next(k for k in manifest["sha256"] if k != "dataset_canonical")
    manifest["sha256"][key] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert runner.main(_argv(fixture)) == 2
    assert "frozen input integrity check failed" in capsys.readouterr().err
    assert _artifacts(fixture) == []
    assert not fixture["ledger"].exists()


@pytest.mark.parametrize("dataset_id", [FILINGS_DS, PRICES_DS])
def test_canonical_mismatch_against_the_frozen_record_refuses(
    runner: ModuleType,
    tmp_path: Path,
    sessions: pd.DatetimeIndex,
    calendar: NyseCalendar,
    capsys: pytest.CaptureFixture[str],
    dataset_id: str,
) -> None:
    """Test 7: the frozen record's canonical does not match the bytes."""
    fixture = _build_fixture(tmp_path, sessions, calendar)
    if dataset_id == FILINGS_DS:
        fixture["frozen_inputs"]["filings_canonical"] = "0" * 64
    else:
        fixture["frozen_inputs"]["prices_canonical"] = "0" * 64

    assert runner.main(_argv(fixture)) == 2
    assert "frozen input integrity check failed" in capsys.readouterr().err
    assert _artifacts(fixture) == []
    assert not fixture["ledger"].exists()


@pytest.mark.parametrize("dataset_id", [FILINGS_DS, PRICES_DS])
def test_manifest_declared_canonical_mismatch_refuses(
    runner: ModuleType,
    tmp_path: Path,
    sessions: pd.DatetimeIndex,
    calendar: NyseCalendar,
    capsys: pytest.CaptureFixture[str],
    dataset_id: str,
) -> None:
    """Test 7: the manifest's own declared canonical does not match the bytes."""
    fixture = _build_fixture(tmp_path, sessions, calendar)
    manifest_path = fixture["manifests"] / f"{dataset_id}.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"]["dataset_canonical"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert runner.main(_argv(fixture)) == 2
    assert "frozen input integrity check failed" in capsys.readouterr().err
    assert _artifacts(fixture) == []
    assert not fixture["ledger"].exists()


@pytest.mark.parametrize("dataset_id", [FILINGS_DS, PRICES_DS])
def test_re_frozen_manifest_refuses(
    runner: ModuleType,
    tmp_path: Path,
    sessions: pd.DatetimeIndex,
    calendar: NyseCalendar,
    capsys: pytest.CaptureFixture[str],
    dataset_id: str,
) -> None:
    """A later freeze stamp is a different snapshot, whatever its hashes say."""
    fixture = _build_fixture(tmp_path, sessions, calendar)
    manifest_path = fixture["manifests"] / f"{dataset_id}.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["freeze_timestamp_utc"] = "2026-09-18T00:00:00Z"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert runner.main(_argv(fixture)) == 2
    assert "frozen input integrity check failed" in capsys.readouterr().err
    assert _artifacts(fixture) == []


def test_input_verification_makes_no_network_call(
    runner: ModuleType,
    tmp_path: Path,
    sessions: pd.DatetimeIndex,
    calendar: NyseCalendar,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 9: the gate, and the run that passes through it, touch no socket."""
    fixture = _build_fixture(tmp_path, sessions, calendar)
    _assert_no_network(monkeypatch)
    # If the blocker cannot fire, the run below would prove nothing.
    with pytest.raises(AssertionError):
        socket.create_connection(("example.invalid", 80))

    assert runner.main(_argv(fixture)) == 0
    assert _artifacts(fixture) != []


_ENTRY_POINTS: dict[str, tuple[tuple[str, ...], dict[str, Any]]] = {
    "dev_and_2024": ((), {}),
    "select_final_c": (("--select-final-c",), {}),
    "validation_only": (("--periods", "validation"), {}),
    "historical_evaluation_2025": (
        ("--periods", "historical_evaluation", "--allow-2025"),
        {"config_status": "frozen-final", "final_c": FROZEN_C_BY_TARGET},
    ),
}


@pytest.mark.parametrize("label", sorted(_ENTRY_POINTS))
def test_every_entry_point_passes_through_the_same_integrity_gate(
    runner: ModuleType,
    tmp_path: Path,
    sessions: pd.DatetimeIndex,
    calendar: NyseCalendar,
    capsys: pytest.CaptureFixture[str],
    label: str,
) -> None:
    """Test 10: DEV/2024, 2024-only, --select-final-c and 2025 share ONE gate.
   The control run of the same shape proves the entry point is otherwise
   allowed through, so the refusal below is attributable to the mutated byte
   rather than to a gate that refuses everything.
    """
    extra, kwargs = _ENTRY_POINTS[label]

    control_root = tmp_path / "control"
    control_root.mkdir()
    control = _build_fixture(control_root, sessions, calendar, **kwargs)
    assert runner.main(_argv(control, *extra)) == 0
    capsys.readouterr()

    mutated_root = tmp_path / "mutated"
    mutated_root.mkdir()
    mutated = _build_fixture(mutated_root, sessions, calendar, **kwargs)
    _mutate_one_byte(_FROZEN_TARGETS["price_csv"](mutated))

    assert runner.main(_argv(mutated, *extra)) == 2
    assert "frozen input integrity check failed" in capsys.readouterr().err
    assert _artifacts(mutated) == []
    assert not mutated["ledger"].exists()


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
        final_c=FROZEN_C_BY_TARGET,
    )
    assert runner.main(_argv(fixture, "--periods", "validation", "--allow-2025")) == 0
    assert _artifacts(fixture) == [
        "fx_text_study_primary_validation.json",
        "fx_text_study_secondary_validation.json",
    ]
