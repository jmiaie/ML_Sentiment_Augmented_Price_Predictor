from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant_sentiment.event_frame_v2 import ALL_FEATURE_COLUMNS, FilingEvent, build_event_frame
from quant_sentiment.historical_text_study_v2 import (
    C_GRID,
    MODEL_SPECS,
    PeriodSpec,
    run_period_study,
)
from quant_sentiment.market_features_v2 import MARKET_FEATURE_COLUMNS_V2
from quant_sentiment.nyse_calendar import NyseCalendar

# The one calendar every synthetic frame and every run_period_study call
# below shares: the boundary invariant compares session ordinals, so the
# frame and the slice must be built from the SAME session list.
CAL = NyseCalendar(schedule_start="2015-01-01", schedule_end="2020-12-31")


def _synthetic_event_frame(n_issuers: int = 3, seed: int = 0) -> pd.DataFrame:
    cal = CAL
    sessions = pd.DatetimeIndex(cal._schedule.index)  # type: ignore[attr-defined]
    rng = np.random.default_rng(seed)

    def price_frame(s: int) -> pd.DataFrame:
        r = np.random.default_rng(s)
        close = 100 * np.exp(np.cumsum(r.normal(0.0003, 0.015, len(sessions))))
        volume = r.integers(1_000_000, 5_000_000, len(sessions)).astype(float)
        return pd.DataFrame({"Close": close, "Volume": volume}, index=sessions.tz_localize(None))

    spy_frame = price_frame(seed + 1)
    issuer_frames = {f"ISSUER{i}": price_frame(seed + 10 + i) for i in range(n_issuers)}

    words = "growth improvement strong record profitable decline loss weak litigation risk".split()
    events = []
    for ticker in issuer_frames:
        for k in range(80, len(sessions) - 30, 22):
            session = sessions[k]
            hour = int(rng.choice([15, 20, 21]))
            ts = session.tz_localize("UTC") + pd.Timedelta(hours=hour)
            text = " ".join(rng.choice(words, size=150))
            events.append(
                FilingEvent(
                    cik=f"cik-{ticker}",
                    ticker=ticker,
                    company=f"{ticker} Inc.",
                    accession=f"{ticker}-{k}",
                    form="8-K",
                    filing_date=str(session.date()),
                    acceptance_datetime=ts,
                    text=text,
                )
            )

    frame, _ = build_event_frame(events, issuer_frames, spy_frame, cal)
    return frame


@pytest.fixture(scope="module")
def event_frame() -> pd.DataFrame:
    return _synthetic_event_frame()


def test_model_specs_match_confirmatory_ablation_structure() -> None:
    assert MODEL_SPECS["model0_majority_baseline"] == []
    assert MODEL_SPECS["model1_market_only"] == MARKET_FEATURE_COLUMNS_V2
    assert set(MODEL_SPECS["model2_text_only"]) == set(ALL_FEATURE_COLUMNS) - set(
        MARKET_FEATURE_COLUMNS_V2
    )
    assert set(MODEL_SPECS["model3_market_text_combined"]) == set(ALL_FEATURE_COLUMNS)
    # Model 3 is exactly the union of Model 1 and Model 2's features -- no
    # extra/missing features sneaking into the "combined" ablation arm.
    assert set(MODEL_SPECS["model3_market_text_combined"]) == set(
        MODEL_SPECS["model1_market_only"]
    ) | set(MODEL_SPECS["model2_text_only"])


def test_c_grid_matches_spec_exactly() -> None:
    assert C_GRID == (0.01, 0.1, 1.0, 10.0)


def test_2025_evaluation_blocked_without_allow_holdout(event_frame: pd.DataFrame) -> None:
    formation = PeriodSpec("formation_dev", "2015-01-01", "2018-12-31")
    historical_eval = PeriodSpec("historical_evaluation", "2019-01-01", "2019-12-31")
    with pytest.raises(RuntimeError, match="FINAL CONFIGURATION FROZEN"):
        run_period_study(
            event_frame,
            calendar=CAL,
            formation=formation,
            eval_period=historical_eval,
            target_column="primary_direction",
            embargo_sessions=1,
            allow_holdout=False,
        )


def test_deterministic_c_selection_is_reproducible(event_frame: pd.DataFrame) -> None:
    formation = PeriodSpec("formation_dev", "2015-01-01", "2018-12-31")
    validation = PeriodSpec("validation", "2019-01-01", "2019-12-31")

    def _run() -> dict[str, Any]:
        return run_period_study(
            event_frame,
            calendar=CAL,
            formation=formation,
            eval_period=validation,
            target_column="primary_direction",
            embargo_sessions=1,
            allow_holdout=False,
            initial_train_size=15,
            validation_size=5,
            step_size=5,
            formation_internal_test_size=5,
            bootstrap_n_resamples=20,
            bootstrap_block_size=3,
        )

    result_a = _run()
    result_b = _run()
    assert result_a["selected_regularization"] == result_b["selected_regularization"]
    assert result_a["key_metrics"]["model3_log_loss"] == result_b["key_metrics"]["model3_log_loss"]


def test_run_produces_headline_delta_and_bootstrap_ci(event_frame: pd.DataFrame) -> None:
    formation = PeriodSpec("formation_dev", "2015-01-01", "2018-12-31")
    validation = PeriodSpec("validation", "2019-01-01", "2019-12-31")
    result = run_period_study(
        event_frame,
        calendar=CAL,
        formation=formation,
        eval_period=validation,
        target_column="primary_direction",
        embargo_sessions=1,
        allow_holdout=False,
        initial_train_size=15,
        validation_size=5,
        step_size=5,
        formation_internal_test_size=5,
        bootstrap_n_resamples=20,
        bootstrap_block_size=3,
    )
    km = result["key_metrics"]
    assert "headline_delta_log_loss_model3_minus_model1" in km
    ci = km["headline_bootstrap_delta_log_loss_ci95"]
    assert ci["ci_low"] <= ci["mean_delta"] <= ci["ci_high"]
    assert len(km["headline_fold_delta_log_loss_distribution"]) == km["walk_forward_fold_count"]
    assert km["embargo_sessions"] == 1
    # Every model's regularization choice came from the spec's own C grid.
    for c in result["selected_regularization"].values():
        assert c is None or c in C_GRID


def test_secondary_target_uses_five_session_embargo(event_frame: pd.DataFrame) -> None:
    formation = PeriodSpec("formation_dev", "2015-01-01", "2018-12-31")
    validation = PeriodSpec("validation", "2019-01-01", "2019-12-31")
    result = run_period_study(
        event_frame,
        calendar=CAL,
        formation=formation,
        eval_period=validation,
        target_column="secondary_direction",
        embargo_sessions=5,
        allow_holdout=False,
        initial_train_size=15,
        validation_size=5,
        step_size=5,
        formation_internal_test_size=5,
        bootstrap_n_resamples=10,
        bootstrap_block_size=3,
    )
    assert result["target_column"] == "secondary_direction"
    assert result["key_metrics"]["embargo_sessions"] == 5


def test_allow_holdout_true_permits_2025_style_period(event_frame: pd.DataFrame) -> None:
    formation = PeriodSpec("formation_dev", "2015-01-01", "2018-12-31")
    historical_eval = PeriodSpec("historical_evaluation", "2019-01-01", "2019-12-31")
    result = run_period_study(
        event_frame,
        calendar=CAL,
        formation=formation,
        eval_period=historical_eval,
        target_column="primary_direction",
        embargo_sessions=1,
        allow_holdout=True,
        initial_train_size=15,
        validation_size=5,
        step_size=5,
        formation_internal_test_size=5,
        bootstrap_n_resamples=10,
        bootstrap_block_size=3,
    )
    assert result["eval_period"]["name"] == "historical_evaluation"
