"""Unit tests for the pre-2025 final-C selection (D9-D review, correction 3).

The runner's integration tests exercise ``--select-final-c`` through the full
corpus path. These tests pin the properties that make the selection legitimate
in the first place: it can only ever see pre-2025 evidence, it never grids the
evaluation window, and its output is a deterministic function of the pre-2025
data alone -- so the 2025 HISTORICAL EVALUATION cannot leak into it.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

import quant_sentiment.historical_text_study_v2 as study
from quant_sentiment.historical_text_study_v2 import (
    C_GRID,
    FINAL_C_SELECTION_MAX_SESSION,
    MODEL_SPECS,
    PRIMARY_TARGET_COLUMN,
    SECONDARY_TARGET_COLUMN,
    PeriodSpec,
    select_final_c_on_validation,
)
from quant_sentiment.nyse_calendar import NyseCalendar

FORMATION = PeriodSpec("formation_dev", "2015-01-01", "2023-12-31")
VALIDATION_2024 = PeriodSpec("validation", "2024-01-01", "2024-12-31")
EVALUATION_2025 = PeriodSpec("historical_evaluation", "2025-01-01", "2025-12-31")
CAL = NyseCalendar(schedule_start="2015-01-01", schedule_end="2027-12-31")


def _sessions(start: str, n: int) -> list[pd.Timestamp]:
    """``n`` consecutive real NYSE sessions from ``start`` (which must be one)."""
    first = CAL.session_ordinal(date.fromisoformat(start))
    return [pd.Timestamp(CAL.session_at_ordinal(first + k)) for k in range(n)]


def _ordinals(sessions: list[pd.Timestamp]) -> np.ndarray:
    return np.array([CAL.session_ordinal(s.date()) for s in sessions], dtype=float)


def _frame(
    *, n_formation: int = 60, n_validation: int = 20, n_2025: int = 20, seed: int = 0
) -> pd.DataFrame:
    """A frame shaped like the runner's event frame, dated across all periods.

    Real NYSE sessions and real session ordinals: the target-window boundary
    invariant is a session-ordinal comparison, so synthetic business-day
    ordinals would make these tests vacuous.
    """
    rng = np.random.default_rng(seed)
    sessions = [
        *_sessions("2015-01-05", n_formation),
        *_sessions("2024-01-02", n_validation),
        *_sessions("2025-01-02", n_2025),
    ]
    n_rows = len(sessions)
    ordinal = _ordinals(sessions)
    data: dict[str, Any] = {
        "effective_session": sessions,
        "effective_session_ordinal": ordinal,
        "primary_target_end_session_ordinal": ordinal + 1,
        "secondary_target_end_session_ordinal": ordinal + 5,
        PRIMARY_TARGET_COLUMN: rng.integers(0, 2, n_rows),
        SECONDARY_TARGET_COLUMN: rng.integers(0, 2, n_rows),
    }
    for column in sorted({c for cols in MODEL_SPECS.values() for c in cols}):
        data[column] = rng.normal(size=n_rows)
    return pd.DataFrame(data)


def _frame_with_2024_boundary_rows(*, seed: int = 0) -> pd.DataFrame:
    """The standard frame plus late-Dec-2024 rows whose forward windows close in
    2025 -- the rows the target-horizon invariant has to exclude.

    * 2024-12-31 -- primary closes 2025-01-02, secondary closes 2025-01-08:
      excluded from BOTH validation slices.
    * 2024-12-30 -- primary closes 2024-12-31 (legitimately in-period, so it
      stays), secondary closes 2025-01-07 (excluded from the secondary slice).
    """
    frame = _frame(seed=seed)
    rng = np.random.default_rng(seed + 99)
    extra: list[dict[str, Any]] = []
    for day, primary_end, secondary_end in (
        ("2024-12-31", "2025-01-02", "2025-01-08"),
        ("2024-12-30", "2024-12-31", "2025-01-07"),
    ):
        row: dict[str, Any] = {
            "effective_session": pd.Timestamp(day),
            "effective_session_ordinal": float(CAL.session_ordinal(date.fromisoformat(day))),
            "primary_target_end_session_ordinal": float(
                CAL.session_ordinal(date.fromisoformat(primary_end))
            ),
            "secondary_target_end_session_ordinal": float(
                CAL.session_ordinal(date.fromisoformat(secondary_end))
            ),
            PRIMARY_TARGET_COLUMN: int(rng.integers(0, 2)),
            SECONDARY_TARGET_COLUMN: int(rng.integers(0, 2)),
        }
        for column in sorted({c for cols in MODEL_SPECS.values() for c in cols}):
            row[column] = float(rng.normal())
        extra.append(row)
    combined = pd.concat([frame, pd.DataFrame(extra)], ignore_index=True)
    return combined.sort_values("effective_session_ordinal").reset_index(drop=True)


def _select(frame: pd.DataFrame, target_column: str = PRIMARY_TARGET_COLUMN) -> dict[str, Any]:
    return select_final_c_on_validation(
        frame,
        formation=FORMATION,
        validation=VALIDATION_2024,
        target_column=target_column,
        calendar=CAL,
        embargo_sessions=1,
    )


def test_selector_refuses_a_2025_validation_window() -> None:
    """The 2025 window is an evaluation window; it must never be a selection one."""
    with pytest.raises(RuntimeError, match="2025 information must never reach"):
        select_final_c_on_validation(
            _frame(),
            formation=FORMATION,
            validation=EVALUATION_2025,
            target_column=PRIMARY_TARGET_COLUMN,
            calendar=CAL,
            embargo_sessions=1,
        )


def test_selection_max_session_is_the_end_of_2024() -> None:
    assert FINAL_C_SELECTION_MAX_SESSION == "2024-12-31"


@pytest.mark.parametrize("target_column", [PRIMARY_TARGET_COLUMN, SECONDARY_TARGET_COLUMN])
def test_every_logistic_model_gets_a_grid_value_and_the_baseline_gets_none(
    target_column: str,
) -> None:
    selection = _select(_frame(), target_column)
    selected = selection["selected_c"]

    assert set(selected) == set(MODEL_SPECS)
    assert selected["model0_majority_baseline"] is None
    for model, columns in MODEL_SPECS.items():
        if not columns:
            continue
        assert selected[model] in C_GRID
        # Every grid value was actually scored, so "selected" means "won".
        assert set(selection["validation_log_loss_by_model_and_c"][model]) == {
            str(c) for c in C_GRID
        }


def test_selection_is_deterministic() -> None:
    frame = _frame()
    assert _select(frame)["selected_c"] == _select(frame)["selected_c"]


def test_ties_break_toward_the_smallest_c(monkeypatch: pytest.MonkeyPatch) -> None:
    """With identical loss at every C, the most regularized (smallest) value wins."""

    def _constant_probabilities(
        *,
        train_frame: pd.DataFrame,
        evaluation_frame: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        c_value: float | None = None,
    ) -> np.ndarray:
        return np.full(evaluation_frame.shape[0], 0.5)

    monkeypatch.setattr(study, "_fit_probabilities", _constant_probabilities)

    selected = _select(_frame())["selected_c"]
    for model, columns in MODEL_SPECS.items():
        if columns:
            assert selected[model] == min(C_GRID)


def test_2025_rows_cannot_influence_the_selection() -> None:
    """Scrambling every 2025 row must not change the selected C: the selector
    reads the formation and 2024 validation windows only."""
    frame = _frame()
    baseline = _select(frame)["selected_c"]

    scrambled = frame.copy()
    is_2025 = scrambled["effective_session"] >= pd.Timestamp("2025-01-01")
    n_2025 = int(is_2025.sum())
    rng = np.random.default_rng(1234)
    for column in scrambled.columns:
        if column == "effective_session":
            continue
        if column in (PRIMARY_TARGET_COLUMN, SECONDARY_TARGET_COLUMN):
            # Inverting the labels corrupts them just as thoroughly as noise
            # would, and keeps the integer dtype pandas requires.
            scrambled.loc[is_2025, column] = 1 - scrambled.loc[is_2025, column]
        else:
            scrambled.loc[is_2025, column] = rng.normal(size=n_2025)

    assert _select(scrambled)["selected_c"] == baseline


def test_report_records_the_pre_2025_evidence_it_used() -> None:
    selection = _select(_frame())

    assert selection["formation_period"]["end_inclusive"] == "2023-12-31"
    assert selection["validation_period"]["end_inclusive"] == "2024-12-31"
    assert selection["selection_max_session"] == "2024-12-31"
    assert selection["c_grid"] == list(C_GRID)
    assert selection["tie_break"] == "smallest_c"
    assert selection["n_formation_rows"] > 0
    assert selection["n_validation_rows"] > 0
    # P1 semantics: the 2024 fit uses ALL admissible pre-2024 observations, not
    # the plan's reserved pre-test block, so the training rows are a subset of
    # the formation window. The rows the rule DOES hold back (a label window
    # reaching into 2024) are covered in tests/test_execution_geometry.py.
    assert 0 < selection["n_pre_2025_train_rows"] <= selection["n_formation_rows"]
    assert (
        selection["pre_evaluation_training"]["n_train_rows"]
        == selection["n_pre_2025_train_rows"]
    )


def test_an_empty_validation_window_is_refused() -> None:
    empty_validation = PeriodSpec("validation", "2024-06-01", "2024-06-02")
    with pytest.raises(ValueError, match="Empty validation frame"):
        select_final_c_on_validation(
            _frame(),
            formation=FORMATION,
            validation=empty_validation,
            target_column=PRIMARY_TARGET_COLUMN,
            calendar=CAL,
            embargo_sessions=1,
        )


def test_2024_boundary_rows_are_excluded_from_the_selection_slices() -> None:
    """A 2024-12-31 filing's forward window closes in 2025, so it is not pre-2025
    evidence -- and the selection report states how many rows the boundary rule
    removed, per target, instead of hiding them."""
    primary = _select(_frame_with_2024_boundary_rows())
    assert primary["target_window_slicing"]["validation"]["n_excluded_target_window_crossing"] == 1
    assert primary["target_window_slicing"]["formation"]["n_excluded_target_window_crossing"] == 0

    secondary = _select(_frame_with_2024_boundary_rows(), SECONDARY_TARGET_COLUMN)
    secondary_stats = secondary["target_window_slicing"]["validation"]
    assert secondary_stats["n_excluded_target_window_crossing"] == 2


def test_2024_boundary_rows_cannot_influence_the_selection() -> None:
    """Corrupting the boundary rows (2024-effective session, target end in 2025)
    cannot move the selected C: they never enter the slices."""
    frame = _frame_with_2024_boundary_rows()
    baseline = _select(frame)["selected_c"]

    # A boundary crossing only exists for a row whose effective session is
    # already inside 2024: a 2025 row is not a 2024 row with a late window.
    in_2024 = frame["effective_session"].between(
        pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")
    )
    last_2024 = CAL.session_ordinal(date(2024, 12, 31))
    crossing = in_2024 & (frame["primary_target_end_session_ordinal"] > last_2024)
    assert int(crossing.sum()) == 1

    scrambled = frame.copy()
    rng = np.random.default_rng(4321)
    n = int(crossing.sum())
    # The window columns are deliberately NOT corrupted: the boundary rule reads
    # them, so overwriting one stops the row from being a boundary row at all
    # (measured: n_excluded_target_window_crossing 1 -> 0) instead of testing
    # that a boundary row is held out. Everything a model fits on -- features and
    # labels -- is corrupted.
    for column in scrambled.columns:
        if column in (
            "effective_session",
            "effective_session_ordinal",
            "primary_target_end_session_ordinal",
            "secondary_target_end_session_ordinal",
        ):
            continue
        if column in (PRIMARY_TARGET_COLUMN, SECONDARY_TARGET_COLUMN):
            scrambled.loc[crossing, column] = 1 - scrambled.loc[crossing, column]
        else:
            scrambled.loc[crossing, column] = rng.normal(size=n)

    corrupted = _select(scrambled)
    # Fail-if-broken: the row must still be held out, i.e. the corruption above
    # must not have quietly made it admissible again.
    assert (
        corrupted["target_window_slicing"]["validation"]["n_excluded_target_window_crossing"]
        == 1
    )
    assert corrupted["selected_c"] == baseline


def test_the_boundary_exclusion_is_load_bearing() -> None:
    """Positive control: were the 2024-12-31 row's window (wrongly) counted, its
    label WOULD move the validation grid. So the exclusion changes the evidence
    rather than nothing, and the test above cannot pass vacuously."""
    frame = _frame_with_2024_boundary_rows()
    # A boundary crossing only exists for a row whose effective session is
    # already inside 2024: a 2025 row is not a 2024 row with a late window.
    in_2024 = frame["effective_session"].between(
        pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")
    )
    last_2024 = CAL.session_ordinal(date(2024, 12, 31))
    crossing = in_2024 & (frame["primary_target_end_session_ordinal"] > last_2024)

    pinned = frame.copy()
    pinned.loc[crossing, "primary_target_end_session_ordinal"] = float(last_2024)
    flipped = pinned.copy()
    flipped.loc[crossing, PRIMARY_TARGET_COLUMN] = 1 - flipped.loc[crossing, PRIMARY_TARGET_COLUMN]

    counted = _select(pinned)["validation_log_loss_by_model_and_c"]
    counted_flipped = _select(flipped)["validation_log_loss_by_model_and_c"]
    assert counted != counted_flipped, (
        "the boundary row's label does not move the validation grid even when counted, "
        "so test_2024_boundary_rows_cannot_influence_the_selection would be vacuous"
    )
