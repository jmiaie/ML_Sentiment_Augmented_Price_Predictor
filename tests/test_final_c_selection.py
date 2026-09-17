"""Unit tests for the pre-2025 final-C selection (D9-D review, correction 3).

The runner's integration tests exercise ``--select-final-c`` through the full
corpus path. These tests pin the properties that make the selection legitimate
in the first place: it can only ever see pre-2025 evidence, it never grids the
evaluation window, and its output is a deterministic function of the pre-2025
data alone -- so the 2025 HISTORICAL EVALUATION cannot leak into it.
"""

from __future__ import annotations

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

FORMATION = PeriodSpec("formation_dev", "2015-01-01", "2023-12-31")
VALIDATION_2024 = PeriodSpec("validation", "2024-01-01", "2024-12-31")
EVALUATION_2025 = PeriodSpec("historical_evaluation", "2025-01-01", "2025-12-31")
WALK_FORWARD_KWARGS: dict[str, Any] = {
    "initial_train_size": 8,
    "validation_size": 4,
    "step_size": 4,
    "formation_internal_test_size": 4,
}


def _frame(
    *, n_formation: int = 60, n_validation: int = 20, n_2025: int = 20, seed: int = 0
) -> pd.DataFrame:
    """A frame shaped like the runner's event frame, dated across all periods."""
    rng = np.random.default_rng(seed)
    sessions = [
        *pd.date_range("2015-01-02", periods=n_formation, freq="B"),
        *pd.date_range("2024-01-02", periods=n_validation, freq="B"),
        *pd.date_range("2025-01-02", periods=n_2025, freq="B"),
    ]
    n_rows = len(sessions)
    ordinal = np.arange(n_rows, dtype=float)
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


def _select(frame: pd.DataFrame, target_column: str = PRIMARY_TARGET_COLUMN) -> dict[str, Any]:
    return select_final_c_on_validation(
        frame,
        formation=FORMATION,
        validation=VALIDATION_2024,
        target_column=target_column,
        embargo_sessions=1,
        **WALK_FORWARD_KWARGS,
    )


def test_selector_refuses_a_2025_validation_window() -> None:
    """The 2025 window is an evaluation window; it must never be a selection one."""
    with pytest.raises(RuntimeError, match="2025 information must never reach"):
        select_final_c_on_validation(
            _frame(),
            formation=FORMATION,
            validation=EVALUATION_2025,
            target_column=PRIMARY_TARGET_COLUMN,
            embargo_sessions=1,
            **WALK_FORWARD_KWARGS,
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
    # Purged walk-forward holds out the tail of the formation window, so the
    # training rows are a strict subset of it.
    assert 0 < selection["n_pre_2025_train_rows"] < selection["n_formation_rows"]


def test_an_empty_validation_window_is_refused() -> None:
    empty_validation = PeriodSpec("validation", "2024-06-01", "2024-06-02")
    with pytest.raises(ValueError, match="Empty validation frame"):
        select_final_c_on_validation(
            _frame(),
            formation=FORMATION,
            validation=empty_validation,
            target_column=PRIMARY_TARGET_COLUMN,
            embargo_sessions=1,
            **WALK_FORWARD_KWARGS,
        )
