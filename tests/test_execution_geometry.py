"""Execution-geometry tests (independent-review P1).

Four claims are pinned here, each with a control that fails if the claim is
vacuous:

* DEV scores ONLY the formation plan's reserved internal test block, and that
  block cannot overlap the rows the model was fitted on.
* The 2024 validation run trains on ALL admissible pre-2024 observations -- the
  DEV stage's reserved test rows are ordinary fitting data by then.
* ``--select-final-c`` uses the IDENTICAL pre-2024 training geometry (proved by
  hashing the training row set and comparing, not by reading the comments).
* The 2025 evaluation expands to every admissible earlier observation -- which
  includes the 2024 rows -- while tuning nothing, and it still excludes a
  late-2024 row whose label window crosses into 2025.
"""

from __future__ import annotations

import hashlib
from datetime import date
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant_sentiment import historical_text_study_v2 as study
from quant_sentiment.nyse_calendar import NyseCalendar

PC = study.PRIMARY_TARGET_COLUMN
SC = study.SECONDARY_TARGET_COLUMN
FORMATION = study.PeriodSpec("formation_dev", "2015-01-01", "2023-12-31")
VALIDATION_2024 = study.PeriodSpec("validation", "2024-01-01", "2024-12-31")
EVALUATION_2025 = study.PeriodSpec("historical_evaluation", "2025-01-01", "2025-12-31")
CAL = NyseCalendar(schedule_start="2015-01-01", schedule_end="2027-12-31")
DEV_FOLD_KWARGS: dict[str, Any] = {
    "initial_train_size": 20,
    "validation_size": 8,
    "step_size": 8,
    "formation_internal_test_size": 8,
}


def _sessions(start: str, n: int) -> list[pd.Timestamp]:
    first = CAL.session_ordinal(date.fromisoformat(start))
    return [pd.Timestamp(CAL.session_at_ordinal(first + k)) for k in range(n)]


def _frame_with_boundary_rows(*, seed: int = 7) -> pd.DataFrame:
    """Real NYSE sessions across all three periods, plus the two late-2024 rows
    whose forward windows close in 2025 (2024-12-30 closes 2024-12-31 for the
    primary target, 2024-12-31 closes 2025-01-02), so the boundary rule has
    something to exclude."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    sessions = [
        *_sessions("2015-01-05", 40),
        *_sessions("2024-01-02", 20),
        *_sessions("2025-01-02", 15),
    ]
    ends = [(s, s + 1, s + 5) for s in [CAL.session_ordinal(x.date()) for x in sessions]]
    ends.append((CAL.session_ordinal(date(2024, 12, 30)),
                 CAL.session_ordinal(date(2024, 12, 31)),
                 CAL.session_ordinal(date(2025, 1, 7))))
    ends.append((CAL.session_ordinal(date(2024, 12, 31)),
                 CAL.session_ordinal(date(2025, 1, 2)),
                 CAL.session_ordinal(date(2025, 1, 8))))
    by_ordinal = {int(o): (str(CAL.session_at_ordinal(int(o))), s, p) for o, s, p in ends}
    feature_columns = sorted({c for cols in study.MODEL_SPECS.values() for c in cols})
    for ordinal in sorted(by_ordinal):
        session, primary_end, secondary_end = by_ordinal[ordinal]
        row: dict[str, Any] = {
            "ticker": f"T{ordinal % 3}",
            "effective_session": pd.Timestamp(session),
            "effective_session_ordinal": float(ordinal),
            "primary_target_end_session_ordinal": float(primary_end),
            "secondary_target_end_session_ordinal": float(secondary_end),
            PC: int(rng.integers(0, 2)),
            SC: int(rng.integers(0, 2)),
        }
        for column in feature_columns:
            row[column] = float(rng.normal())
        rows.append(row)
    return pd.DataFrame(rows)


def _expected_train_positions(
    frame: pd.DataFrame, target_column: str, period: study.PeriodSpec, embargo: int
) -> list[int]:
    """Re-derive the admissible training row set from the STATED rule, the same
    way a reviewer would: the row's own forward label window must close at least
    ``embargo`` sessions before the first ACTUAL evaluation event."""
    end_col = study._target_end_ordinal_column(target_column)
    eval_frame, _ = study.slice_period(
        frame, period, target_column=target_column, calendar=CAL
    )
    eval_frame = eval_frame.dropna(subset=[target_column]).reset_index(drop=True)
    first_eval_ordinal = int(eval_frame["effective_session_ordinal"].iloc[0])
    cutoff = first_eval_ordinal - embargo
    ends = frame[end_col]
    first_eval_session = eval_frame["effective_session"].iloc[0]
    return [
        position
        for position in range(len(frame))
        if not pd.isna(ends.iloc[position])
        and int(ends.iloc[position]) < cutoff
        and frame["effective_session"].iloc[position] < first_eval_session
        and not pd.isna(frame[target_column].iloc[position])
    ]


def _positions_sha(positions: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(int(position)) for position in positions).encode("utf-8")
    ).hexdigest()


def _run(frame: pd.DataFrame, period: study.PeriodSpec, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "formation": FORMATION,
        "eval_period": period,
        "target_column": PC,
        "embargo_sessions": 1,
        "allow_holdout": period == EVALUATION_2025,
        "calendar": CAL,
        **DEV_FOLD_KWARGS,
    }
    kwargs.update(overrides)
    return study.run_period_study(frame, **kwargs)


# ---- DEV stage: internal out-of-sample test only -------------------------


def test_dev_scores_only_the_reserved_internal_test_block() -> None:
    frame = _frame_with_boundary_rows()
    km = _run(frame, FORMATION)["key_metrics"]

    assert km["evaluation_block_label"] == "DEVELOPMENT INTERNAL OOS TEST"
    assert km["train_eval_intersection_count"] == 0
    assert km["n_train_rows"] > 0
    assert km["n_internal_test_rows"] > 0
    # The DEV block lives inside the formation window: it is neither the 2024
    # validation nor the 2025 holdout.
    assert km["internal_test_start_session"] >= "2015-01-01"
    assert km["internal_test_end_session"] < "2024-01-01"
    # Falsifiable invariant, not an assertion with two identical sides.
    assert km["train_windows_close_before_eval_start"] is True
    assert (
        km["max_train_target_end_ordinal"]
        < km["eval_first_effective_session_ordinal"] - km["embargo_sessions"]
    )


def test_a_dev_test_block_that_overlaps_its_own_fit_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control for the test above: if the plan handed back an overlapping block,
    the guard must fire. Without this, "intersection == 0" could be vacuous."""
    frame = _frame_with_boundary_rows()
    formation_frame, _ = study.slice_period(
        frame, FORMATION, target_column=PC, calendar=CAL
    )
    formation_frame = formation_frame.dropna(subset=[PC]).reset_index(drop=True)
    real = study._build_formation_plan(
        formation_frame, target_column=PC, embargo_sessions=1, **DEV_FOLD_KWARGS
    )
    overlapping = SimpleNamespace(
        pre_test_train_indices=np.asarray(real.test_indices, dtype=int),
        test_indices=np.asarray(real.test_indices, dtype=int),
        validation_splits=real.validation_splits,
    )
    monkeypatch.setattr(study, "_build_formation_plan", lambda *a, **k: overlapping)

    with pytest.raises(RuntimeError, match="shares"):
        _run(frame, FORMATION)


# ---- 2024 validation: all admissible pre-2024 evidence -------------------


def test_2024_training_uses_all_admissible_pre_2024_evidence() -> None:
    frame = _frame_with_boundary_rows()
    km = _run(frame, VALIDATION_2024)["key_metrics"]
    stats = km["pre_evaluation_training"]
    expected = _expected_train_positions(frame, PC, VALIDATION_2024, 1)

    assert km["evaluation_block_label"] == "EVALUATION PERIOD OUT-OF-SAMPLE"
    assert stats["train_row_index_sha256"] == _positions_sha(expected)
    assert stats["n_train_rows"] == len(expected) == km["n_train_rows"]
    assert km["train_end_session"] < "2024-01-01"  # pre-2024 evidence only
    assert km["train_windows_close_before_eval_start"] is True

    # P1: the DEV plan's reserved internal-test rows are NOT held back again --
    # they are ordinary admissible fitting data for the 2024 stage.
    formation_frame, _ = study.slice_period(
        frame, FORMATION, target_column=PC, calendar=CAL
    )
    formation_frame = formation_frame.dropna(subset=[PC]).reset_index(drop=True)
    plan = study._build_formation_plan(
        formation_frame, target_column=PC, embargo_sessions=1, **DEV_FOLD_KWARGS
    )
    train_sessions = {
        str(session.date()) for session in frame.iloc[expected]["effective_session"]
    }
    reserved_sessions = {
        str(session.date())
        for session in formation_frame.iloc[list(plan.test_indices)]["effective_session"]
    }
    assert reserved_sessions
    assert reserved_sessions <= train_sessions
    assert stats["n_train_rows"] > len(plan.pre_test_train_indices)


def test_final_c_selection_uses_identical_pre_2024_geometry() -> None:
    """Parity by measurement: the two paths must select C on the same row SET."""
    frame = _frame_with_boundary_rows()
    selection = study.select_final_c_on_validation(
        frame,
        formation=FORMATION,
        validation=VALIDATION_2024,
        target_column=PC,
        embargo_sessions=1,
        calendar=CAL,
    )
    km = _run(frame, VALIDATION_2024)["key_metrics"]

    assert (
        selection["pre_evaluation_training"]["train_row_index_sha256"]
        == km["pre_evaluation_training"]["train_row_index_sha256"]
    )
    assert selection["n_pre_2025_train_rows"] == km["n_train_rows"]
    assert (
        selection["pre_evaluation_training"]["purge_cutoff_ordinal"]
        == km["pre_evaluation_training"]["purge_cutoff_ordinal"]
    )


# ---- 2025 evaluation: expanding window, frozen C, no tuning --------------


def test_2025_training_expands_to_2024_rows_and_tunes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a 2025 evaluation must not tune: C is frozen")

    monkeypatch.setattr(study, "_tune_logistic_model", _explode)
    frame = _frame_with_boundary_rows()
    fixed_c = {
        name: (None if not columns else 1.0)
        for name, columns in study.MODEL_SPECS.items()
    }
    km = _run(frame, EVALUATION_2025, fixed_c=fixed_c)["key_metrics"]
    stats = km["pre_evaluation_training"]
    expected = _expected_train_positions(frame, PC, EVALUATION_2025, 1)

    assert km["c_selection"] == "fixed_from_frozen_config"
    assert km["selected_regularization"] == fixed_c
    assert stats["train_row_index_sha256"] == _positions_sha(expected)
    assert stats["n_train_rows"] == len(expected)
    assert km["evaluation_block_label"] == "EVALUATION PERIOD OUT-OF-SAMPLE"
    assert km["train_windows_close_before_eval_start"] is True

    train_sessions = {
        str(session.date()) for session in frame.iloc[expected]["effective_session"]
    }
    # Expanding window: the 2024 rows are admissible earlier observations.
    assert train_sessions and any(session.startswith("2024-") for session in train_sessions)
    assert stats["train_end_session"] >= "2024-01-02"
    assert stats["train_end_session"] < "2025-01-01"
    assert not any(session.startswith("2025-") for session in train_sessions)
    # Late-2024 rows whose label window crosses into 2025 stay excluded.
    assert "2024-12-30" not in train_sessions
    assert "2024-12-31" not in train_sessions


def test_2025_evaluation_is_refused_without_the_holdout_flag() -> None:
    with pytest.raises(RuntimeError, match="FINAL CONFIGURATION FROZEN"):
        _run(_frame_with_boundary_rows(), EVALUATION_2025, allow_holdout=False)


def test_the_2024_rows_really_are_the_ones_the_purge_admits() -> None:
    """Control for the expanding-window claim: drop the 2024 rows and the 2025
    training set must shrink by exactly that many rows -- so the test above is
    measuring the 2024 admission and not some coincidental count."""
    frame = _frame_with_boundary_rows()
    full = _run(frame, EVALUATION_2025, fixed_c=_all_frozen())["key_metrics"]
    without_2024 = frame[~frame["effective_session"].dt.year.eq(2024)].reset_index(
        drop=True
    )
    trimmed = _run(without_2024, EVALUATION_2025, fixed_c=_all_frozen())["key_metrics"]

    admitted_2024 = sum(
        1
        for session in frame.iloc[
            _expected_train_positions(frame, PC, EVALUATION_2025, 1)
        ]["effective_session"]
        if session.year == 2024
    )
    assert admitted_2024 > 0
    assert full["n_train_rows"] - trimmed["n_train_rows"] == admitted_2024


def _all_frozen() -> dict[str, float | None]:
    return {
        name: (None if not columns else 1.0)
        for name, columns in study.MODEL_SPECS.items()
    }


# ---- the shared helper's own refusals ------------------------------------


def test_purge_refuses_an_empty_or_unusable_evaluation_window() -> None:
    frame = _frame_with_boundary_rows()
    with pytest.raises(ValueError, match="empty evaluation frame"):
        study.build_pre_evaluation_train_frame(
            frame, target_column=PC, evaluation_frame=frame.iloc[0:0],
            embargo_sessions=1, calendar=CAL,
        )
    at_the_start = study.PeriodSpec("validation", "2015-01-05", "2015-01-20")
    eval_frame, _ = study.slice_period(
        frame, at_the_start, target_column=PC, calendar=CAL
    )
    with pytest.raises(ValueError, match="purging left no training rows"):
        study.build_pre_evaluation_train_frame(
            frame, target_column=PC,
            evaluation_frame=eval_frame.dropna(subset=[PC]).reset_index(drop=True),
            embargo_sessions=1, calendar=CAL,
        )
