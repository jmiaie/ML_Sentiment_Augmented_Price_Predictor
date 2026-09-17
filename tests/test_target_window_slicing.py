"""Target-horizon boundary invariant tests (independent-review defect, 2026-09-17).

A filing's forward target window can close AFTER the period the event belongs
to: a 2024-12-31 filing's 1-session target closes on the first 2025 session,
and a 2024-12-30 filing's 5-session target closes in Jan-2025. Slicing a period
on ``effective_session`` alone would pull those later-period outcomes into the
period's evidence -- and for the 2024 validation window that means 2025
information could reach final-C selection.

``slice_period`` therefore requires BOTH:
  1. ``effective_session`` inside the period, and
  2. that target's complete forward window closing inside the SAME period
     (``*_target_end_session_ordinal <= period.last_session_ordinal``),
both as real NYSE session-ordinal comparisons against the calendar that built
the frame -- never calendar-day arithmetic.

Every test below uses a real NYSE calendar, so the ordinals are real session
positions and the boundaries are real session boundaries.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from quant_sentiment.historical_text_study_v2 import (
    PRIMARY_TARGET_COLUMN,
    SECONDARY_TARGET_COLUMN,
    PeriodSpec,
    slice_period,
)
from quant_sentiment.nyse_calendar import NyseCalendar

CAL = NyseCalendar()
FORMATION = PeriodSpec("formation_dev", "2015-01-01", "2023-12-31")
VALIDATION = PeriodSpec("validation", "2024-01-01", "2024-12-31")
EVALUATION_2025 = PeriodSpec("historical_evaluation", "2025-01-01", "2025-12-31")

LAST_SESSION_2023 = date(2023, 12, 29)  # 2023-12-31 is a Sunday
LAST_SESSION_2024 = date(2024, 12, 31)
LAST_SESSION_2025 = date(2025, 12, 31)


def _row(effective: str, *, primary_end: str, secondary_end: str) -> dict[str, Any]:
    """One event row: an effective session plus both targets' window-close
    sessions, all converted to real session ordinals via ``CAL``."""
    return {
        "effective_session": pd.Timestamp(effective),
        "effective_session_ordinal": float(CAL.session_ordinal(date.fromisoformat(effective))),
        "primary_target_end_session_ordinal": float(
            CAL.session_ordinal(date.fromisoformat(primary_end))
        ),
        "secondary_target_end_session_ordinal": float(
            CAL.session_ordinal(date.fromisoformat(secondary_end))
        ),
        PRIMARY_TARGET_COLUMN: 1,
        SECONDARY_TARGET_COLUMN: 1,
    }


def _frame(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows)).sort_values("effective_session_ordinal").reset_index(drop=True)


def _slice(
    frame: pd.DataFrame, period: PeriodSpec, target_column: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return slice_period(frame, period, target_column=target_column, calendar=CAL)


# A 2023 event whose complete windows close in 2024: filed on the last 2023
# session, primary target closes on the first 2024 session, secondary in the
# second week of 2024.
_LATE_2023 = _row("2023-12-29", primary_end="2024-01-02", secondary_end="2024-01-05")
_LATE_2024 = _row("2024-12-31", primary_end="2025-01-02", secondary_end="2025-01-08")
_LATE_2025 = _row("2025-12-31", primary_end="2026-01-02", secondary_end="2026-01-08")


def test_2023_primary_closing_in_2024_is_not_formation_primary_evidence() -> None:
    kept, stats = _slice(_frame(_LATE_2023), FORMATION, PRIMARY_TARGET_COLUMN)
    assert kept.empty
    assert stats["n_effective_session_in_period"] == 1
    assert stats["n_excluded_target_window_crossing"] == 1
    assert stats["n_included"] == 0


def test_2023_secondary_closing_in_2024_is_not_formation_secondary_evidence() -> None:
    kept, stats = _slice(_frame(_LATE_2023), FORMATION, SECONDARY_TARGET_COLUMN)
    assert kept.empty
    assert stats["n_excluded_target_window_crossing"] == 1


def test_2024_primary_closing_in_2025_is_not_validation_primary_evidence() -> None:
    kept, stats = _slice(_frame(_LATE_2024), VALIDATION, PRIMARY_TARGET_COLUMN)
    assert kept.empty
    assert stats["n_excluded_target_window_crossing"] == 1


def test_2024_secondary_closing_in_2025_is_not_validation_secondary_evidence() -> None:
    kept, stats = _slice(_frame(_LATE_2024), VALIDATION, SECONDARY_TARGET_COLUMN)
    assert kept.empty
    assert stats["n_excluded_target_window_crossing"] == 1


def test_a_2024_event_whose_complete_horizon_closes_in_2024_is_kept() -> None:
    """The invariant must not be a blanket truncation of December: a filing
    whose whole window closes inside 2024 stays in the 2024 evidence."""
    legit = _row("2024-12-20", primary_end="2024-12-23", secondary_end="2024-12-30")
    kept, stats = _slice(_frame(legit), VALIDATION, PRIMARY_TARGET_COLUMN)
    assert len(kept) == 1
    assert stats["n_excluded_target_window_crossing"] == 0
    assert stats["n_included"] == 1

    kept_secondary, stats_secondary = _slice(_frame(legit), VALIDATION, SECONDARY_TARGET_COLUMN)
    assert len(kept_secondary) == 1
    assert stats_secondary["n_excluded_target_window_crossing"] == 0


def test_boundary_is_the_last_real_session_not_the_calendar_end_date() -> None:
    """2023-12-31 is a Sunday: the formation boundary must resolve to the last
    actual session (2023-12-29), and a window closing there counts as inside."""
    at_boundary = _row("2023-12-20", primary_end="2023-12-29", secondary_end="2023-12-29")
    kept, stats = _slice(_frame(at_boundary), FORMATION, PRIMARY_TARGET_COLUMN)
    assert len(kept) == 1
    assert stats["period_last_session"] == str(LAST_SESSION_2023)
    assert stats["period_last_session_ordinal"] == CAL.session_ordinal(LAST_SESSION_2023)


def test_the_same_rule_bounds_the_2025_historical_evaluation() -> None:
    """The eventual 2025 evaluation follows the identical complete-window rule:
    a 2025-12-31 filing whose targets close in 2026 is not 2025 evidence."""
    kept, stats = _slice(_frame(_LATE_2025), EVALUATION_2025, PRIMARY_TARGET_COLUMN)
    assert kept.empty
    assert stats["n_excluded_target_window_crossing"] == 1
    assert stats["period_last_session"] == str(LAST_SESSION_2025)

    legit = _row("2025-12-19", primary_end="2025-12-22", secondary_end="2025-12-29")
    kept_legit, stats_legit = _slice(_frame(legit), EVALUATION_2025, SECONDARY_TARGET_COLUMN)
    assert len(kept_legit) == 1
    assert stats_legit["n_excluded_target_window_crossing"] == 0


def test_exclusions_are_counted_per_target_and_per_period() -> None:
    """Each slice reports its own crossing count, so the artifacts can state how
    much evidence the boundary rule removed instead of hiding it."""
    frame = _frame(_LATE_2023, _LATE_2024)
    formation_primary = _slice(frame, FORMATION, PRIMARY_TARGET_COLUMN)[1]
    formation_secondary = _slice(frame, FORMATION, SECONDARY_TARGET_COLUMN)[1]
    validation_primary = _slice(frame, VALIDATION, PRIMARY_TARGET_COLUMN)[1]
    validation_secondary = _slice(frame, VALIDATION, SECONDARY_TARGET_COLUMN)[1]

    assert (formation_primary["period"], formation_primary["target_column"]) == (
        "formation_dev",
        PRIMARY_TARGET_COLUMN,
    )
    assert formation_primary["n_excluded_target_window_crossing"] == 1
    assert formation_secondary["n_excluded_target_window_crossing"] == 1
    assert validation_primary["n_excluded_target_window_crossing"] == 1
    assert validation_secondary["n_excluded_target_window_crossing"] == 1

    # Accounting identity: every in-period row is either included, dropped for a
    # crossing window, or dropped for a missing target end -- no row vanishes.
    for stats in (formation_primary, formation_secondary, validation_primary, validation_secondary):
        assert stats["n_effective_session_in_period"] == (
            stats["n_included"]
            + stats["n_excluded_target_window_crossing"]
            + stats["n_excluded_missing_target_end"]
        )


def test_a_missing_target_end_is_never_counted_as_inside_the_window() -> None:
    """No target window means no proof it closes in-period: excluded, and
    counted separately from a boundary crossing."""
    row = _LATE_2024 | {"primary_target_end_session_ordinal": float("nan")}
    kept, stats = _slice(_frame(row), VALIDATION, PRIMARY_TARGET_COLUMN)
    assert kept.empty
    assert stats["n_excluded_missing_target_end"] == 1
    assert stats["n_excluded_target_window_crossing"] == 0


def test_an_unknown_target_column_is_refused_rather_than_silently_mis_sliced() -> None:
    """A typo must not fall back to the secondary target's window."""
    with pytest.raises(ValueError, match="unknown target column"):
        _slice(_frame(_LATE_2024), VALIDATION, "primary_direction_typo")
