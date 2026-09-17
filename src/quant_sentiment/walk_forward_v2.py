"""Purged, embargoed walk-forward splitting for event-level (irregular,
multi-issuer) samples -- Directive #9 D9-D AUTHORITATIVE (v2).

v1's ``validation.build_walk_forward_plan`` assumes each row is one fixed
time step (one trading session) and protects overlapping labels with a
pure ROW-COUNT gap (``gap = label_horizon + embargo`` positions). That
assumption breaks for D9-D's event-level sample unit: 12 issuers can each
file on their own schedule, so consecutive ROWS in a pooled,
session-ordered event frame are not one session apart -- several events
can share a session (multiple issuers filing the same day), or be many
sessions apart. A row-count gap in that setting does not reliably
correspond to a session-count embargo.

This module purges by actual NYSE session distance instead (the standard
"purged & embargoed" cross-validation technique for overlapping,
irregularly-timed financial labels -- see Lopez de Prado, "Advances in
Financial Machine Learning," ch. 7): a training row is kept only if its
own label's forward-looking window (``target_end_session_ordinal``) ends
at least ``embargo_sessions`` NYSE sessions before the evaluation
window's first event.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventSplit:
    name: str
    train_indices: np.ndarray
    evaluation_indices: np.ndarray


@dataclass(frozen=True)
class EventWalkForwardPlan:
    validation_splits: list[EventSplit]
    pre_test_train_indices: np.ndarray
    test_indices: np.ndarray
    embargo_sessions: int


def build_purged_event_walk_forward_plan(
    effective_session_ordinal: pd.Series,
    target_end_session_ordinal: pd.Series,
    *,
    initial_train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int,
    embargo_sessions: int,
) -> EventWalkForwardPlan:
    """``effective_session_ordinal``/``target_end_session_ordinal`` are
    per-row NYSE session ordinals (see ``nyse_calendar.NyseCalendar.
    session_ordinal``) -- the row's own decision session and the last
    session its own forward label depends on. Rows must already be sorted
    ascending by ``effective_session_ordinal`` (ties broken deterministically
    by the caller, e.g. by CIK) -- this function does not re-sort, so a
    caller-side sort bug would silently produce a wrong split rather than
    being masked here."""
    n = len(effective_session_ordinal)
    if len(target_end_session_ordinal) != n:
        raise ValueError("effective_session_ordinal and target_end_session_ordinal length mismatch")
    if not effective_session_ordinal.is_monotonic_increasing:
        raise ValueError("effective_session_ordinal must be sorted ascending")
    if min(initial_train_size, validation_size, test_size, step_size) <= 0:
        raise ValueError("split sizes must all be positive")
    if embargo_sessions < 0:
        raise ValueError("embargo_sessions must be non-negative")

    effective = effective_session_ordinal.to_numpy()
    target_end = target_end_session_ordinal.to_numpy()

    def _purged_train_indices(train_end: int, eval_start_row: int) -> np.ndarray:
        purge_cutoff = effective[eval_start_row] - embargo_sessions
        candidate = np.arange(0, train_end, dtype=int)
        keep = target_end[candidate] < purge_cutoff
        return candidate[keep]

    development_size = n - test_size
    if development_size <= initial_train_size + validation_size:
        raise ValueError("Not enough observations for validation folds and a final test window.")

    splits: list[EventSplit] = []
    train_end = initial_train_size
    fold_number = 1
    while train_end + validation_size <= development_size:
        eval_start = train_end
        eval_end = eval_start + validation_size
        train_idx = _purged_train_indices(train_end, eval_start)
        if len(train_idx) > 0:
            splits.append(
                EventSplit(
                    name=f"validation_fold_{fold_number}",
                    train_indices=train_idx,
                    evaluation_indices=np.arange(eval_start, eval_end, dtype=int),
                )
            )
            fold_number += 1
        train_end += step_size

    if not splits:
        raise ValueError("Unable to construct any purged walk-forward validation splits.")

    test_start = development_size
    test_indices = np.arange(test_start, test_start + test_size, dtype=int)
    pre_test_train_indices = _purged_train_indices(development_size, test_start)
    if len(pre_test_train_indices) == 0:
        raise ValueError("Purging left no training rows before the final test window.")

    return EventWalkForwardPlan(
        validation_splits=splits,
        pre_test_train_indices=pre_test_train_indices,
        test_indices=test_indices,
        embargo_sessions=embargo_sessions,
    )
