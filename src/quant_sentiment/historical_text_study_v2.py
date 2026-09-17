"""Directive #9 D9-D AUTHORITATIVE (v2) historical textual-signal study
orchestrator -- supersedes ``historical_text_study.py`` (v1), which
Directive #9 itself labels EXPLORATORY / NON-CONFORMING TO FINAL D9-D
(3 issuers, 8-K only, non-LM lexicon, day-level sample unit with
manufactured non-event days, raw rather than excess-return targets, no
proper NYSE calendar).

Confirmatory models (per spec, never FinBERT/XGBoost here -- those are
EXPLORATORY-only and run, if at all, after this study is frozen/evaluated):

* Model 0 -- majority / unconditional baseline
* Model 1 -- market-only logistic regression
* Model 2 -- text-only logistic regression (real LM categories)
* Model 3 -- market + text logistic regression

Headline comparison: Model 3 minus Model 1, on the PRIMARY (1-session
excess-return direction) target -- "the key research question is
incremental value of text beyond market features. Not raw combined-model
accuracy." The secondary (5-session) target is run through the same four
models as a pre-specified robustness check, not the headline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .event_frame_v2 import ALL_FEATURE_COLUMNS, LM_FEATURE_COLUMNS
from .market_features_v2 import MARKET_FEATURE_COLUMNS_V2
from .modeling import (
    _classification_metrics,
    _fit_probabilities,
    _majority_validation_metrics,
    _tune_logistic_model,
)
from .nyse_calendar import NyseCalendar
from .validation import WalkForwardSplit
from .walk_forward_v2 import (
    EventSplit,
    EventWalkForwardPlan,
    build_purged_event_walk_forward_plan,
)

C_GRID = (0.01, 0.1, 1.0, 10.0)
PRIMARY_TARGET_COLUMN = "primary_direction"
SECONDARY_TARGET_COLUMN = "secondary_direction"
PRIMARY_EMBARGO_SESSIONS = 1
SECONDARY_EMBARGO_SESSIONS = 5
# The 2024 validation window is the LAST window allowed to influence
# regularization selection. 2025 is an evaluation window, never a selection one.
FINAL_C_SELECTION_MAX_SESSION = "2024-12-31"

# Which forward-window end column belongs to which target. A dict (rather than
# an if/else) so an unknown target column RAISES instead of silently slicing on
# the secondary target's window.
TARGET_END_ORDINAL_COLUMN: dict[str, str] = {
    PRIMARY_TARGET_COLUMN: "primary_target_end_session_ordinal",
    SECONDARY_TARGET_COLUMN: "secondary_target_end_session_ordinal",
}


def _target_end_ordinal_column(target_column: str) -> str:
    try:
        return TARGET_END_ORDINAL_COLUMN[target_column]
    except KeyError:
        raise ValueError(
            f"unknown target column {target_column!r}; expected one of "
            f"{sorted(TARGET_END_ORDINAL_COLUMN)}"
        ) from None

MODEL_SPECS: dict[str, list[str]] = {
    "model0_majority_baseline": [],
    "model1_market_only": MARKET_FEATURE_COLUMNS_V2,
    "model2_text_only": LM_FEATURE_COLUMNS,
    "model3_market_text_combined": ALL_FEATURE_COLUMNS,
}


@dataclass(frozen=True)
class PeriodSpec:
    name: str
    start: str
    end_inclusive: str

    def contains_session(self, session: date | pd.Timestamp) -> bool:
        start = date.fromisoformat(self.start)
        end = date.fromisoformat(self.end_inclusive)
        day = session.date() if isinstance(session, pd.Timestamp) else session
        return bool((day - start).days >= 0 and (end - day).days >= 0)

    def last_session_ordinal(self, calendar: NyseCalendar) -> int:
        """Ordinal of this period's last real NYSE session.

        ``end_inclusive`` is a calendar date (2023-12-31 is a Sunday), so the
        period boundary has to be resolved against the actual session
        calendar -- never approximated with calendar-day arithmetic.
        """
        last_day = calendar.last_session_on_or_before(date.fromisoformat(self.end_inclusive))
        return calendar.session_ordinal(last_day)


def slice_period(
    frame: pd.DataFrame,
    period: PeriodSpec,
    *,
    target_column: str,
    calendar: NyseCalendar,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rows of ``frame`` that belong to ``period`` FOR ``target_column``.

    Two conditions, both session-ordinal based against the same NYSE calendar
    that built the frame:

    1. ``effective_session`` falls inside the period, and
    2. that target's complete forward label window CLOSES inside the period
       (``*_target_end_session_ordinal <= period.last_session_ordinal``).

    Condition 2 is the target-horizon boundary invariant: a filing's target can
    end after the period it was filed in (a 2024-12-30 filing's 5-session
    target ends in Jan-2025). Slicing on ``effective_session`` alone would let
    those next-period outcomes into the period's evidence -- for the 2024
    validation window that means 2025 information could reach final-C
    selection, i.e. "pre-2025 C selection" consuming 2025 outcomes.

    Returns ``(frame_for_period, stats)``; ``stats`` carries the boundary
    exclusions separately by target and period so downstream artifacts can
    report them rather than hide them.
    """
    end_col = _target_end_ordinal_column(target_column)
    last_ordinal = period.last_session_ordinal(calendar)
    in_period = frame["effective_session"].map(period.contains_session)
    end_ordinal = frame[end_col]
    window_closed = end_ordinal.notna() & (end_ordinal <= last_ordinal)
    crossing = in_period & end_ordinal.notna() & ~window_closed
    missing_end = in_period & end_ordinal.isna()

    stats: dict[str, Any] = {
        "period": period.name,
        "target_column": target_column,
        "period_last_session": str(calendar.session_at_ordinal(last_ordinal)),
        "period_last_session_ordinal": int(last_ordinal),
        "n_rows_in_frame": int(frame.shape[0]),
        "n_effective_session_in_period": int(in_period.sum()),
        "n_included": int((in_period & window_closed).sum()),
        "n_excluded_target_window_crossing": int(crossing.sum()),
        "n_excluded_missing_target_end": int(missing_end.sum()),
    }
    return frame.loc[in_period & window_closed].reset_index(drop=True), stats


def _session_at_ordinal_or_none(calendar: NyseCalendar, ordinal: int) -> str | None:
    """Session date at ``ordinal``, or None when that ordinal has no session.

    A display-only statistic: a cutoff ordinal below zero or past the end of the
    calendar's session list simply has no session to name, and None says that
    instead of indexing from the wrong end of the list.
    """
    if ordinal < 0:
        return None
    try:
        return str(calendar.session_at_ordinal(ordinal))
    except IndexError:
        return None


def build_pre_evaluation_train_frame(
    full_frame: pd.DataFrame,
    *,
    target_column: str,
    evaluation_frame: pd.DataFrame,
    embargo_sessions: int,
    calendar: NyseCalendar,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every admissible observation that PRECEDES an evaluation window.

    ONE geometry, shared by the 2024 validation run, ``--select-final-c`` and the
    eventual 2025 evaluation, so those three cannot drift into subtly different
    purge rules. Two session-ordinal conditions:

    1. Purge/embargo -- the row's own forward label window closes at least
       ``embargo_sessions`` NYSE sessions before the evaluation window's first
       actual event:
       ``target_end_session_ordinal < first_eval_effective_session_ordinal - embargo``
       (the same strict inequality ``walk_forward_v2`` purges with), so no training
       row's label can overlap an evaluation row's outcome window.
    2. Filing date -- the row must have been filed before the evaluation window's
       first actual event (its own ``effective_session`` precedes it). Keyed on
       that session timestamp, not on a hardcoded period name and not on the
       ordinal column, so the 2024 and the 2025 window share ONE rule while the
       2025 training set still expands to include the admissible 2024 rows.

    There is deliberately NO formation-internal test reservation here. That block
    belongs to the DEV stage; once DEV is finished those observations are ordinary
    admissible model-fitting data for the later windows, and reserving them again
    would truncate the 2024/2025 training sets for no methodological reason.
    """
    end_col = _target_end_ordinal_column(target_column)
    if evaluation_frame.empty:
        raise ValueError("empty evaluation frame: nothing to purge the training rows against")
    first_eval_session = evaluation_frame["effective_session"].iloc[0]
    first_eval_ordinal = int(evaluation_frame["effective_session_ordinal"].iloc[0])
    purge_cutoff = first_eval_ordinal - embargo_sessions

    history_frame = full_frame
    end_ordinal = history_frame[end_col]
    # Both conditions read off the evaluation window's first ACTUAL event, so no
    # period name is hardcoded here: 1) the row was filed before that event
    # (session timestamp, so a corrupted ordinal column cannot smuggle a later
    # row into the fit); 2) its forward label window closes `embargo_sessions`
    # sessions before it.
    admissible = (
        end_ordinal.notna()
        & (end_ordinal < purge_cutoff)
        & (history_frame["effective_session"] < first_eval_session)
        & history_frame[target_column].notna()
    )
    positions = np.flatnonzero(admissible.to_numpy())
    train_frame = history_frame.loc[admissible].reset_index(drop=True)
    if train_frame.empty:
        raise ValueError(
            "purging left no training rows before the evaluation window "
            f"(target_column={target_column!r}, purge cutoff ordinal {purge_cutoff})"
        )
    if len(positions) != train_frame.shape[0]:
        raise RuntimeError("training row bookkeeping mismatch; refusing to report a wrong row set")

    stats: dict[str, Any] = {
        "target_column": target_column,
        "n_history_rows": int(history_frame.shape[0]),
        "first_eval_effective_session": str(
            evaluation_frame["effective_session"].iloc[0].date()
        ),
        "first_eval_effective_session_ordinal": first_eval_ordinal,
        "embargo_sessions": int(embargo_sessions),
        "purge_cutoff_ordinal": int(purge_cutoff),
        "purge_cutoff_session": _session_at_ordinal_or_none(calendar, purge_cutoff),
        "n_train_rows": int(train_frame.shape[0]),
        "train_start_session": str(train_frame["effective_session"].iloc[0].date()),
        "train_end_session": str(train_frame["effective_session"].iloc[-1].date()),
        "train_target_end_max_ordinal": int(end_ordinal.loc[admissible].max()),
        # Identity of the training row SET, so two code paths claiming the same
        # geometry can be compared instead of trusted.
        "train_row_index_sha256": hashlib.sha256(
            ",".join(str(int(position)) for position in positions).encode("utf-8")
        ).hexdigest(),
    }
    return train_frame, stats


def _as_validation_splits(splits: list[EventSplit]) -> list[WalkForwardSplit]:
    """``modeling._tune_logistic_model``/``_majority_validation_metrics``
    only touch ``.name``/``.train_indices``/``.evaluation_indices`` -- this
    adapts ``EventSplit`` (purged, session-distance-based) to the exact
    same shape so those already-tested functions can be reused unchanged."""
    return [
        WalkForwardSplit(
            name=split.name,
            train_indices=split.train_indices,
            evaluation_indices=split.evaluation_indices,
        )
        for split in splits
    ]


def _build_formation_plan(
    formation_frame: pd.DataFrame,
    *,
    target_column: str,
    initial_train_size: int,
    validation_size: int,
    step_size: int,
    formation_internal_test_size: int,
    embargo_sessions: int,
) -> EventWalkForwardPlan:
    """Purged event walk-forward plan over the formation window.

    Used by ``run_period_study``: the validation folds tune C on formation-internal
    evidence, and ``pre_test_train_indices``/``test_indices`` give the DEV stage its
    internal out-of-sample test block.

    This is NOT the training geometry for the 2024 validation run or for the final-C
    selection -- those use ``build_pre_evaluation_train_frame``, which admits every
    admissible pre-evaluation observation rather than reserving a formation-internal
    test block.
    """
    n_form = formation_frame.shape[0]
    test_size = formation_internal_test_size
    if n_form <= initial_train_size + validation_size + test_size:
        test_size = max(5, n_form // 10)
        if n_form <= initial_train_size + validation_size + test_size:
            raise ValueError(
                f"Formation too small for purged walk-forward "
                f"(n={n_form}, need > {initial_train_size + validation_size + test_size})"
            )

    end_col = _target_end_ordinal_column(target_column)
    return build_purged_event_walk_forward_plan(
        formation_frame["effective_session_ordinal"],
        formation_frame[end_col],
        initial_train_size=initial_train_size,
        validation_size=validation_size,
        test_size=test_size,
        step_size=step_size,
        embargo_sessions=embargo_sessions,
    )


def select_final_c_on_validation(
    full_frame: pd.DataFrame,
    *,
    formation: PeriodSpec,
    validation: PeriodSpec,
    target_column: str,
    embargo_sessions: int,
    calendar: NyseCalendar,
    c_values: tuple[float, ...] = C_GRID,
) -> dict[str, Any]:
    """Select the FINAL C on PRE-2025 evidence only.

    Train on ALL admissible pre-2024 development observations, score every C in
    ``c_values`` on the 2024 validation window, and return the per-model argmin
    log-loss. This is what DEV + 2024 exist for; the result is written into
    ``walk_forward.final_selected_c`` and read verbatim by the 2025 run.

    The training set comes from ``build_pre_evaluation_train_frame`` -- the SAME
    helper the ordinary 2024 validation run uses -- so the metrics the selection
    is based on and the metrics reported as validation come from one and the same
    training geometry. The formation plan's internal-test reservation is NOT
    applied: DEV is finished by the time C is selected, and holding those rows back
    again would shrink the 2024 training set on no methodological basis.

    The 2025 window is an evaluation window, not a selection window, so a
    validation period ending after ``FINAL_C_SELECTION_MAX_SESSION`` is
    refused rather than silently consumed. Ties break toward the smaller
    (more regularized) C.

    Both slices are target-window bounded via ``slice_period``, so a 2024
    event whose forward target closes in 2025 is excluded from BOTH the
    training and the validation evidence -- the pre-2025 selection cannot
    consume a 2025 outcome through a boundary row.
    """
    if validation.end_inclusive > FINAL_C_SELECTION_MAX_SESSION:
        raise RuntimeError(
            "final-C selection may only consume a validation window ending on or "
            f"before {FINAL_C_SELECTION_MAX_SESSION}; got one ending "
            f"{validation.end_inclusive!r}. 2025 information must never reach "
            "regularization selection."
        )

    formation_frame, formation_slice = slice_period(
        full_frame, formation, target_column=target_column, calendar=calendar
    )
    formation_frame = formation_frame.dropna(subset=[target_column])
    formation_frame = formation_frame.reset_index(drop=True)
    validation_frame, validation_slice = slice_period(
        full_frame, validation, target_column=target_column, calendar=calendar
    )
    validation_frame = validation_frame.dropna(subset=[target_column])
    validation_frame = validation_frame.reset_index(drop=True)

    if formation_frame.empty:
        raise ValueError("Empty formation frame after dropna")
    if validation_frame.empty:
        raise ValueError(f"Empty validation frame for period {validation.name!r}")

    # Same shared geometry as the ordinary 2024 validation run: ALL admissible
    # pre-2024 development observations, purged against the first ACTUAL 2024
    # evaluation event -- "admissible" is exactly "its forward label window closes
    # before that event minus the embargo", which is what bounds it to pre-2024.
    # Not the formation plan's pre-test training indices -- those exist to keep the
    # DEV stage's internal test out of its own fit, which is not a constraint on
    # the 2024 stage.
    pre_2025_train_frame, pre_2025_train_stats = build_pre_evaluation_train_frame(
        full_frame,
        target_column=target_column,
        evaluation_frame=validation_frame,
        embargo_sessions=embargo_sessions,
        calendar=calendar,
    )

    y_validation = validation_frame[target_column].astype(int).to_numpy()
    selected_c: dict[str, float | None] = {}
    log_loss_by_model_and_c: dict[str, dict[str, float]] = {}
    for model_name, feature_columns in MODEL_SPECS.items():
        if not feature_columns:
            # Model 0 is the majority/unconditional baseline -- there is no C.
            selected_c[model_name] = None
            log_loss_by_model_and_c[model_name] = {}
            continue
        losses: dict[str, float] = {}
        for c_value in c_values:
            probabilities = _fit_probabilities(
                train_frame=pre_2025_train_frame,
                evaluation_frame=validation_frame,
                feature_columns=feature_columns,
                target_column=target_column,
                c_value=c_value,
            )
            metrics = _classification_metrics(y_validation, probabilities)
            losses[str(c_value)] = float(metrics["log_loss"])
        log_loss_by_model_and_c[model_name] = losses
        selected_c[model_name] = min(c_values, key=lambda c: (losses[str(c)], c))

    return {
        "target_column": target_column,
        "formation_period": {
            "name": formation.name,
            "start": formation.start,
            "end_inclusive": formation.end_inclusive,
        },
        "validation_period": {
            "name": validation.name,
            "start": validation.start,
            "end_inclusive": validation.end_inclusive,
        },
        "selection_max_session": FINAL_C_SELECTION_MAX_SESSION,
        "c_grid": list(c_values),
        "tie_break": "smallest_c",
        "n_formation_rows": int(formation_frame.shape[0]),
        "n_pre_2025_train_rows": int(pre_2025_train_frame.shape[0]),
        "pre_evaluation_training": pre_2025_train_stats,
        "n_validation_rows": int(validation_frame.shape[0]),
        # Rows dropped because their forward target window closes outside the
        # period they belong to -- reported per target/period, not hidden.
        "target_window_slicing": {
            "formation": formation_slice,
            "validation": validation_slice,
        },
        "validation_embargo_sessions": embargo_sessions,
        "selected_c": selected_c,
        "validation_log_loss_by_model_and_c": log_loss_by_model_and_c,
        "notes": (
            "C selected on pre-2025 evidence only (purged formation train -> 2024 "
            "validation); the selected values are frozen into the experiment config "
            "and read verbatim by the 2025 evaluation run."
        ),
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def write_json_artifact(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _block_bootstrap_delta_ci(
    losses_model3: np.ndarray,
    losses_model1: np.ndarray,
    *,
    n_resamples: int,
    block_size: int,
    seed: int,
) -> dict[str, float]:
    """Event-aware (block, not iid-per-row) bootstrap CI on
    mean(losses_model3 - losses_model1). Blocks of `block_size`
    consecutive (session-ordered) rows are resampled with replacement,
    preserving local temporal/event dependence within a block rather than
    treating every row as an independent draw."""
    n = len(losses_model3)
    if n == 0:
        return {"mean_delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    diffs = losses_model3 - losses_model1
    rng = np.random.default_rng(seed)
    n_blocks = max(1, n // block_size)
    block_starts = np.arange(0, n - block_size + 1) if n > block_size else np.array([0])
    resample_means = np.empty(n_resamples)
    for i in range(n_resamples):
        chosen_starts = rng.choice(block_starts, size=n_blocks, replace=True)
        resampled = np.concatenate([diffs[s : s + block_size] for s in chosen_starts])
        resample_means[i] = float(resampled.mean())
    return {
        "mean_delta": float(diffs.mean()),
        "ci_low": float(np.percentile(resample_means, 2.5)),
        "ci_high": float(np.percentile(resample_means, 97.5)),
        "n_resamples": n_resamples,
        "block_size": block_size,
    }


def run_period_study(
    full_frame: pd.DataFrame,
    *,
    formation: PeriodSpec,
    eval_period: PeriodSpec,
    target_column: str,
    embargo_sessions: int,
    allow_holdout: bool,
    calendar: NyseCalendar,
    fixed_c: dict[str, float | None] | None = None,
    initial_train_size: int = 60,
    validation_size: int = 20,
    step_size: int = 20,
    formation_internal_test_size: int = 20,
    bootstrap_n_resamples: int = 500,
    bootstrap_block_size: int = 10,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Tune C on formation-period purged walk-forward only; evaluate once
    on eval_period with the locked C (per model) -- freeze-before-2025
    discipline mirrors D9-A/B/C's own pattern in this program.

    ``fixed_c`` (post-freeze path): when supplied, C is NOT re-selected at all
    -- the values frozen from pre-2025 evidence are used verbatim, so a 2025
    evaluation cannot tune on its own window."""
    if eval_period.name in ("historical_evaluation", "holdout") and not allow_holdout:
        raise RuntimeError(
            "2025 evaluation blocked until FINAL CONFIGURATION FROZEN "
            "(pass allow_holdout=True only after freeze)."
        )
    if fixed_c is not None:
        missing = [name for name in MODEL_SPECS if name not in fixed_c]
        if missing:
            raise ValueError(f"fixed_c is missing entries for {missing}")

    formation_frame, formation_slice = slice_period(
        full_frame, formation, target_column=target_column, calendar=calendar
    )
    formation_frame = formation_frame.dropna(subset=[target_column])
    formation_frame = formation_frame.reset_index(drop=True)
    eval_frame, eval_slice = slice_period(
        full_frame, eval_period, target_column=target_column, calendar=calendar
    )
    eval_frame = eval_frame.dropna(subset=[target_column])
    eval_frame = eval_frame.reset_index(drop=True)

    if formation_frame.empty:
        raise ValueError("Empty formation frame after dropna")
    if eval_frame.empty:
        raise ValueError(f"Empty evaluation frame for period {eval_period.name}")

    plan = _build_formation_plan(
        formation_frame,
        target_column=target_column,
        initial_train_size=initial_train_size,
        validation_size=validation_size,
        step_size=step_size,
        formation_internal_test_size=formation_internal_test_size,
        embargo_sessions=embargo_sessions,
    )
    validation_splits = _as_validation_splits(plan.validation_splits)

    selected_regularization: dict[str, float | None] = {}
    validation_metrics: dict[str, list[dict[str, Any]]] = {}
    for model_name, feature_columns in MODEL_SPECS.items():
        if not feature_columns:
            selected_regularization[model_name] = None
            validation_metrics[model_name] = _majority_validation_metrics(
                frame=formation_frame,
                splits=validation_splits,
                target_column=target_column,
            )
            continue
        if fixed_c is not None:
            # Post-freeze evaluation path (2025): C was selected on PRE-2025
            # evidence and written into the frozen config. Re-tuning here would
            # silently re-select C on the evaluation window itself, which
            # constraints.no_retune_after_freeze forbids.
            selected_regularization[model_name] = fixed_c[model_name]
            validation_metrics[model_name] = []
            continue
        selection = _tune_logistic_model(
            frame=formation_frame,
            splits=validation_splits,
            feature_columns=feature_columns,
            target_column=target_column,
            c_values=C_GRID,
        )
        selected_regularization[model_name] = selection.best_c
        validation_metrics[model_name] = selection.validation_metrics

    # ---- training / evaluation geometry (independent-review P1) -----------
    # Two regimes, ONE invariant: a row's forward label window must close before
    # the rows being scored begin. The DEV stage scores the purged plan's own
    # reserved internal test block; every later window (2024 validation, and the
    # eventual 2025 evaluation) trains on ALL admissible earlier observations via
    # the shared helper -- never scoring a row the model was fitted on.
    target_end_col = _target_end_ordinal_column(target_column)
    geometry: dict[str, Any]
    if eval_period == formation:
        train_indices = np.asarray(plan.pre_test_train_indices, dtype=int)
        test_indices = np.asarray(plan.test_indices, dtype=int)
        overlap = np.intersect1d(train_indices, test_indices)
        if overlap.size:
            raise RuntimeError(
                f"DEV internal test block shares {overlap.size} row indices with the "
                "training rows; the headline metrics would not be out of sample."
            )
        pre_test_train_frame = formation_frame.iloc[train_indices]
        eval_frame = formation_frame.iloc[test_indices].reset_index(drop=True)
        geometry = {
            "evaluation_block_label": "DEVELOPMENT INTERNAL OOS TEST",
            "n_train_rows": int(pre_test_train_frame.shape[0]),
            "n_internal_test_rows": int(eval_frame.shape[0]),
            "internal_test_start_session": str(
                eval_frame["effective_session"].iloc[0].date()
            ),
            "internal_test_end_session": str(eval_frame["effective_session"].iloc[-1].date()),
            "train_eval_intersection_count": int(overlap.size),
            "formation_rows_reserved_for_fold_tuning": int(
                formation_frame.shape[0] - train_indices.size - test_indices.size
            ),
        }
    else:
        # Expanding pre-window training: every admissible earlier observation, one
        # rule evaluated one window later. A 2024 validation run therefore fits on
        # pre-2024 rows only (the purge excludes 2024), while the 2025 evaluation
        # also admits the admissible 2024 rows.
        pre_test_train_frame, pre_eval_stats = build_pre_evaluation_train_frame(
            full_frame,
            target_column=target_column,
            evaluation_frame=eval_frame,
            embargo_sessions=embargo_sessions,
            calendar=calendar,
        )
        geometry = {
            "evaluation_block_label": "EVALUATION PERIOD OUT-OF-SAMPLE",
            "n_train_rows": int(pre_test_train_frame.shape[0]),
            "n_internal_test_rows": None,
            "internal_test_start_session": None,
            "internal_test_end_session": None,
            "train_eval_intersection_count": 0,
            "pre_evaluation_training": pre_eval_stats,
        }
    geometry["n_eval_rows"] = int(eval_frame.shape[0])
    geometry["train_start_session"] = str(
        pre_test_train_frame["effective_session"].iloc[0].date()
    )
    geometry["train_end_session"] = str(
        pre_test_train_frame["effective_session"].iloc[-1].date()
    )
    geometry["max_train_target_end_ordinal"] = int(
        pre_test_train_frame[target_end_col].max()
    )
    geometry["eval_first_effective_session_ordinal"] = int(
        eval_frame["effective_session_ordinal"].iloc[0]
    )
    geometry["embargo_sessions"] = int(embargo_sessions)
    geometry["train_windows_close_before_eval_start"] = bool(
        geometry["max_train_target_end_ordinal"]
        < geometry["eval_first_effective_session_ordinal"] - embargo_sessions
    )
    if not geometry["train_windows_close_before_eval_start"]:
        raise RuntimeError(
            "training rows' forward label windows reach into the evaluation window "
            f"(max train target-end ordinal {geometry['max_train_target_end_ordinal']}, "
            f"first eval ordinal {geometry['eval_first_effective_session_ordinal']}, "
            f"embargo {embargo_sessions}); the evaluation would not be out of sample."
        )

    # Locked evaluation: score eval_period once with the locked C.
    period_metrics: dict[str, dict[str, Any]] = {}
    period_probabilities: dict[str, np.ndarray] = {}
    for model_name, feature_columns in MODEL_SPECS.items():
        probabilities = _fit_probabilities(
            train_frame=pre_test_train_frame,
            evaluation_frame=eval_frame,
            feature_columns=feature_columns,
            target_column=target_column,
            c_value=selected_regularization[model_name],
        )
        period_probabilities[model_name] = probabilities
        period_metrics[model_name] = _classification_metrics(
            eval_frame[target_column].astype(int).to_numpy(),
            probabilities,
        )

    y_eval = eval_frame[target_column].astype(int).to_numpy()
    from sklearn.metrics import log_loss

    per_row_loss_model3 = np.array(
        [
            log_loss([y], [[1 - p, p]], labels=[0, 1])
            for y, p in zip(
                y_eval, period_probabilities["model3_market_text_combined"], strict=True
            )
        ]
    )
    per_row_loss_model1 = np.array(
        [
            log_loss([y], [[1 - p, p]], labels=[0, 1])
            for y, p in zip(y_eval, period_probabilities["model1_market_only"], strict=True)
        ]
    )
    headline_bootstrap = _block_bootstrap_delta_ci(
        per_row_loss_model3,
        per_row_loss_model1,
        n_resamples=bootstrap_n_resamples,
        block_size=bootstrap_block_size,
        seed=bootstrap_seed,
    )

    fold_deltas = [
        fold3["log_loss"] - fold1["log_loss"]
        for fold3, fold1 in zip(
            validation_metrics["model3_market_text_combined"],
            validation_metrics["model1_market_only"],
            strict=True,
        )
    ]

    model3 = period_metrics["model3_market_text_combined"]
    model1 = period_metrics["model1_market_only"]
    key_metrics = {
        "n_formation_rows": int(formation_frame.shape[0]),
        # Training/evaluation geometry: row counts, session ranges, and the
        # falsifiable statement that no training row's label window reaches into
        # the scored rows. Reported for every run, not only the DEV one.
        **geometry,
        # Session-ordinal boundary exclusions, per target and period: rows whose
        # effective session is in the period but whose forward target closes
        # outside it, so the period's evidence never consumes a later period's
        # outcomes. Separate from the ordinary dropna/missing-label count.
        "target_window_slicing": {
            "formation": formation_slice,
            "evaluation": eval_slice,
        },
        "n_unique_issuers_eval": int(eval_frame["ticker"].nunique()),
        "eval_start_session": str(eval_frame["effective_session"].iloc[0].date()),
        "eval_end_session": str(eval_frame["effective_session"].iloc[-1].date()),
        "model0_log_loss": period_metrics["model0_majority_baseline"]["log_loss"],
        "model0_balanced_accuracy": period_metrics["model0_majority_baseline"]["balanced_accuracy"],
        "model1_log_loss": model1["log_loss"],
        "model1_brier_score": model1["brier_score"],
        "model1_balanced_accuracy": model1["balanced_accuracy"],
        "model1_roc_auc": model1["roc_auc"],
        "model2_log_loss": period_metrics["model2_text_only"]["log_loss"],
        "model2_balanced_accuracy": period_metrics["model2_text_only"]["balanced_accuracy"],
        "model3_log_loss": model3["log_loss"],
        "model3_brier_score": model3["brier_score"],
        "model3_balanced_accuracy": model3["balanced_accuracy"],
        "model3_roc_auc": model3["roc_auc"],
        "headline_delta_log_loss_model3_minus_model1": float(
            model3["log_loss"] - model1["log_loss"]
        ),
        "headline_delta_balanced_accuracy_model3_minus_model1": float(
            model3["balanced_accuracy"] - model1["balanced_accuracy"]
        ),
        "headline_bootstrap_delta_log_loss_ci95": headline_bootstrap,
        "headline_fold_delta_log_loss_distribution": fold_deltas,
        "selected_regularization": selected_regularization,
        "c_selection": "fixed_from_frozen_config" if fixed_c is not None else "tuned_on_formation",
        "walk_forward_fold_count": len(plan.validation_splits),
        "embargo_sessions": embargo_sessions,
    }

    return {
        "formation": {
            "name": formation.name,
            "start": formation.start,
            "end": formation.end_inclusive,
        },
        "eval_period": {
            "name": eval_period.name,
            "start": eval_period.start,
            "end": eval_period.end_inclusive,
        },
        "target_column": target_column,
        "model_specs": {name: cols for name, cols in MODEL_SPECS.items()},
        "c_grid": list(C_GRID),
        "selected_regularization": selected_regularization,
        "period_metrics": period_metrics,
        "key_metrics": key_metrics,
        "notes": (
            "Event-level filing observations (no manufactured non-event days). "
            "Real Loughran-McDonald categories via pysentiment2 (not vendored). "
            "Excess-return-over-SPY direction target. Purged, session-distance-embargoed "
            "walk-forward (not a row-count gap). C selected on formation only; eval uses "
            "locked C. Headline is Model 3 minus Model 1 log loss, on the PRIMARY target "
            "unless this run's target_column is secondary. Training/evaluation geometry is "
            "reported in key_metrics: a formation_dev run scores the DEVELOPMENT INTERNAL OOS "
            "TEST block (neither the 2025 holdout nor final validation), while a later "
            "evaluation window trains on every admissible earlier observation."
        ),
    }
