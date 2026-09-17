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
from .validation import WalkForwardSplit
from .walk_forward_v2 import EventSplit, build_purged_event_walk_forward_plan

C_GRID = (0.01, 0.1, 1.0, 10.0)
PRIMARY_TARGET_COLUMN = "primary_direction"
SECONDARY_TARGET_COLUMN = "secondary_direction"
PRIMARY_EMBARGO_SESSIONS = 1
SECONDARY_EMBARGO_SESSIONS = 5

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


def slice_period(frame: pd.DataFrame, period: PeriodSpec) -> pd.DataFrame:
    mask = frame["effective_session"].map(lambda s: period.contains_session(s))
    return frame.loc[mask].reset_index(drop=True)


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

    formation_frame = slice_period(full_frame, formation).dropna(subset=[target_column])
    formation_frame = formation_frame.reset_index(drop=True)
    eval_frame = slice_period(full_frame, eval_period).dropna(subset=[target_column])
    eval_frame = eval_frame.reset_index(drop=True)

    if formation_frame.empty:
        raise ValueError("Empty formation frame after dropna")
    if eval_frame.empty:
        raise ValueError(f"Empty evaluation frame for period {eval_period.name}")

    n_form = formation_frame.shape[0]
    test_size = formation_internal_test_size
    if n_form <= initial_train_size + validation_size + test_size:
        test_size = max(5, n_form // 10)
        if n_form <= initial_train_size + validation_size + test_size:
            raise ValueError(
                f"Formation too small for purged walk-forward "
                f"(n={n_form}, need > {initial_train_size + validation_size + test_size})"
            )

    end_col = (
        "primary_target_end_session_ordinal"
        if target_column == PRIMARY_TARGET_COLUMN
        else "secondary_target_end_session_ordinal"
    )
    plan = build_purged_event_walk_forward_plan(
        formation_frame["effective_session_ordinal"],
        formation_frame[end_col],
        initial_train_size=initial_train_size,
        validation_size=validation_size,
        test_size=test_size,
        step_size=step_size,
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

    pre_test_train_frame = formation_frame.iloc[plan.pre_test_train_indices]

    # Locked evaluation: train on all purged pre-eval-period formation rows, score eval_period.
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
        "n_eval_rows": int(eval_frame.shape[0]),
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
            "unless this run's target_column is secondary."
        ),
    }
