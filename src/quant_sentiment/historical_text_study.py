"""Directive #9 historical SEC filing + price sentiment validation helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .calendar import USMarketCalendar
from .features import MARKET_FEATURE_COLUMNS, SENTIMENT_FEATURE_COLUMNS, build_modeling_frame
from .lexicon import LEXICON_ID, LEXICON_VERSION
from .modeling import (
    _classification_metrics,
    _fit_probabilities,
    _majority_validation_metrics,
    _tune_logistic_model,
)
from .schema import SentimentEvent, events_to_frame
from .validation import build_walk_forward_plan

NEW_YORK = ZoneInfo("America/New_York")
UTC = timezone.utc


@dataclass(frozen=True)
class PeriodSpec:
    name: str
    start: str
    end_inclusive: str

    def contains_session(self, session: date | pd.Timestamp) -> bool:
        start = date.fromisoformat(self.start)
        end = date.fromisoformat(self.end_inclusive)
        if isinstance(session, pd.Timestamp):
            day = session.date()
        else:
            day = session
        return bool((day - start).days >= 0 and (end - day).days >= 0)


def write_json_artifact(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def load_price_frame(raw_dir: Path, symbol: str, calendar: USMarketCalendar) -> pd.DataFrame:
    path = raw_dir / "prices" / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing price file: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "Close" not in df.columns:
        raise ValueError(f"{path} missing Close column")
    if "Volume" in df.columns:
        volume_series = df["Volume"].astype(float)
    else:
        volume_series = pd.Series(0.0, index=df.index, dtype=float)
    sessions: list[date] = []
    closes: list[float] = []
    volumes: list[float] = []
    close_series = df["Close"].astype(float)
    for ts, close, vol in zip(df.index, close_series, volume_series, strict=True):
        day = pd.Timestamp(ts).date()
        if not calendar.is_trading_day(day):
            continue
        if pd.isna(close):
            continue
        sessions.append(day)
        closes.append(float(close))
        volumes.append(float(vol) if not pd.isna(vol) else 0.0)
    return pd.DataFrame(
        {
            "session": sessions,
            "decision_timestamp": [calendar.session_close(session) for session in sessions],
            "close": closes,
            "volume": volumes,
        }
    )


def load_events_frame(raw_dir: Path, symbol: str, calendar: USMarketCalendar) -> pd.DataFrame:
    path = raw_dir / "events" / f"{symbol}_sentiment_events.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing events file: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        return events_to_frame([], calendar.resolve_effective_trading_timestamp)

    events: list[SentimentEvent] = []
    for row in frame.itertuples(index=False):
        availability = pd.Timestamp(row.availability_timestamp)
        if availability.tzinfo is None:
            availability = availability.tz_localize("UTC")
        publication = pd.Timestamp(row.publication_timestamp)
        if publication.tzinfo is None:
            publication = publication.tz_localize("UTC")
        events.append(
            SentimentEvent(
                source=str(row.source),
                event_id=str(row.event_id),
                entity=str(row.entity),
                symbol=str(row.symbol),
                publication_timestamp=publication.to_pydatetime(),
                ingestion_timestamp=publication.to_pydatetime(),
                availability_timestamp=availability.to_pydatetime(),
                timezone="UTC",
                sentiment_score=float(row.sentiment_score),
                model_version=str(row.model_version),
                confidence=float(row.confidence),
            )
        )
    return events_to_frame(events, calendar.resolve_effective_trading_timestamp)


def build_symbol_modeling_frame(
    raw_dir: Path,
    symbol: str,
    *,
    holidays: frozenset[date] | None = None,
) -> pd.DataFrame:
    calendar = USMarketCalendar(holidays=holidays or frozenset())
    market = load_price_frame(raw_dir, symbol, calendar)
    events = load_events_frame(raw_dir, symbol, calendar)
    frame = build_modeling_frame(market, events)
    frame["symbol"] = symbol
    return frame


def slice_period(frame: pd.DataFrame, period: PeriodSpec) -> pd.DataFrame:
    mask = frame["session"].map(lambda s: period.contains_session(s))
    return frame.loc[mask].reset_index(drop=True)



def _slim_fold_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep fold-count + mean scalar metrics; drop per-fold calibration tables."""
    if not metrics:
        return {"fold_count": 0}
    scalar_keys = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "log_loss",
        "brier_score",
        "roc_auc",
        "samples",
    ]
    summary: dict[str, Any] = {"fold_count": len(metrics)}
    for key in scalar_keys:
        values = [m[key] for m in metrics if m.get(key) is not None]
        summary[f"mean_{key}"] = float(sum(values) / len(values)) if values else None
    return summary


def run_period_ablation(
    full_frame: pd.DataFrame,
    *,
    formation: PeriodSpec,
    eval_period: PeriodSpec,
    config: dict[str, Any],
    allow_holdout: bool = False,
) -> dict[str, Any]:
    """Tune on formation walk-forward; evaluate once on eval_period with locked C."""
    if eval_period.name == "holdout" and not allow_holdout:
        raise RuntimeError(
            "Holdout evaluation blocked until FINAL CONFIGURATION FROZEN "
            "(pass allow_holdout=True only after freeze)."
        )

    wf = config.get("walk_forward", {})
    initial_train_size = int(wf.get("initial_train_size", 60))
    validation_size = int(wf.get("validation_size", 20))
    step_size = int(wf.get("step_size", 20))
    label_horizon = int(wf.get("label_horizon", 1))
    embargo = int(wf.get("embargo", 0))
    c_values = tuple(float(x) for x in wf.get("c_values", [0.1, 1.0, 10.0]))
    target_column = str(config.get("target_column", "next_session_direction"))

    formation_frame = slice_period(full_frame, formation).dropna(subset=[target_column])
    formation_frame = formation_frame.reset_index(drop=True)
    eval_frame = slice_period(full_frame, eval_period).dropna(subset=[target_column])
    eval_frame = eval_frame.reset_index(drop=True)

    if formation_frame.empty:
        raise ValueError("Empty formation frame after dropna")
    if eval_frame.empty:
        raise ValueError(f"Empty evaluation frame for period {eval_period.name}")

    # Walk-forward plan on formation only (for C selection). Use remaining
    # formation tail as an internal test probe when large enough; otherwise
    # set test_size to a minimal positive value that still fits.
    n_form = formation_frame.shape[0]
    test_size = int(wf.get("formation_internal_test_size", 20))
    gap_pad = max(label_horizon - 1, 0) + embargo
    min_needed = initial_train_size + validation_size + test_size + gap_pad
    if not (n_form > min_needed):
        test_size = max(5, n_form // 10)
        if not (n_form > initial_train_size + validation_size + test_size):
            raise ValueError(
                f"Formation too small for walk-forward "
                f"(n={n_form}, need > {initial_train_size + validation_size + test_size})"
            )

    plan = build_walk_forward_plan(
        n_samples=n_form,
        initial_train_size=initial_train_size,
        validation_size=validation_size,
        test_size=test_size,
        step_size=step_size,
        label_horizon=label_horizon,
        embargo=embargo,
    )

    model_specs = {
        "majority_baseline": [],
        "market_only_logistic": MARKET_FEATURE_COLUMNS,
        "sentiment_only_logistic": SENTIMENT_FEATURE_COLUMNS,
        "combined_logistic": [*MARKET_FEATURE_COLUMNS, *SENTIMENT_FEATURE_COLUMNS],
    }

    selected_regularization: dict[str, float | None] = {}
    validation_metrics: dict[str, list[dict[str, Any]]] = {}
    for model_name, feature_columns in model_specs.items():
        if not feature_columns:
            selected_regularization[model_name] = None
            validation_metrics[model_name] = _majority_validation_metrics(
                frame=formation_frame,
                splits=plan.validation_splits,
                target_column=target_column,
            )
            continue
        selection = _tune_logistic_model(
            frame=formation_frame,
            splits=plan.validation_splits,
            feature_columns=feature_columns,
            target_column=target_column,
            c_values=c_values,
        )
        selected_regularization[model_name] = selection.best_c
        validation_metrics[model_name] = selection.validation_metrics

    # Locked evaluation: train on all formation rows, score eval_period.
    period_metrics: dict[str, dict[str, Any]] = {}
    for model_name, feature_columns in model_specs.items():
        probabilities = _fit_probabilities(
            train_frame=formation_frame,
            evaluation_frame=eval_frame,
            feature_columns=feature_columns,
            target_column=target_column,
            c_value=selected_regularization[model_name],
        )
        period_metrics[model_name] = _classification_metrics(
            eval_frame[target_column].astype(int).to_numpy(),
            probabilities,
        )

    combined = period_metrics["combined_logistic"]
    market_only = period_metrics["market_only_logistic"]
    key_metrics = {
        "n_formation_rows": int(formation_frame.shape[0]),
        "n_eval_rows": int(eval_frame.shape[0]),
        "n_sentiment_events_in_frame": int(
            (full_frame["sentiment_event_count_5d"].fillna(0) > 0).sum()
        ),
        "eval_start_session": str(eval_frame["session"].iloc[0]),
        "eval_end_session": str(eval_frame["session"].iloc[-1]),
        "majority_log_loss": period_metrics["majority_baseline"]["log_loss"],
        "majority_accuracy": period_metrics["majority_baseline"]["accuracy"],
        "market_only_log_loss": market_only["log_loss"],
        "market_only_accuracy": market_only["accuracy"],
        "market_only_roc_auc": market_only["roc_auc"],
        "sentiment_only_log_loss": period_metrics["sentiment_only_logistic"]["log_loss"],
        "sentiment_only_accuracy": period_metrics["sentiment_only_logistic"]["accuracy"],
        "sentiment_only_roc_auc": period_metrics["sentiment_only_logistic"]["roc_auc"],
        "combined_log_loss": combined["log_loss"],
        "combined_accuracy": combined["accuracy"],
        "combined_roc_auc": combined["roc_auc"],
        "combined_brier_score": combined["brier_score"],
        "delta_log_loss_combined_minus_market": float(
            combined["log_loss"] - market_only["log_loss"]
        ),
        "selected_regularization": selected_regularization,
        "lexicon_id": LEXICON_ID,
        "lexicon_version": LEXICON_VERSION,
        "walk_forward_gap": plan.gap,
        "walk_forward_fold_count": len(plan.validation_splits),
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
        "market_feature_columns": MARKET_FEATURE_COLUMNS,
        "sentiment_feature_columns": SENTIMENT_FEATURE_COLUMNS,
        "target_column": target_column,
        "selected_regularization": selected_regularization,
        "formation_walk_forward_validation_metrics_summary": {
            name: _slim_fold_metrics(folds) for name, folds in validation_metrics.items()
        },
        "period_metrics": period_metrics,
        "key_metrics": key_metrics,
        "notes": (
            "Historical EDGAR 8-K subset + yfinance prices. Lexicon polarity scores "
            "are deterministic and not LM-complete. No invented filing text or labels. "
            "C selected on formation walk-forward only; eval uses locked C."
        ),
    }
