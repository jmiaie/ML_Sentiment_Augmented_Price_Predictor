from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .calendar import USMarketCalendar
from .features import MARKET_FEATURE_COLUMNS, SENTIMENT_FEATURE_COLUMNS, build_modeling_frame
from .modeling import run_ablation_study
from .schema import SentimentEvent, events_to_frame

NEW_YORK = ZoneInfo("America/New_York")
SYNTHETIC_LABEL = "Synthetic methodology validation"
PRIMARY_QUESTION = (
    "Does point-in-time sentiment add incremental predictive information beyond "
    "market-only features for short-horizon asset returns?"
)


def _build_sessions(start: date, count: int, calendar: USMarketCalendar) -> list[date]:
    sessions: list[date] = []
    current = start
    while len(sessions) < count:
        if calendar.is_trading_day(current):
            sessions.append(current)
        current += timedelta(days=1)
    return sessions


def generate_synthetic_inputs(
    n_sessions: int = 140,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendar = USMarketCalendar(holidays=frozenset({date(2024, 7, 4)}))
    sessions = _build_sessions(date(2024, 1, 2), n_sessions, calendar)
    rng = np.random.default_rng(seed)

    closes: list[float] = [100.0]
    volumes: list[float] = [1_000_000.0]
    sentiment_events: list[SentimentEvent] = []

    for index, session in enumerate(sessions[:-1]):
        prior_return = 0.0 if index == 0 else closes[index] / closes[index - 1] - 1.0
        sentiment_signal = 0.8 * np.sin(index / 7.0) + rng.normal(0.0, 0.25)
        next_return = 0.35 * prior_return + 0.020 * sentiment_signal + rng.normal(0.0, 0.004)
        closes.append(closes[index] * (1.0 + next_return))
        volumes.append(1_000_000.0 + 7_500.0 * index + rng.normal(0.0, 15_000.0))

        pre_open = datetime(session.year, session.month, session.day, 8, 15, tzinfo=NEW_YORK)
        intraday = datetime(session.year, session.month, session.day, 14, 0, tzinfo=NEW_YORK)
        base_score = float(np.tanh(sentiment_signal))

        sentiment_events.extend(
            [
                SentimentEvent(
                    source="synthetic_news",
                    event_id=f"{session.isoformat()}-preopen",
                    entity="Synthetic Asset",
                    symbol="SYN",
                    publication_timestamp=pre_open,
                    ingestion_timestamp=pre_open + timedelta(minutes=1),
                    availability_timestamp=pre_open + timedelta(minutes=2),
                    timezone="America/New_York",
                    sentiment_score=base_score,
                    model_version="synthetic-v1",
                    confidence=0.80,
                ),
                SentimentEvent(
                    source="synthetic_news",
                    event_id=f"{session.isoformat()}-intraday",
                    entity="Synthetic Asset",
                    symbol="SYN",
                    publication_timestamp=intraday,
                    ingestion_timestamp=intraday + timedelta(minutes=1),
                    availability_timestamp=intraday + timedelta(minutes=2),
                    timezone="America/New_York",
                    sentiment_score=float(np.clip(base_score + rng.normal(0.0, 0.05), -1.0, 1.0)),
                    model_version="synthetic-v1",
                    confidence=0.90,
                ),
            ]
        )

    market_frame = calendar.decision_cutoffs(sessions)
    market_frame["close"] = closes
    market_frame["volume"] = volumes
    events_frame = events_to_frame(sentiment_events, calendar.resolve_effective_trading_timestamp)
    return market_frame, events_frame


def _ablation_conclusion(results: dict[str, Any]) -> str:
    combined = results["test_metrics"]["combined_logistic"]["log_loss"]
    market_only = results["test_metrics"]["market_only_logistic"]["log_loss"]
    sentiment_only = results["test_metrics"]["sentiment_only_logistic"]["log_loss"]
    if combined < min(market_only, sentiment_only):
        return (
            "In this synthetic methodology validation, the combined model outperformed both "
            "single-source logistic baselines on the held-out test window. This is consistent "
            "with the injected incremental signal and is not historical evidence."
        )
    return (
        "The synthetic methodology harness ran successfully, but the combined model did not "
        "materially outperform both single-source baselines on the held-out window."
    )


def run_synthetic_methodology_validation(output_path: str | Path) -> dict[str, Any]:
    market_frame, events_frame = generate_synthetic_inputs()
    modeling_frame = build_modeling_frame(market_frame, events_frame)
    results = run_ablation_study(
        frame=modeling_frame,
        market_feature_columns=MARKET_FEATURE_COLUMNS,
        sentiment_feature_columns=SENTIMENT_FEATURE_COLUMNS,
    )

    payload: dict[str, Any] = {
        "label": SYNTHETIC_LABEL,
        "question": PRIMARY_QUESTION,
        "dataset_disclosure": (
            "Synthetic data only. Historical results pending reproducible point-in-time dataset."
        ),
        "artifact_generation_mode": "deterministic",
        "inputs": {
            "sessions": int(market_frame.shape[0]),
            "events": int(events_frame.shape[0]),
            "symbol": "SYN",
            "random_seed": 7,
            "market_features": MARKET_FEATURE_COLUMNS,
            "sentiment_features": SENTIMENT_FEATURE_COLUMNS,
            "primary_target": "next_session_direction",
            "secondary_target": "next_two_session_direction",
        },
        "ablation_results": results,
        "ablation_conclusion": _ablation_conclusion(results),
        "uncertainty": (
            "Synthetic validation confirms pipeline behavior only. It does not establish "
            "real-world predictive value, robustness across assets, or economic viability."
        ),
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic sentiment methodology validation.")
    parser.add_argument(
        "--output",
        default="artifacts/synthetic_methodology_validation.json",
        help="Path to the JSON artifact to generate.",
    )
    args = parser.parse_args()
    run_synthetic_methodology_validation(args.output)


if __name__ == "__main__":
    main()
