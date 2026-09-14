from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import build_direction_labels

MARKET_FEATURE_COLUMNS = [
    "market_return_1d",
    "market_return_3d",
    "market_volatility_3d",
    "market_volume_z_3d",
]
SENTIMENT_FEATURE_COLUMNS = [
    "sentiment_event_count_5d",
    "sentiment_mean_5d",
    "sentiment_weighted_mean_5d",
    "sentiment_confidence_mean_5d",
]


def build_market_features(market_frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"session", "decision_timestamp", "close"}
    missing_columns = required_columns.difference(market_frame.columns)
    if missing_columns:
        raise ValueError(f"Missing market columns: {sorted(missing_columns)}")

    frame = market_frame.copy().sort_values("session").reset_index(drop=True)
    frame["volume"] = frame["volume"] if "volume" in frame.columns else 0.0

    daily_return = frame["close"].pct_change()
    rolling_volume_mean = frame["volume"].rolling(window=3, min_periods=1).mean()
    rolling_volume_std = frame["volume"].rolling(window=3, min_periods=2).std(ddof=0)
    safe_volume_std = rolling_volume_std.replace(0.0, np.nan)

    frame["market_return_1d"] = daily_return.fillna(0.0)
    frame["market_return_3d"] = frame["close"].pct_change(periods=3).fillna(0.0)
    frame["market_volatility_3d"] = (
        daily_return.rolling(window=3, min_periods=2).std(ddof=0).fillna(0.0)
    )
    frame["market_volume_z_3d"] = (
        ((frame["volume"] - rolling_volume_mean) / safe_volume_std).fillna(0.0)
    )

    return frame[["session", "decision_timestamp", "close", *MARKET_FEATURE_COLUMNS]]


def build_sentiment_features(
    events_frame: pd.DataFrame,
    decision_timestamps: list[pd.Timestamp],
    lookback_days: int = 5,
) -> pd.DataFrame:
    if events_frame.empty:
        return pd.DataFrame(
            {
                "decision_timestamp": decision_timestamps,
                "sentiment_event_count_5d": [0] * len(decision_timestamps),
                "sentiment_mean_5d": [0.0] * len(decision_timestamps),
                "sentiment_weighted_mean_5d": [0.0] * len(decision_timestamps),
                "sentiment_confidence_mean_5d": [0.0] * len(decision_timestamps),
            }
        )

    required_columns = {
        "effective_trading_timestamp",
        "sentiment_score",
        "confidence",
    }
    missing_columns = required_columns.difference(events_frame.columns)
    if missing_columns:
        raise ValueError(f"Missing sentiment columns: {sorted(missing_columns)}")

    frame = events_frame.copy().sort_values("effective_trading_timestamp").reset_index(drop=True)
    records: list[dict[str, object]] = []

    for decision_timestamp in decision_timestamps:
        cutoff = pd.Timestamp(decision_timestamp)
        window_start = cutoff - pd.Timedelta(days=lookback_days)
        window = frame.loc[
            frame["effective_trading_timestamp"].gt(window_start)
            & frame["effective_trading_timestamp"].le(cutoff)
        ]
        if window.empty:
            records.append(
                {
                    "decision_timestamp": cutoff,
                    "sentiment_event_count_5d": 0,
                    "sentiment_mean_5d": 0.0,
                    "sentiment_weighted_mean_5d": 0.0,
                    "sentiment_confidence_mean_5d": 0.0,
                }
            )
            continue

        confidence_sum = float(window["confidence"].sum())
        weighted_mean = 0.0
        if confidence_sum > 0:
            weighted_mean = float(
                (window["sentiment_score"] * window["confidence"]).sum() / confidence_sum
            )

        records.append(
            {
                "decision_timestamp": cutoff,
                "sentiment_event_count_5d": int(window.shape[0]),
                "sentiment_mean_5d": float(window["sentiment_score"].mean()),
                "sentiment_weighted_mean_5d": weighted_mean,
                "sentiment_confidence_mean_5d": float(window["confidence"].mean()),
            }
        )

    return pd.DataFrame.from_records(records)


def build_modeling_frame(market_frame: pd.DataFrame, events_frame: pd.DataFrame) -> pd.DataFrame:
    market_features = build_market_features(market_frame)
    decision_timestamps = [pd.Timestamp(value) for value in market_features["decision_timestamp"]]
    sentiment_features = build_sentiment_features(events_frame, decision_timestamps)
    labels = build_direction_labels(market_features[["session", "decision_timestamp", "close"]])

    return (
        market_features.merge(sentiment_features, on="decision_timestamp", how="left")
        .merge(labels, on=["session", "decision_timestamp"], how="left")
        .sort_values("session")
        .reset_index(drop=True)
    )
