from __future__ import annotations

import pandas as pd


def build_direction_labels(market_frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"session", "decision_timestamp", "close"}
    missing_columns = required_columns.difference(market_frame.columns)
    if missing_columns:
        raise ValueError(f"Missing label columns: {sorted(missing_columns)}")

    frame = market_frame.copy().sort_values("session").reset_index(drop=True)
    next_close = frame["close"].shift(-1)
    second_close = frame["close"].shift(-2)

    frame["next_session_return"] = next_close.div(frame["close"]).sub(1.0)
    frame["next_session_direction"] = (frame["next_session_return"] > 0).astype("Int64")
    frame.loc[frame["next_session_return"].isna(), "next_session_direction"] = pd.NA

    frame["next_two_session_return"] = second_close.div(frame["close"]).sub(1.0)
    frame["next_two_session_direction"] = (frame["next_two_session_return"] > 0).astype("Int64")
    frame.loc[frame["next_two_session_return"].isna(), "next_two_session_direction"] = pd.NA

    return frame[
        [
            "session",
            "decision_timestamp",
            "next_session_return",
            "next_session_direction",
            "next_two_session_return",
            "next_two_session_direction",
        ]
    ]
