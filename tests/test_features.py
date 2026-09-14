from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.testing import assert_frame_equal

from quant_sentiment.calendar import USMarketCalendar
from quant_sentiment.features import (
    MARKET_FEATURE_COLUMNS,
    SENTIMENT_FEATURE_COLUMNS,
    build_market_features,
    build_modeling_frame,
)
from quant_sentiment.labels import build_direction_labels
from quant_sentiment.schema import SentimentEvent, events_to_frame

NEW_YORK = ZoneInfo("America/New_York")


def _sample_market_frame(calendar: USMarketCalendar) -> pd.DataFrame:
    sessions = [date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10), date(2024, 1, 11)]
    return pd.DataFrame(
        {
            "session": sessions,
            "decision_timestamp": [calendar.session_close(session) for session in sessions],
            "close": [100.0, 101.0, 102.0, 104.0],
            "volume": [1_000_000.0, 1_010_000.0, 1_020_000.0, 1_030_000.0],
        }
    )


def _sample_events(calendar: USMarketCalendar) -> pd.DataFrame:
    events = [
        SentimentEvent(
            source="test",
            event_id="e1",
            entity="Example Corp",
            symbol="EX",
            publication_timestamp=datetime(2024, 1, 8, 8, 0, tzinfo=NEW_YORK),
            ingestion_timestamp=datetime(2024, 1, 8, 8, 1, tzinfo=NEW_YORK),
            availability_timestamp=datetime(2024, 1, 8, 8, 2, tzinfo=NEW_YORK),
            timezone="America/New_York",
            sentiment_score=0.2,
            model_version="v1",
            confidence=0.8,
        ),
        SentimentEvent(
            source="test",
            event_id="e2",
            entity="Example Corp",
            symbol="EX",
            publication_timestamp=datetime(2024, 1, 9, 15, 0, tzinfo=NEW_YORK),
            ingestion_timestamp=datetime(2024, 1, 9, 15, 1, tzinfo=NEW_YORK),
            availability_timestamp=datetime(2024, 1, 9, 15, 2, tzinfo=NEW_YORK),
            timezone="America/New_York",
            sentiment_score=0.4,
            model_version="v1",
            confidence=0.9,
        ),
    ]
    return events_to_frame(events, calendar.resolve_effective_trading_timestamp)


def test_future_events_do_not_change_prior_sentiment_features() -> None:
    calendar = USMarketCalendar()
    market_frame = _sample_market_frame(calendar)
    events_frame = _sample_events(calendar)

    base = build_modeling_frame(market_frame, events_frame)

    future_event = SentimentEvent(
        source="test",
        event_id="future",
        entity="Example Corp",
        symbol="EX",
        publication_timestamp=datetime(2024, 1, 12, 15, 0, tzinfo=NEW_YORK),
        ingestion_timestamp=datetime(2024, 1, 12, 15, 1, tzinfo=NEW_YORK),
        availability_timestamp=datetime(2024, 1, 12, 15, 2, tzinfo=NEW_YORK),
        timezone="America/New_York",
        sentiment_score=1.0,
        model_version="v1",
        confidence=1.0,
    )
    extended_events = pd.concat(
        [
            events_frame,
            events_to_frame([future_event], calendar.resolve_effective_trading_timestamp),
        ],
        ignore_index=True,
    )

    updated = build_modeling_frame(market_frame, extended_events)

    assert_frame_equal(
        base.loc[:, SENTIMENT_FEATURE_COLUMNS],
        updated.loc[:, SENTIMENT_FEATURE_COLUMNS],
    )


def test_future_market_extremes_do_not_change_prior_market_features() -> None:
    calendar = USMarketCalendar()
    market_frame = _sample_market_frame(calendar)
    base = build_market_features(market_frame)

    extended_market_frame = pd.concat(
        [
            market_frame,
            pd.DataFrame(
                {
                    "session": [date(2024, 1, 12)],
                    "decision_timestamp": [calendar.session_close(date(2024, 1, 12))],
                    "close": [500.0],
                    "volume": [9_999_999.0],
                }
            ),
        ],
        ignore_index=True,
    )
    updated = build_market_features(extended_market_frame)

    assert_frame_equal(
        base.loc[:, MARKET_FEATURE_COLUMNS],
        updated.loc[: base.shape[0] - 1, MARKET_FEATURE_COLUMNS].reset_index(drop=True),
    )


def test_labels_start_after_the_feature_cutoff() -> None:
    calendar = USMarketCalendar()
    market_frame = _sample_market_frame(calendar)

    labels = build_direction_labels(market_frame[["session", "decision_timestamp", "close"]])

    expected_next_session_return = market_frame.loc[1, "close"] / market_frame.loc[0, "close"] - 1.0
    assert labels.loc[0, "decision_timestamp"] == calendar.session_close(date(2024, 1, 8))
    assert labels.loc[0, "next_session_return"] == expected_next_session_return
    assert pd.isna(labels.loc[3, "next_session_direction"])
