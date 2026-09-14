from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

UTC = timezone.utc


def ensure_utc_aware(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Timestamps must be timezone-aware.")
    return timestamp.astimezone(UTC)


@dataclass(frozen=True)
class SentimentEvent:
    source: str
    event_id: str
    entity: str
    symbol: str
    publication_timestamp: datetime
    ingestion_timestamp: datetime
    availability_timestamp: datetime
    timezone: str
    sentiment_score: float
    model_version: str
    confidence: float

    def to_record(self, effective_trading_timestamp: datetime) -> dict[str, object]:
        publication_timestamp = ensure_utc_aware(self.publication_timestamp)
        ingestion_timestamp = ensure_utc_aware(self.ingestion_timestamp)
        availability_timestamp = ensure_utc_aware(self.availability_timestamp)
        effective_timestamp = ensure_utc_aware(effective_trading_timestamp)

        if (
            publication_timestamp > ingestion_timestamp
            or ingestion_timestamp > availability_timestamp
        ):
            raise ValueError(
                "Event timestamps must satisfy publication <= ingestion <= availability."
            )
        if not -1.0 <= self.sentiment_score <= 1.0:
            raise ValueError("Sentiment scores must fall in [-1, 1].")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must fall in [0, 1].")

        record = asdict(self)
        record["publication_timestamp"] = publication_timestamp
        record["ingestion_timestamp"] = ingestion_timestamp
        record["availability_timestamp"] = availability_timestamp
        record["effective_trading_timestamp"] = effective_timestamp
        return record


def events_to_frame(
    events: list[SentimentEvent],
    effective_timestamp_resolver: Callable[[datetime], datetime],
) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(
            columns=[
                "source",
                "event_id",
                "entity",
                "symbol",
                "publication_timestamp",
                "ingestion_timestamp",
                "availability_timestamp",
                "timezone",
                "sentiment_score",
                "model_version",
                "confidence",
                "effective_trading_timestamp",
            ]
        )

    records = [
        event.to_record(effective_timestamp_resolver(event.availability_timestamp))
        for event in events
    ]
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["effective_trading_timestamp", "event_id"])
        .reset_index(drop=True)
    )
