"""Offline tests for Directive #9 historical text study helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from quant_sentiment.calendar import USMarketCalendar
from quant_sentiment.edgar_text import html_to_plain_text
from quant_sentiment.features import build_modeling_frame
from quant_sentiment.historical_text_study import PeriodSpec, run_period_ablation
from quant_sentiment.schema import SentimentEvent, events_to_frame


def test_html_to_plain_text_strips_tags() -> None:
    raw = (
        chr(60) + "html" + chr(62)
        + chr(60) + "body" + chr(62)
        + chr(60) + "p" + chr(62) + "Hello "
        + chr(60) + "b" + chr(62) + "world" + chr(60) + "/b" + chr(62)
        + chr(60) + "/p" + chr(62)
        + chr(60) + "/body" + chr(62)
        + chr(60) + "/html" + chr(62)
    )
    text = html_to_plain_text(raw)
    assert "Hello" in text and "world" in text
    assert chr(60) not in text


def test_period_spec_contains() -> None:
    period = PeriodSpec("validation", "2024-01-01", "2024-12-31")
    assert period.contains_session(date(2024, 6, 15))
    assert not period.contains_session(date(2025, 1, 2))


def test_holdout_blocked_without_flag() -> None:
    calendar = USMarketCalendar()
    sessions = pd.bdate_range("2015-01-02", periods=200).date
    market = pd.DataFrame(
        {
            "session": list(sessions),
            "decision_timestamp": [calendar.session_close(s) for s in sessions],
            "close": [100.0 + i * 0.1 for i in range(len(sessions))],
            "volume": [1_000_000.0] * len(sessions),
        }
    )
    events = events_to_frame([], calendar.resolve_effective_trading_timestamp)
    frame = build_modeling_frame(market, events)
    formation = PeriodSpec("formation_dev", "2015-01-01", "2023-12-31")
    holdout = PeriodSpec("holdout", "2025-01-01", "2025-12-31")
    with pytest.raises(RuntimeError, match="Holdout evaluation blocked"):
        run_period_ablation(
            frame,
            formation=formation,
            eval_period=holdout,
            config={
                "walk_forward": {
                    "initial_train_size": 40,
                    "validation_size": 10,
                    "formation_internal_test_size": 10,
                    "step_size": 10,
                    "label_horizon": 1,
                    "embargo": 0,
                    "c_values": [1.0],
                },
                "target_column": "next_session_direction",
            },
            allow_holdout=False,
        )


def test_run_period_ablation_formation(tmp_path: Path) -> None:
    calendar = USMarketCalendar()
    # Long enough formation window
    sessions = pd.bdate_range("2015-01-02", periods=400).date
    closes = [100.0]
    for i in range(1, len(sessions)):
        closes.append(closes[-1] * (1.0 + (0.001 if i % 3 else -0.0005)))
    market = pd.DataFrame(
        {
            "session": list(sessions),
            "decision_timestamp": [calendar.session_close(s) for s in sessions],
            "close": closes,
            "volume": [1_000_000.0 + 1000 * i for i in range(len(sessions))],
        }
    )
    # Sprinkle a few synthetic-structured events (test fixture only — not D9 historical claim)
    events_list = []
    for i, session in enumerate(sessions[::20]):
        ts = datetime(session.year, session.month, session.day, 14, 0, tzinfo=timezone.utc)
        events_list.append(
            SentimentEvent(
                source="fixture",
                event_id=f"e-{i}",
                entity="TEST",
                symbol="TEST",
                publication_timestamp=ts,
                ingestion_timestamp=ts,
                availability_timestamp=ts,
                timezone="UTC",
                sentiment_score=0.2 if i % 2 == 0 else -0.1,
                model_version="fixture-v1",
                confidence=0.5,
            )
        )
    events = events_to_frame(events_list, calendar.resolve_effective_trading_timestamp)
    frame = build_modeling_frame(market, events)
    formation = PeriodSpec("formation_dev", "2015-01-01", "2016-06-30")
    result = run_period_ablation(
        frame,
        formation=formation,
        eval_period=formation,
        config={
            "walk_forward": {
                "initial_train_size": 60,
                "validation_size": 20,
                "formation_internal_test_size": 20,
                "step_size": 20,
                "label_horizon": 1,
                "embargo": 0,
                "c_values": [0.1, 1.0],
            },
            "target_column": "next_session_direction",
        },
        allow_holdout=False,
    )
    assert "key_metrics" in result
    assert result["key_metrics"]["n_eval_rows"] > 0
    assert "combined_log_loss" in result["key_metrics"]
