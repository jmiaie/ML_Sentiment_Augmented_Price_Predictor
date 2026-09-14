from datetime import date, datetime
from zoneinfo import ZoneInfo

from quant_sentiment.calendar import USMarketCalendar

NEW_YORK = ZoneInfo("America/New_York")


def test_after_close_events_roll_to_next_open() -> None:
    calendar = USMarketCalendar()

    effective_timestamp = calendar.resolve_effective_trading_timestamp(
        datetime(2024, 1, 8, 16, 30, tzinfo=NEW_YORK)
    )

    assert effective_timestamp == calendar.session_open(date(2024, 1, 9))


def test_weekend_and_holiday_events_roll_to_next_trading_open() -> None:
    calendar = USMarketCalendar(holidays=frozenset({date(2024, 1, 15)}))

    effective_timestamp = calendar.resolve_effective_trading_timestamp(
        datetime(2024, 1, 13, 12, 0, tzinfo=NEW_YORK)
    )

    assert effective_timestamp == calendar.session_open(date(2024, 1, 16))
