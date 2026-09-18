import pandas as pd
import pytest

from quant_sentiment.nyse_calendar import NyseCalendar


@pytest.fixture(scope="module")
def calendar() -> NyseCalendar:
    return NyseCalendar(schedule_start="2015-01-01", schedule_end="2026-12-31")


def test_weekend_is_not_a_trading_day(calendar: NyseCalendar) -> None:
    # 2024-03-16 is a Saturday.
    assert not calendar.is_trading_day(pd.Timestamp("2024-03-16").date())


def test_nyse_holiday_is_not_a_trading_day(calendar: NyseCalendar) -> None:
    # 2025-11-27 is Thanksgiving -- an NYSE holiday.
    assert not calendar.is_trading_day(pd.Timestamp("2025-11-27").date())
    # 2024-01-01 New Year's Day.
    assert not calendar.is_trading_day(pd.Timestamp("2024-01-01").date())


def test_early_close_day_has_a_real_early_close_time(calendar: NyseCalendar) -> None:
    # Day after Thanksgiving 2025 is a scheduled NYSE half day (13:00 ET close).
    close = calendar.session_close(pd.Timestamp("2025-11-28").date())
    assert close == pd.Timestamp("2025-11-28T18:00:00", tz="UTC")  # 13:00 ET = 18:00 UTC


def test_normal_day_closes_at_16_00_eastern(calendar: NyseCalendar) -> None:
    # 2024-03-15 is EDT (UTC-4): 16:00 ET = 20:00 UTC.
    close = calendar.session_close(pd.Timestamp("2024-03-15").date())
    assert close == pd.Timestamp("2024-03-15T20:00:00", tz="UTC")


def test_before_close_routes_to_same_session(calendar: NyseCalendar) -> None:
    ts = pd.Timestamp("2024-03-15T19:00:00", tz="UTC")  # 15:00 ET, before 16:00 close
    assert calendar.resolve_effective_session(ts) == pd.Timestamp("2024-03-15").date()


def test_at_or_after_close_routes_to_next_session(calendar: NyseCalendar) -> None:
    at_close = pd.Timestamp("2024-03-15T20:00:00", tz="UTC")
    after_close = pd.Timestamp("2024-03-15T20:30:00", tz="UTC")
    expected_next = pd.Timestamp("2024-03-18").date()  # next trading day (Mon)
    assert calendar.resolve_effective_session(at_close) == expected_next
    assert calendar.resolve_effective_session(after_close) == expected_next


def test_early_close_boundary_is_earlier_than_a_normal_day(calendar: NyseCalendar) -> None:
    before_early_close = pd.Timestamp("2025-11-28T17:30:00", tz="UTC")  # 12:30pm ET
    after_early_close = pd.Timestamp("2025-11-28T18:30:00", tz="UTC")  # 1:30pm ET
    assert (
        calendar.resolve_effective_session(before_early_close) == pd.Timestamp("2025-11-28").date()
    )
    # After the 1pm early close on Fri -> next session is Monday 12/1 (weekend skipped).
    assert (
        calendar.resolve_effective_session(after_early_close) == pd.Timestamp("2025-12-01").date()
    )


def test_weekend_timestamp_routes_to_next_trading_day(calendar: NyseCalendar) -> None:
    ts = pd.Timestamp("2025-11-29T15:00:00", tz="UTC")  # Saturday
    assert calendar.resolve_effective_session(ts) == pd.Timestamp("2025-12-01").date()


def test_holiday_timestamp_routes_to_next_trading_day(calendar: NyseCalendar) -> None:
    ts = pd.Timestamp("2025-11-27T15:00:00", tz="UTC")  # Thanksgiving
    assert calendar.resolve_effective_session(ts) == pd.Timestamp("2025-11-28").date()


def test_naive_timestamp_is_rejected(calendar: NyseCalendar) -> None:
    with pytest.raises(ValueError):
        calendar.resolve_effective_session(pd.Timestamp("2024-03-15T19:00:00"))


def test_timezone_conversion_matches_across_representations(calendar: NyseCalendar) -> None:
    # 15:00 ET expressed directly in America/New_York must route identically
    # to the same instant expressed in UTC.
    ts_ny = pd.Timestamp("2024-03-15T15:00:00", tz="America/New_York")
    ts_utc = ts_ny.tz_convert("UTC")
    assert calendar.resolve_effective_session(ts_ny) == calendar.resolve_effective_session(ts_utc)


def test_session_ordinal_is_consistent_and_monotonic(calendar: NyseCalendar) -> None:
    d1 = pd.Timestamp("2024-03-15").date()
    d2 = calendar.next_trading_day(d1)
    assert calendar.session_ordinal(d2) == calendar.session_ordinal(d1) + 1
    assert calendar.session_at_ordinal(calendar.session_ordinal(d1)) == d1


def test_session_ordinal_rejects_non_trading_day(calendar: NyseCalendar) -> None:
    with pytest.raises(ValueError):
        calendar.session_ordinal(pd.Timestamp("2025-11-27").date())  # Thanksgiving
