import numpy as np
import pandas as pd

from quant_sentiment.event_frame_v2 import ALL_FEATURE_COLUMNS, FilingEvent, build_event_frame
from quant_sentiment.nyse_calendar import NyseCalendar


def _price_frame(sessions: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(sessions))))
    volume = rng.integers(1_000_000, 5_000_000, len(sessions)).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=sessions.tz_localize(None))


def _calendar() -> NyseCalendar:
    return NyseCalendar(schedule_start="2018-01-01", schedule_end="2020-12-31")


def _sessions(cal: NyseCalendar) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(cal._schedule.index)  # type: ignore[attr-defined]


def test_one_row_per_filing_not_per_calendar_day() -> None:
    """Directly tests the spec's 'do not manufacture zero-sentiment
    non-event days' requirement: with 3 filings, the event frame must
    have exactly 3 rows, not one row per trading day in the period."""
    cal = _calendar()
    sessions = _sessions(cal)
    issuer_frame = _price_frame(sessions, seed=1)
    spy_frame = _price_frame(sessions, seed=2)

    accept_ts = [sessions[100], sessions[150], sessions[200]]
    events = [
        FilingEvent(
            cik="0001",
            ticker="AAA",
            company="AAA Inc.",
            accession=f"acc-{i}",
            form="8-K",
            filing_date=str(ts.date()),
            acceptance_datetime=ts.tz_localize("UTC") + pd.Timedelta(hours=15),
            text="growth improvement strong record " * 10,
        )
        for i, ts in enumerate(accept_ts)
    ]

    frame, skips = build_event_frame(events, {"AAA": issuer_frame}, spy_frame, cal)
    assert len(frame) == 3
    assert sum(skips.values()) == 0


def test_duplicate_accession_is_deduplicated() -> None:
    cal = _calendar()
    sessions = _sessions(cal)
    issuer_frame = _price_frame(sessions, seed=3)
    spy_frame = _price_frame(sessions, seed=4)

    ts = sessions[100].tz_localize("UTC") + pd.Timedelta(hours=15)
    events = [
        FilingEvent(
            cik="0001",
            ticker="AAA",
            company="AAA Inc.",
            accession="acc-dup",
            form="8-K",
            filing_date=str(sessions[100].date()),
            acceptance_datetime=ts,
            text="growth improvement " * 10,
        )
        for _ in range(3)
    ]
    frame, skips = build_event_frame(events, {"AAA": issuer_frame}, spy_frame, cal)
    assert len(frame) == 1
    assert skips["duplicate_accession"] == 2


def test_insufficient_history_and_forward_window_are_skipped_and_counted() -> None:
    cal = _calendar()
    sessions = _sessions(cal)
    issuer_frame = _price_frame(sessions, seed=5)
    spy_frame = _price_frame(sessions, seed=6)

    too_early = sessions[10]  # short of the 60-session realized-vol window
    too_late = sessions[-2]  # short of the 5-session forward window
    events = [
        FilingEvent(
            cik="0001",
            ticker="AAA",
            company="AAA Inc.",
            accession="acc-early",
            form="8-K",
            filing_date=str(too_early.date()),
            acceptance_datetime=too_early.tz_localize("UTC") + pd.Timedelta(hours=15),
            text="growth " * 10,
        ),
        FilingEvent(
            cik="0001",
            ticker="AAA",
            company="AAA Inc.",
            accession="acc-late",
            form="8-K",
            filing_date=str(too_late.date()),
            acceptance_datetime=too_late.tz_localize("UTC") + pd.Timedelta(hours=15),
            text="growth " * 10,
        ),
    ]
    frame, skips = build_event_frame(events, {"AAA": issuer_frame}, spy_frame, cal)
    assert frame.empty
    assert skips["insufficient_market_history"] == 1
    assert skips["insufficient_forward_window"] == 1


def test_empty_text_is_skipped() -> None:
    cal = _calendar()
    sessions = _sessions(cal)
    issuer_frame = _price_frame(sessions, seed=7)
    spy_frame = _price_frame(sessions, seed=8)

    ts = sessions[100]
    events = [
        FilingEvent(
            cik="0001",
            ticker="AAA",
            company="AAA Inc.",
            accession="acc-empty",
            form="8-K",
            filing_date=str(ts.date()),
            acceptance_datetime=ts.tz_localize("UTC") + pd.Timedelta(hours=15),
            text="   ",
        )
    ]
    frame, skips = build_event_frame(events, {"AAA": issuer_frame}, spy_frame, cal)
    assert frame.empty
    assert skips["empty_text"] == 1


def test_frame_carries_all_feature_columns_and_both_targets() -> None:
    cal = _calendar()
    sessions = _sessions(cal)
    issuer_frame = _price_frame(sessions, seed=9)
    spy_frame = _price_frame(sessions, seed=10)
    ts = sessions[100]
    events = [
        FilingEvent(
            cik="0001",
            ticker="AAA",
            company="AAA Inc.",
            accession="acc-1",
            form="10-K",
            filing_date=str(ts.date()),
            acceptance_datetime=ts.tz_localize("UTC") + pd.Timedelta(hours=15),
            text="growth improvement litigation uncertainty " * 20,
        )
    ]
    frame, _ = build_event_frame(events, {"AAA": issuer_frame}, spy_frame, cal)
    assert len(frame) == 1
    for column in ALL_FEATURE_COLUMNS:
        assert column in frame.columns
    assert "primary_direction" in frame.columns
    assert "secondary_direction" in frame.columns
    assert frame.loc[0, "primary_direction"] in (0, 1)
    assert frame.loc[0, "secondary_direction"] in (0, 1)


def test_rows_sorted_by_effective_session_ordinal_ascending() -> None:
    cal = _calendar()
    sessions = _sessions(cal)
    issuer_frame = _price_frame(sessions, seed=11)
    spy_frame = _price_frame(sessions, seed=12)

    events = [
        FilingEvent(
            cik="0001",
            ticker="AAA",
            company="AAA Inc.",
            accession=f"acc-{i}",
            form="8-K",
            filing_date=str(sessions[k].date()),
            acceptance_datetime=sessions[k].tz_localize("UTC") + pd.Timedelta(hours=15),
            text="growth " * 10,
        )
        for i, k in enumerate([200, 100, 150])
    ]
    frame, _ = build_event_frame(events, {"AAA": issuer_frame}, spy_frame, cal)
    assert frame["effective_session_ordinal"].is_monotonic_increasing
