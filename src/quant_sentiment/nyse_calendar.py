"""Real NYSE trading calendar for Directive #9 D9-D AUTHORITATIVE (v2).

v1's ``calendar.USMarketCalendar`` is weekday-only: ``holidays`` defaults
to an empty ``frozenset`` and every session close is a fixed 16:00 ET,
with no early-close handling at all -- exactly the "no weekday-only
calendar" gap Directive #9's D9-D spec calls out by name. This module
replaces it with ``pandas_market_calendars``'s NYSE calendar, which
carries the actual historical/scheduled holiday list and early-close
days (half days) with their real closing times, so
``resolve_effective_session`` can correctly route a filing's acceptance
timestamp to a trading session per the spec's decision-cutoff rule:

* Published before that session's regular-session close (its real close,
  16:00 ET on a normal day, 13:00 ET on a half day): available as of
  that session's own close.
* Published at/after close, or on a weekend/holiday: advance to the next
  valid exchange session.

Session-close timestamps returned here are tz-aware UTC.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import cast

import pandas as pd
import pandas_market_calendars as mcal

CALENDAR_NAME = "NYSE"


def _to_date(ts: pd.Timestamp) -> date:
    """pandas-stubs types Timestamp.date() as returning Any; this
    documents and narrows that known-true fact (mirrors the
    ``_datetime_index`` cast pattern used elsewhere in this program)."""
    return cast(date, pd.Timestamp(ts).date())


@dataclass(frozen=True)
class NyseCalendar:
    """Thin, cached wrapper around ``pandas_market_calendars``'s NYSE
    calendar, exposing exactly what the point-in-time alignment logic
    needs. ``schedule_start``/``schedule_end`` bound the precomputed
    session-close lookup table (inclusive); querying outside that range
    raises rather than silently returning a wrong session."""

    schedule_start: str = "2010-01-01"
    schedule_end: str = "2027-12-31"

    def __post_init__(self) -> None:
        cal = mcal.get_calendar(CALENDAR_NAME)
        schedule = cal.schedule(start_date=self.schedule_start, end_date=self.schedule_end)
        object.__setattr__(self, "_schedule", schedule)

    @property
    def _sessions(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self._schedule.index)  # type: ignore[attr-defined]

    def is_trading_day(self, day: date) -> bool:
        return pd.Timestamp(day).normalize() in self._sessions

    def session_close(self, day: date) -> pd.Timestamp:
        """The real close for this session -- 16:00 ET on a normal day,
        the actual early-close time on a half day (e.g. 13:00 ET the day
        after Thanksgiving) -- returned as tz-aware UTC."""
        ts = pd.Timestamp(day).normalize()
        if ts not in self._sessions:
            raise ValueError(f"{day} is not an NYSE trading session in this calendar's range")
        row = self._schedule.loc[ts]  # type: ignore[attr-defined]
        return pd.Timestamp(row["market_close"]).tz_convert("UTC")

    def session_ordinal(self, day: date) -> int:
        """This session's 0-based position in the calendar's ordered
        session list -- lets callers do session-count arithmetic (e.g. an
        N-session embargo/purge) as plain integer subtraction instead of
        calendar-day math, which would be wrong across weekends/holidays."""
        ts = pd.Timestamp(day).normalize()
        if ts not in self._sessions:
            raise ValueError(f"{day} is not an NYSE trading session in this calendar's range")
        return int(self._sessions.get_loc(ts))

    def session_at_ordinal(self, ordinal: int) -> date:
        return _to_date(self._sessions[ordinal])

    def next_trading_day(self, day: date) -> date:
        ts = pd.Timestamp(day).normalize()
        later = self._sessions[self._sessions > ts]
        if len(later) == 0:
            raise ValueError(
                f"No NYSE session after {day} within schedule range ending {self.schedule_end}"
            )
        return _to_date(later[0])

    def last_session_on_or_before(self, day: date) -> date:
        """The last NYSE session on or before ``day`` (``day`` itself when it
        already is a session).

        Turns a period's calendar end date into a real session boundary:
        2023-12-31 is a Sunday, so the last 2023 session is 2023-12-29. Used
        to bound forward target windows by the period they close in instead
        of approximating with calendar-day arithmetic.
        """
        ts = pd.Timestamp(day).normalize()
        pos = int(self._sessions.searchsorted(ts, side="right")) - 1
        if pos < 0:
            raise ValueError(
                f"No NYSE session on or before {day} within schedule range "
                f"starting {self.schedule_start}"
            )
        return _to_date(self._sessions[pos])

    def resolve_effective_session(self, acceptance_timestamp: datetime) -> date:
        """Map a filing's SEC EDGAR acceptance timestamp to the NYSE
        trading session as of whose close the filing is considered
        available for a next-session decision, per the spec's
        decision-cutoff rule (before close -> same session; at/after
        close, weekend, or holiday -> next valid session)."""
        ts = pd.Timestamp(acceptance_timestamp)
        if ts.tzinfo is None:
            raise ValueError("acceptance_timestamp must be timezone-aware")
        ts_utc = ts.tz_convert("UTC")
        local_day: date = ts_utc.tz_convert("America/New_York").date()

        if not self.is_trading_day(local_day):
            return self.next_trading_day(local_day)
        close = self.session_close(local_day)
        if ts_utc < close:
            return local_day
        return self.next_trading_day(local_day)
