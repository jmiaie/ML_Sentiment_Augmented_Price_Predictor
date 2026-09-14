from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from .schema import ensure_utc_aware

UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class USMarketCalendar:
    holidays: frozenset[date] = field(default_factory=frozenset)
    open_time: time = time(9, 30)
    close_time: time = time(16, 0)

    def is_trading_day(self, day: date) -> bool:
        return day.weekday() < 5 and day not in self.holidays

    def next_trading_day(self, day: date) -> date:
        candidate = day
        while True:
            candidate += timedelta(days=1)
            if self.is_trading_day(candidate):
                return candidate

    def session_open(self, day: date) -> datetime:
        return datetime.combine(day, self.open_time, tzinfo=NEW_YORK).astimezone(UTC)

    def session_close(self, day: date) -> datetime:
        return datetime.combine(day, self.close_time, tzinfo=NEW_YORK).astimezone(UTC)

    def resolve_effective_trading_timestamp(self, availability_timestamp: datetime) -> datetime:
        timestamp_utc = ensure_utc_aware(availability_timestamp)
        local_timestamp = timestamp_utc.astimezone(NEW_YORK)
        local_day = local_timestamp.date()
        local_time = local_timestamp.time()

        if not self.is_trading_day(local_day):
            return self.session_open(self.next_trading_day(local_day))
        if local_time < self.open_time:
            return self.session_open(local_day)
        if local_time >= self.close_time:
            return self.session_open(self.next_trading_day(local_day))
        return timestamp_utc

    def decision_cutoffs(self, sessions: list[date]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "session": sessions,
                "decision_timestamp": [self.session_close(session) for session in sessions],
            }
        )
