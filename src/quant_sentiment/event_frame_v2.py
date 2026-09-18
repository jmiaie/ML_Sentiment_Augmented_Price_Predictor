"""Event-level sample construction for Directive #9 D9-D AUTHORITATIVE (v2).

Spec: "Sample unit: Event-level filing observation. Do not manufacture
zero-sentiment non-event days." v1's ``features.build_modeling_frame``
violated this directly -- it emitted one row per CALENDAR trading day
(with a rolling 5-day sentiment-event lookback, defaulting to zeros on
days with no filing). This module instead emits exactly one row per
unique (CIK, accession) filing, with no synthetic non-event rows at all.

Each row combines: the real Loughran-McDonald text features
(``lm_dictionary.score_text``), the 9 trailing market-only features
(``market_features_v2.trailing_market_features``), and both the primary
(1-session) and secondary (5-session) excess-return targets
(``market_features_v2.excess_return_target``) -- all anchored to the
filing's NYSE-calendar-resolved effective session
(``nyse_calendar.NyseCalendar.resolve_effective_session``).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .lm_dictionary import score_text
from .market_features_v2 import (
    MARKET_FEATURE_COLUMNS_V2,
    SessionPriceSeries,
    build_session_price_series,
    excess_return_target,
    trailing_market_features,
)
from .nyse_calendar import NyseCalendar

LM_FEATURE_COLUMNS = [
    "negative_fraction",
    "positive_fraction",
    "uncertainty_fraction",
    "litigious_fraction",
    "constraining_fraction",
    "net_tone",
    "log_word_count",
]
ALL_FEATURE_COLUMNS = [*MARKET_FEATURE_COLUMNS_V2, *LM_FEATURE_COLUMNS]

SKIP_REASONS = (
    "duplicate_accession",
    "no_price_session",
    "insufficient_market_history",
    "insufficient_forward_window",
    "empty_text",
)


@dataclass(frozen=True)
class FilingEvent:
    cik: str
    ticker: str
    company: str
    accession: str
    form: str
    filing_date: str
    acceptance_datetime: pd.Timestamp
    text: str


def build_event_frame(
    events: list[FilingEvent],
    issuer_price_frames: dict[str, pd.DataFrame],
    spy_price_frame: pd.DataFrame,
    calendar: NyseCalendar,
) -> tuple[pd.DataFrame, dict[str, int]]:
    issuer_series: dict[str, SessionPriceSeries] = {
        ticker: build_session_price_series(frame) for ticker, frame in issuer_price_frames.items()
    }
    spy_series = build_session_price_series(spy_price_frame)

    seen_accessions: set[tuple[str, str]] = set()
    rows: list[dict[str, object]] = []
    skip_counts: dict[str, int] = {reason: 0 for reason in SKIP_REASONS}

    for event in events:
        key = (event.cik, event.accession)
        if key in seen_accessions:
            skip_counts["duplicate_accession"] += 1
            continue
        seen_accessions.add(key)

        if event.ticker not in issuer_series:
            raise KeyError(f"no price series supplied for ticker {event.ticker!r}")
        issuer = issuer_series[event.ticker]

        effective_session = pd.Timestamp(
            calendar.resolve_effective_session(event.acceptance_datetime)
        )
        idx = issuer.index_of(effective_session)
        spy_idx = spy_series.index_of(effective_session)
        if idx is None or spy_idx is None:
            skip_counts["no_price_session"] += 1
            continue

        market_features = trailing_market_features(issuer, spy_series, idx, spy_idx)
        if market_features is None:
            skip_counts["insufficient_market_history"] += 1
            continue

        primary = excess_return_target(issuer, spy_series, idx, spy_idx, 1)
        secondary = excess_return_target(issuer, spy_series, idx, spy_idx, 5)
        if primary is None or secondary is None:
            skip_counts["insufficient_forward_window"] += 1
            continue

        if not event.text.strip():
            skip_counts["empty_text"] += 1
            continue

        lm = score_text(event.text)
        effective_session_date = effective_session.date()

        rows.append(
            {
                "cik": event.cik,
                "ticker": event.ticker,
                "company": event.company,
                "accession": event.accession,
                "form": event.form,
                "filing_date": event.filing_date,
                "acceptance_datetime": event.acceptance_datetime,
                "effective_session": effective_session,
                "effective_session_ordinal": calendar.session_ordinal(effective_session_date),
                **market_features,
                "negative_fraction": lm.negative_fraction,
                "positive_fraction": lm.positive_fraction,
                "uncertainty_fraction": lm.uncertainty_fraction,
                "litigious_fraction": lm.litigious_fraction,
                "constraining_fraction": lm.constraining_fraction,
                "net_tone": lm.net_tone,
                "log_word_count": lm.log_word_count,
                "lm_token_count": lm.token_count,
                "primary_excess_return": primary.excess_return,
                "primary_direction": primary.direction,
                "primary_target_end_session_ordinal": calendar.session_ordinal(
                    primary.target_end_session.date()
                ),
                "secondary_excess_return": secondary.excess_return,
                "secondary_direction": secondary.direction,
                "secondary_target_end_session_ordinal": calendar.session_ordinal(
                    secondary.target_end_session.date()
                ),
            }
        )

    frame = pd.DataFrame.from_records(rows)
    if not frame.empty:
        frame = frame.sort_values(["effective_session_ordinal", "cik", "accession"]).reset_index(
            drop=True
        )
    return frame, skip_counts
