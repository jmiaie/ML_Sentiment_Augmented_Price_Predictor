"""Trailing-only market features and excess-return targets for Directive
#9 D9-D AUTHORITATIVE (v2).

v1's ``features.py`` computed a 4-column, day-level feature set
(``market_return_1d/3d``, ``market_volatility_3d``, ``market_volume_z_3d``)
with no SPY/market-relative context at all, and its labels
(``labels.py``) were the issuer's own RAW direction, not excess return
over SPY. D9-D's spec requires 9 trailing market-only features (issuer
1d/5d/20d return, issuer 20d/60d realized vol, issuer 20d volume z-score,
SPY 1d/5d/20d return, SPY 20d realized vol) and EXCESS-return direction
targets (issuer close-to-close return minus SPY close-to-close return).
This module replaces both, operating on a single shared NYSE session
index (see ``nyse_calendar.NyseCalendar``) so issuer and SPY series stay
aligned.

All features are computed AS OF a given session's own close using only
that session and sessions strictly before it (trailing-only, no future
leakage) -- consistent with the spec's "Trailing only ... No future
leakage" requirement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RETURN_WINDOWS = (1, 5, 20)
REALIZED_VOL_WINDOWS = (20, 60)
VOLUME_ZSCORE_WINDOW = 20
ANNUALIZATION_FACTOR = 252.0

MARKET_FEATURE_COLUMNS_V2 = [
    "issuer_return_1d",
    "issuer_return_5d",
    "issuer_return_20d",
    "issuer_realized_vol_20d",
    "issuer_realized_vol_60d",
    "issuer_volume_zscore_20d",
    "spy_return_1d",
    "spy_return_5d",
    "spy_return_20d",
    "spy_realized_vol_20d",
]


@dataclass(frozen=True)
class SessionPriceSeries:
    """Close/volume indexed by position in a shared, ascending NYSE
    session list (index i's session must be the same trading day across
    an issuer's series and SPY's series for the two to be comparable)."""

    sessions: pd.DatetimeIndex
    close: np.ndarray
    volume: np.ndarray

    def __post_init__(self) -> None:
        if not (len(self.sessions) == len(self.close) == len(self.volume)):
            raise ValueError("sessions/close/volume must have matching length")
        if not self.sessions.is_monotonic_increasing:
            raise ValueError("sessions must be strictly ascending")

    def index_of(self, session: pd.Timestamp) -> int | None:
        loc = self.sessions.get_indexer([session])[0]
        return None if loc == -1 else int(loc)


def build_session_price_series(price_frame: pd.DataFrame) -> SessionPriceSeries:
    """``price_frame`` must have a DatetimeIndex and Close/Volume columns
    (yfinance-style raw CSV). Rows are NOT filtered to a calendar here --
    callers align to real NYSE sessions themselves (e.g. by only querying
    ``index_of`` for sessions the calendar itself says are trading days);
    a raw yfinance frame already only contains real trading days."""
    frame = price_frame.sort_index()
    if "Close" not in frame.columns:
        raise ValueError("price_frame missing Close column")
    volume = (
        frame["Volume"].astype(float).to_numpy()
        if "Volume" in frame.columns
        else np.zeros(len(frame))
    )
    return SessionPriceSeries(
        sessions=pd.DatetimeIndex(frame.index).normalize(),
        close=frame["Close"].astype(float).to_numpy(),
        volume=volume,
    )


def _trailing_return(close: np.ndarray, idx: int, window: int) -> float | None:
    if idx - window < 0:
        return None
    return float(close[idx] / close[idx - window] - 1.0)


def _trailing_realized_vol(close: np.ndarray, idx: int, window: int) -> float | None:
    """Annualized close-to-close log-return volatility over the `window`
    sessions ending at (and including) `idx` -- matches this program's
    house convention elsewhere (D9-A/D9-C: log-return std * sqrt(252))."""
    if idx - window < 0:
        return None
    segment = close[idx - window : idx + 1]
    if np.any(segment <= 0):
        return None
    log_returns = np.diff(np.log(segment))
    if len(log_returns) < 2:
        return None
    return float(np.std(log_returns, ddof=1) * np.sqrt(ANNUALIZATION_FACTOR))


def _trailing_volume_zscore(volume: np.ndarray, idx: int, window: int) -> float | None:
    """Today's volume z-scored against the trailing `window` sessions
    STRICTLY BEFORE today (excludes today's own volume from the
    reference distribution, the standard convention for a z-score
    feature -- otherwise today's own value would shrink its own z-score
    toward zero by construction)."""
    if idx - window < 0:
        return None
    reference = volume[idx - window : idx]
    std = float(np.std(reference, ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return None
    return float((volume[idx] - reference.mean()) / std)


def trailing_market_features(
    issuer: SessionPriceSeries, spy: SessionPriceSeries, idx: int, spy_idx: int
) -> dict[str, float] | None:
    """All 9 spec'd trailing features as of session `idx` (issuer) /
    `spy_idx` (SPY) -- returns None if any required trailing window
    doesn't fit (insufficient history), so callers can skip the event
    rather than silently feeding a partially-missing feature vector."""
    values = {
        "issuer_return_1d": _trailing_return(issuer.close, idx, 1),
        "issuer_return_5d": _trailing_return(issuer.close, idx, 5),
        "issuer_return_20d": _trailing_return(issuer.close, idx, 20),
        "issuer_realized_vol_20d": _trailing_realized_vol(issuer.close, idx, 20),
        "issuer_realized_vol_60d": _trailing_realized_vol(issuer.close, idx, 60),
        "issuer_volume_zscore_20d": _trailing_volume_zscore(
            issuer.volume, idx, VOLUME_ZSCORE_WINDOW
        ),
        "spy_return_1d": _trailing_return(spy.close, spy_idx, 1),
        "spy_return_5d": _trailing_return(spy.close, spy_idx, 5),
        "spy_return_20d": _trailing_return(spy.close, spy_idx, 20),
        "spy_realized_vol_20d": _trailing_realized_vol(spy.close, spy_idx, 20),
    }
    if any(v is None for v in values.values()):
        return None
    return {k: float(v) for k, v in values.items()}  # type: ignore[arg-type]


@dataclass(frozen=True)
class ExcessReturnTarget:
    horizon_sessions: int
    issuer_return: float
    spy_return: float
    excess_return: float
    direction: int
    target_end_session: pd.Timestamp


def excess_return_target(
    issuer: SessionPriceSeries,
    spy: SessionPriceSeries,
    idx: int,
    spy_idx: int,
    horizon_sessions: int,
) -> ExcessReturnTarget | None:
    """Excess-return direction target: issuer close-to-close return minus
    SPY close-to-close return, over `horizon_sessions` forward from
    session `idx`/`spy_idx`. Returns None if the forward window runs past
    either series' end (insufficient forward data)."""
    end_idx = idx + horizon_sessions
    end_spy_idx = spy_idx + horizon_sessions
    if end_idx >= len(issuer.close) or end_spy_idx >= len(spy.close):
        return None
    issuer_return = float(issuer.close[end_idx] / issuer.close[idx] - 1.0)
    spy_return = float(spy.close[end_spy_idx] / spy.close[spy_idx] - 1.0)
    excess = issuer_return - spy_return
    return ExcessReturnTarget(
        horizon_sessions=horizon_sessions,
        issuer_return=issuer_return,
        spy_return=spy_return,
        excess_return=excess,
        direction=int(excess > 0),
        target_end_session=pd.Timestamp(issuer.sessions[end_idx]),
    )
