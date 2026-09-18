import numpy as np
import pandas as pd
import pytest

from quant_sentiment.market_features_v2 import (
    MARKET_FEATURE_COLUMNS_V2,
    build_session_price_series,
    excess_return_target,
    trailing_market_features,
)


def _series(n: int = 200, seed: int = 0, drift: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=idx)


def test_insufficient_history_returns_none() -> None:
    issuer = build_session_price_series(_series(seed=1))
    spy = build_session_price_series(_series(seed=2))
    # idx=10 is well short of the 60-session realized-vol window.
    assert trailing_market_features(issuer, spy, 10, 10) is None


def test_all_nine_columns_present_when_sufficient_history() -> None:
    issuer = build_session_price_series(_series(seed=1))
    spy = build_session_price_series(_series(seed=2))
    features = trailing_market_features(issuer, spy, 100, 100)
    assert features is not None
    assert set(features) == set(MARKET_FEATURE_COLUMNS_V2)
    assert all(np.isfinite(v) for v in features.values())


def test_trailing_return_is_causal_no_future_leakage() -> None:
    """Changing prices strictly AFTER idx must not change idx's own
    trailing features -- direct evidence of no future leakage."""
    base = _series(seed=3)
    perturbed = base.copy()
    perturbed.iloc[150:] = perturbed.iloc[150:] * 1.5  # large shock, strictly after idx=100

    spy = build_session_price_series(_series(seed=4))
    issuer_base = build_session_price_series(base)
    issuer_perturbed = build_session_price_series(perturbed)

    features_base = trailing_market_features(issuer_base, spy, 100, 100)
    features_perturbed = trailing_market_features(issuer_perturbed, spy, 100, 100)
    assert features_base == features_perturbed


def test_volume_zscore_excludes_todays_own_volume() -> None:
    """A z-score window that included today's own extreme volume would
    shrink its own z-score toward a smaller magnitude; excluding it (the
    documented convention) keeps today's spike fully visible."""
    frame = _series(seed=5)
    frame = frame.copy()
    frame.iloc[99, frame.columns.get_loc("Volume")] = 50_000_000.0  # huge spike at idx=99
    issuer = build_session_price_series(frame)
    spy = build_session_price_series(_series(seed=6))
    features = trailing_market_features(issuer, spy, 99, 99)
    assert features is not None
    assert features["issuer_volume_zscore_20d"] > 3.0


def test_excess_return_target_is_issuer_minus_spy() -> None:
    idx = pd.bdate_range("2020-01-02", periods=40)
    issuer_close = np.full(40, 100.0)
    issuer_close[11] = 110.0  # +10% on day 11
    spy_close = np.full(40, 100.0)
    spy_close[11] = 103.0  # +3% on day 11

    issuer = build_session_price_series(pd.DataFrame({"Close": issuer_close}, index=idx))
    spy = build_session_price_series(pd.DataFrame({"Close": spy_close}, index=idx))

    target = excess_return_target(issuer, spy, 10, 10, 1)
    assert target is not None
    assert target.issuer_return == pytest.approx(0.10)
    assert target.spy_return == pytest.approx(0.03)
    assert target.excess_return == pytest.approx(0.07)
    assert target.direction == 1


def test_excess_return_target_none_when_forward_window_exceeds_series() -> None:
    issuer = build_session_price_series(_series(n=50, seed=7))
    spy = build_session_price_series(_series(n=50, seed=8))
    assert excess_return_target(issuer, spy, 47, 47, 5) is None


def test_primary_and_secondary_horizons_differ() -> None:
    issuer = build_session_price_series(_series(n=200, seed=9))
    spy = build_session_price_series(_series(n=200, seed=10))
    primary = excess_return_target(issuer, spy, 100, 100, 1)
    secondary = excess_return_target(issuer, spy, 100, 100, 5)
    assert primary is not None and secondary is not None
    assert primary.horizon_sessions == 1
    assert secondary.horizon_sessions == 5
    assert primary.target_end_session != secondary.target_end_session
