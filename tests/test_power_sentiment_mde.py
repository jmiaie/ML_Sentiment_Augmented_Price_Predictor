"""Tests for the power / MDE calibration (research/power/sentiment_mde.py).

Offline. Covers the analytic formulas, the AR(1) variance factor, ledger hash
verification of the committed inputs, translation-equivariance of the
production bootstrap (which the planted-shift design relies on), and a smoke
run of the Monte Carlo through the production bootstrap (< 30 s).
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from quant_sentiment.historical_text_study_v2 import _block_bootstrap_delta_ci

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "research" / "power" / "sentiment_mde.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sentiment_mde", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mde = _load()


def test_analytic_formulas() -> None:
    se = mde.se_from_ci(-0.004850583318328377, 0.01080005306668726)
    assert se == pytest.approx(0.015650636385015637 / 3.92)
    assert mde.mde80(se) == pytest.approx(2.8 * se)
    assert mde.project_se(1.0, 197, 4 * 197) == pytest.approx(0.5)
    assert mde.n_for_mde(0.01, 197, 0.005) == pytest.approx(4 * 197)


def test_ar1_variance_factor() -> None:
    assert mde.ar1_variance_factor(197, 0.0) == 1.0
    rho = 0.5
    # Large-n limit is (1 + rho) / (1 - rho) = 3.
    assert mde.ar1_variance_factor(100000, rho) == pytest.approx(3.0, rel=1e-3)
    # Direct covariance-matrix check at small n.
    n = 7
    idx = np.arange(n)
    cov = rho ** np.abs(idx[:, None] - idx[None, :])
    assert mde.ar1_variance_factor(n, rho) == pytest.approx(cov.sum() / n)


def test_noise_has_unit_variance() -> None:
    rng = np.random.default_rng(0)
    for family, rho in (("gauss", 0.0), ("gauss", 0.6), ("t4", 0.0)):
        u = mde.draw_noise(rng, 200000, family, rho)
        assert np.var(u) == pytest.approx(1.0, abs=0.03)
    with pytest.raises(ValueError):
        mde.draw_noise(rng, 10, "t4", 0.3)


def test_inputs_match_committed_headline() -> None:
    inputs = mde.load_inputs()
    head = inputs[("historical_evaluation", "primary")]
    assert head["n"] == 197
    assert head["ci_low"] == pytest.approx(-0.0048506, abs=1e-7)
    assert head["ci_high"] == pytest.approx(0.0108001, abs=1e-7)
    assert head["n_resamples"] == 500 and head["block_size"] == 10
    assert round(head["observed_positive_rate"] * 197) == 91


def test_input_hash_mismatch_fails(tmp_path: Path) -> None:
    for rel in (mde.LEDGER, mde.PERIOD_METRICS):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    shutil.copytree(ROOT / mde.RESULTS_DIR, tmp_path / mde.RESULTS_DIR)
    mde.load_inputs(tmp_path)  # pristine copy verifies
    target = tmp_path / mde.artifact_rel("primary", "historical_evaluation")
    target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="disagrees with ledger"):
        mde.load_inputs(tmp_path)


def test_production_bootstrap_is_translation_equivariant() -> None:
    rng = np.random.default_rng(3)
    d = rng.normal(0.0, 0.05, 197)
    zeros = np.zeros(197)
    a = _block_bootstrap_delta_ci(d, zeros, n_resamples=200, block_size=10, seed=5)
    b = _block_bootstrap_delta_ci(d - 0.01, zeros, n_resamples=200, block_size=10, seed=5)
    assert b["ci_low"] == pytest.approx(a["ci_low"] - 0.01, abs=1e-12)
    assert b["ci_high"] == pytest.approx(a["ci_high"] - 0.01, abs=1e-12)


def test_interpolate_k() -> None:
    assert mde.interpolate_k([0, 1, 2], [0.1, 0.6, 1.0]) == pytest.approx(1.5)
    assert mde.interpolate_k([0, 1], [0.1, 0.5]) is None


def test_translation_monotone() -> None:
    small = mde.translate_mde(0.002)
    big = mde.translate_mde(0.011)
    assert small["expected_log_loss_gain_at_tau"] == pytest.approx(0.002, rel=1e-6)
    assert 0.5 < small["calibrated_forecast_auc"] < big["calibrated_forecast_auc"] < 0.7
    # Small-spread approximation: gain ~ 2 tau^2.
    assert small["tau_rms_probability_deviation"] == pytest.approx(math.sqrt(0.001), rel=0.02)


def test_baseline_skill_rows() -> None:
    rows = mde.baseline_skill(mde.load_inputs())
    head = next(r for r in rows if r["period"] == "historical_evaluation"
                and r["target"] == "primary")
    assert head["market_minus_baseline_log_loss"] == pytest.approx(-2.4189e-5, abs=1e-8)
    assert head["auc_null_se"] == pytest.approx(math.sqrt(198 / (12 * 91 * 106)))


def test_smoke_run_under_30s(tmp_path: Path) -> None:
    start = time.time()
    out_json, out_svg = tmp_path / "smoke.json", tmp_path / "smoke.svg"
    assert mde.main(["--smoke", "--out-json", str(out_json), "--out-svg", str(out_svg)]) == 0
    elapsed = time.time() - start
    assert elapsed < 30
    result = json.loads(out_json.read_text(encoding="utf-8"))
    assert result["mode"] == "smoke"
    cell = result["simulation"][0]
    assert cell["replications"] == 40 and cell["k_grid"] == [0.0, 2.8]
    power = cell["power_ci_high_below_0"]
    assert power[0] <= 0.2 and power[1] >= 0.4
    assert result["headline"]["analytic_mde80"] == pytest.approx(
        2.8 * (0.01080005306668726 + 0.004850583318328377) / 3.92
    )
    assert out_svg.read_text(encoding="utf-8").startswith("<svg")
    # Deterministic: a second smoke run is byte-identical.
    out2 = tmp_path / "smoke2.json"
    mde.main(["--smoke", "--out-json", str(out2)])
    assert out2.read_bytes() == out_json.read_bytes()
