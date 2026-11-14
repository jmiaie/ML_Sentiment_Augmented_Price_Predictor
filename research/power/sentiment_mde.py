#!/usr/bin/env python3
"""Power / minimum-detectable-effect (MDE) calibration of the filing-text comparison.

Implements the frozen design in research/power/PROTOCOL.md:

* analytic MDE from the committed block-bootstrap intervals (headline);
* projection of the MDE over n under SE proportional to 1/sqrt(n);
* a synthetic Monte Carlo check that runs planted shifts through the
  production bootstrap ``_block_bootstrap_delta_ci``;
* market-only vs majority-baseline skill from the committed tables;
* an illustrative translation of the MDE into calibrated-forecast units.

Offline and deterministic. Reads committed artifacts only; writes only the
paths given on the command line.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from quant_sentiment.historical_text_study_v2 import _block_bootstrap_delta_ci

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = "results/historical_text_v2"
LEDGER = "research/experiment-ledger.csv"
PERIOD_METRICS = "publication/sentiment-study/tables/period-metrics.csv"
PERIODS = ("formation_dev", "validation", "historical_evaluation")
TARGETS = ("primary", "secondary")
HEADLINE = ("historical_evaluation", "primary")

Z_95 = 1.96
Z_80 = 0.84
MDE_MULTIPLIER = Z_95 + Z_80  # 2.80
CI_WIDTH_IN_SE = 2 * Z_95  # 3.92

N_GRID = (197, 500, 1000, 5000, 20000)
SHARE_TARGETS = (0.01, 0.005)
K_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.8, 3.0, 3.5, 4.0)
POWER_TARGET = 0.80
MASTER_SEED = 20261020
N_RESAMPLES = 500
BLOCK_SIZE = 10

# (name, noise family, rho, n, replications) -- order fixes the seed cell index.
FULL_CELLS: tuple[tuple[str, str, float, int, int], ...] = (
    ("gauss_rho0", "gauss", 0.0, 197, 2000),
    ("gauss_rho0", "gauss", 0.0, 500, 1000),
    ("gauss_rho0", "gauss", 0.0, 1000, 1000),
    ("gauss_rho0.3", "gauss", 0.3, 197, 2000),
    ("gauss_rho0.6", "gauss", 0.6, 197, 2000),
    ("t4_rho0", "t4", 0.0, 197, 2000),
)
SMOKE_CELLS: tuple[tuple[str, str, float, int, int], ...] = (
    ("gauss_rho0", "gauss", 0.0, 197, 40),
)
SMOKE_K_GRID = (0.0, 2.8)


# ---------------------------------------------------------------- inputs ----


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_rel(target: str, period: str) -> str:
    return f"{RESULTS_DIR}/sentiment_hist_text_v2_{target}_{period}.json"


def load_inputs(root: Path = ROOT) -> dict[tuple[str, str], dict[str, Any]]:
    """Committed per-(period, target) inputs, hash-verified against the ledger."""
    ledger: dict[str, str] = {}
    with (root / LEDGER).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ledger[row["experiment_id"]] = row["artifact_sha256"]
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for period in PERIODS:
        for target in TARGETS:
            rel = artifact_rel(target, period)
            digest = sha256_file(root / rel)
            exp = f"sentiment_hist_text_v2_{target}_{period}"
            if ledger.get(exp) != digest:
                raise RuntimeError(f"{rel}: sha256 {digest} disagrees with ledger {exp}")
            doc = json.loads((root / rel).read_text(encoding="utf-8"))
            km = doc["result"]["key_metrics"]
            pm = doc["result"]["period_metrics"]
            ci = km["headline_bootstrap_delta_log_loss_ci95"]
            base_cal = pm["model0_majority_baseline"]["calibration_table"]
            n = int(km["n_eval_rows"])
            pos_rate = sum(b["observed_positive_rate"] * b["count"] for b in base_cal) / n
            out[(period, target)] = {
                "artifact": rel,
                "sha256": digest,
                "n": n,
                "ci_low": float(ci["ci_low"]),
                "ci_high": float(ci["ci_high"]),
                "mean_delta": float(ci["mean_delta"]),
                "n_resamples": int(ci["n_resamples"]),
                "block_size": int(ci["block_size"]),
                "baseline_log_loss": float(pm["model0_majority_baseline"]["log_loss"]),
                "market_log_loss": float(pm["model1_market_only"]["log_loss"]),
                "observed_positive_rate": float(pos_rate),
            }
    return out


# -------------------------------------------------------------- analytic ----


def se_from_ci(ci_low: float, ci_high: float) -> float:
    return (ci_high - ci_low) / CI_WIDTH_IN_SE


def mde80(se: float) -> float:
    return MDE_MULTIPLIER * se


def project_se(se_ref: float, n_ref: int, n: int) -> float:
    return se_ref * math.sqrt(n_ref / n)


def n_for_mde(mde_ref: float, n_ref: int, target: float) -> float:
    return n_ref * (mde_ref / target) ** 2


def analytic_table(inputs: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for (period, target), inp in inputs.items():
        se = se_from_ci(inp["ci_low"], inp["ci_high"])
        mde = mde80(se)
        rows.append(
            {
                "period": period,
                "target": target,
                "n": inp["n"],
                "ci_low": inp["ci_low"],
                "ci_high": inp["ci_high"],
                "mean_delta": inp["mean_delta"],
                "se_hat": se,
                "se_hat_times_sqrt_n": se * math.sqrt(inp["n"]),
                "mde80": mde,
                "baseline_log_loss": inp["baseline_log_loss"],
                "mde80_share_of_baseline": mde / inp["baseline_log_loss"],
                "headline": (period, target) == HEADLINE,
            }
        )
    return rows


def projection(se_ref: float, n_ref: int, baseline: float) -> dict[str, Any]:
    mde_ref = mde80(se_ref)
    grid = []
    for n in N_GRID:
        se = project_se(se_ref, n_ref, n)
        grid.append(
            {
                "n": n,
                "se": se,
                "mde80": mde80(se),
                "mde80_share_of_baseline": mde80(se) / baseline,
            }
        )
    needed = [
        {
            "mde_share_of_baseline": share,
            "mde80_target": share * baseline,
            "n_required": n_for_mde(mde_ref, n_ref, share * baseline),
        }
        for share in SHARE_TARGETS
    ]
    return {
        "scaling_assumption": "SE(n) = SE_hat * sqrt(197 / n) (same per-row variance and "
        "block dependence)",
        "grid": grid,
        "n_required": needed,
    }


# ------------------------------------------------------------ simulation ----


def ar1_variance_factor(n: int, rho: float) -> float:
    """v(n, rho) with Var(mean of n unit-variance AR(1) values) = v / n."""
    if rho == 0.0:
        return 1.0
    k = np.arange(1, n)
    return float(1.0 + 2.0 * np.sum((1.0 - k / n) * rho**k))


def draw_noise(rng: np.random.Generator, n: int, family: str, rho: float) -> np.ndarray:
    """Stationary unit-variance noise: Gaussian AR(1) or iid scaled Student-t(4)."""
    if family == "t4":
        if rho != 0.0:
            raise ValueError("t4 family is iid only")
        return rng.standard_t(4, size=n) * math.sqrt(0.5)
    if family != "gauss":
        raise ValueError(f"unknown noise family {family!r}")
    e = rng.standard_normal(n)
    if rho == 0.0:
        return e
    u = np.empty(n)
    u[0] = e[0]
    scale = math.sqrt(1.0 - rho**2)
    for t in range(1, n):
        u[t] = rho * u[t - 1] + scale * e[t]
    return u


def interpolate_k(k_grid: Sequence[float], power: Sequence[float]) -> float | None:
    """First k at which power reaches POWER_TARGET (linear interpolation)."""
    for i, p in enumerate(power):
        if p >= POWER_TARGET:
            if i == 0:
                return float(k_grid[0])
            k0, k1 = k_grid[i - 1], k_grid[i]
            p0, p1 = power[i - 1], p
            return float(k0 + (POWER_TARGET - p0) * (k1 - k0) / (p1 - p0))
    return None


def simulate_cell(
    cell_index: int,
    name: str,
    family: str,
    rho: float,
    n: int,
    reps: int,
    sigma: float,
    k_grid: Sequence[float],
) -> dict[str, Any]:
    se_true = sigma * math.sqrt(ar1_variance_factor(n, rho) / n)
    detect = np.zeros(len(k_grid), dtype=int)
    harm = np.zeros(len(k_grid), dtype=int)
    se_ratio = np.empty(reps)
    zeros = np.zeros(n)
    for rep in range(reps):
        seq = np.random.SeedSequence(MASTER_SEED, spawn_key=(cell_index, rep))
        rng = np.random.default_rng(seq)
        noise = sigma * draw_noise(rng, n, family, rho)
        boot_seed = int(rng.integers(0, 2**31 - 1))
        for j, k in enumerate(k_grid):
            d = noise - k * se_true
            ci = _block_bootstrap_delta_ci(
                d, zeros, n_resamples=N_RESAMPLES, block_size=BLOCK_SIZE, seed=boot_seed
            )
            detect[j] += ci["ci_high"] < 0
            harm[j] += ci["ci_low"] > 0
            if j == 0:
                se_ratio[rep] = se_from_ci(ci["ci_low"], ci["ci_high"]) / se_true
    power = (detect / reps).tolist()
    k80 = interpolate_k(k_grid, power)
    size = float((detect[0] + harm[0]) / reps) if k_grid[0] == 0.0 else None
    return {
        "cell_index": cell_index,
        "name": name,
        "family": family,
        "rho": rho,
        "n": n,
        "replications": reps,
        "sigma_per_row": sigma,
        "se_true": se_true,
        "k_grid": list(k_grid),
        "planted_mean_delta": [-k * se_true for k in k_grid],
        "power_ci_high_below_0": power,
        "rate_ci_low_above_0": (harm / reps).tolist(),
        "size_two_sided_at_k0": size,
        "mean_bootstrap_se_over_true_se": float(np.mean(se_ratio)),
        "sim_mde80_in_se": k80,
        "sim_mde80": None if k80 is None else k80 * se_true,
        "seed_rule": f"SeedSequence({MASTER_SEED}, spawn_key=({cell_index}, rep)); "
        "noise then bootstrap seed = integers(0, 2**31-1)",
    }


def run_simulation(
    se_ref: float,
    n_ref: int,
    cells: Sequence[tuple[str, str, float, int, int]],
    k_grid: Sequence[float],
) -> list[dict[str, Any]]:
    out = []
    for idx, (name, family, rho, n, reps) in enumerate(cells):
        sigma = se_ref / math.sqrt(ar1_variance_factor(n_ref, rho) / n_ref)
        out.append(simulate_cell(idx, name, family, rho, n, reps, sigma, k_grid))
    return out


# ------------------------------------------------------ baseline skill ----


def auc_null_se(n1: int, n0: int) -> float:
    return math.sqrt((n1 + n0 + 1) / (12.0 * n1 * n0))


def baseline_skill(
    inputs: dict[tuple[str, str], dict[str, Any]], root: Path = ROOT
) -> list[dict[str, Any]]:
    table: dict[tuple[str, str, str], dict[str, str]] = {}
    with (root / PERIOD_METRICS).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table[(row["period"], row["target"], row["model"])] = row
    rows = []
    for (period, target), inp in inputs.items():
        b = table[(period, target, "model0_majority_baseline")]
        m = table[(period, target, "model1_market_only")]
        n = int(m["samples"])
        n1 = int(round(inp["observed_positive_rate"] * n))
        se0 = auc_null_se(n1, n - n1)
        auc = float(m["roc_auc"])
        ll_diff = float(m["log_loss"]) - float(b["log_loss"])
        mde_cell = mde80(se_from_ci(inp["ci_low"], inp["ci_high"]))
        z = (auc - 0.5) / se0
        rows.append(
            {
                "period": period,
                "target": target,
                "n": n,
                "n_positive": n1,
                "baseline_log_loss": float(b["log_loss"]),
                "market_log_loss": float(m["log_loss"]),
                "market_minus_baseline_log_loss": ll_diff,
                "market_minus_baseline_brier": float(m["brier_score"])
                - float(b["brier_score"]),
                "market_balanced_accuracy": float(m["balanced_accuracy"]),
                "market_roc_auc": auc,
                "auc_null_se": se0,
                "auc_z": z,
                "auc_mde80": 0.5 + MDE_MULTIPLIER * se0,
                "cell_mde80_log_loss": mde_cell,
                "without_demonstrable_skill": bool(abs(ll_diff) < mde_cell and z < Z_95),
            }
        )
    return rows


# ------------------------------------------------ illustrative translation --


def _calibrated_grid(tau: float, points: int = 20001) -> tuple[np.ndarray, np.ndarray]:
    s = np.linspace(-6 * tau, 6 * tau, points)
    w = np.exp(-0.5 * (s / tau) ** 2)
    w /= w.sum()
    return np.clip(s, -0.49, 0.49), w


def expected_ll_gain(tau: float) -> float:
    s, w = _calibrated_grid(tau)
    p = 0.5 + s
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return float(math.log(2.0) - np.sum(w * entropy))


def calibrated_auc(tau: float) -> float:
    s, w = _calibrated_grid(tau)
    pos = w * (0.5 + s)
    neg = w * (0.5 - s)
    pos /= pos.sum()
    neg /= neg.sum()
    cneg = np.cumsum(neg) - neg  # mass strictly below each grid point
    return float(np.sum(pos * (cneg + 0.5 * neg)))


def translate_mde(mde: float) -> dict[str, Any]:
    lo, hi = 1e-6, 0.3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if expected_ll_gain(mid) < mde:
            lo = mid
        else:
            hi = mid
    tau = 0.5 * (lo + hi)
    return {
        "mde80_log_loss": mde,
        "tau_rms_probability_deviation": tau,
        "expected_log_loss_gain_at_tau": expected_ll_gain(tau),
        "calibrated_forecast_auc": calibrated_auc(tau),
        "assumptions": "calibrated p = 0.5 + s, s ~ N(0, tau^2) clipped to [-0.49, 0.49], "
        "base rate 0.5; gain measured against the constant 0.5 forecast",
    }


# ------------------------------------------------------------------ figure --


def render_svg(proj: dict[str, Any], sims: list[dict[str, Any]], baseline: float) -> str:
    """MDE80 vs n on log-log axes; analytic line + simulated points."""
    width, height = 720, 440
    left, right, top, bottom = 80, 170, 56, 64
    pw, ph = width - left - right, height - top - bottom
    x_lo, x_hi = math.log10(150), math.log10(30000)
    y_lo, y_hi = math.log10(0.0008), math.log10(0.02)

    def x(n: float) -> float:
        return left + (math.log10(n) - x_lo) / (x_hi - x_lo) * pw

    def y(v: float) -> float:
        return top + (1 - (math.log10(v) - y_lo) / (y_hi - y_lo)) * ph

    ink, muted, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
    s1, s2 = "#2a78d6", "#eb6834"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="Helvetica, Arial, sans-serif" '
        'role="img" aria-labelledby="t d">',
        '<title id="t">Minimum detectable log-loss improvement vs number of filing events</title>',
        '<desc id="d">Analytic MDE at 80% power and 5% size, scaled as 1/sqrt(n) from the '
        "committed 2025 primary-target interval (n = 197), with Monte Carlo points from the "
        "production block bootstrap.</desc>",
        f'<rect width="{width}" height="{height}" fill="#fcfcfb"/>',
        f'<text x="{left}" y="24" font-size="15" font-weight="600" fill="{ink}">'
        "Smallest log-loss improvement detectable with 80% power</text>",
        f'<text x="{left}" y="42" font-size="12" fill="{muted}">Combined minus market-only, '
        "primary target; nats per filing event (log scale)</text>",
    ]
    for v in (0.001, 0.002, 0.005, 0.01, 0.02):
        parts.append(
            f'<line x1="{left}" x2="{left + pw}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
            f'stroke="{grid_c}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y(v) + 4:.1f}" font-size="11" fill="{muted}" '
            f'text-anchor="end">{v:g}</text>'
        )
    for n in N_GRID:
        parts.append(
            f'<text x="{x(n):.1f}" y="{top + ph + 18}" font-size="11" fill="{muted}" '
            f'text-anchor="middle">{n:,}</text>'
        )
    parts.append(
        f'<line x1="{left}" x2="{left + pw}" y1="{top + ph}" y2="{top + ph}" '
        f'stroke="{muted}" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{left + pw / 2}" y="{height - 18}" font-size="12" fill="{muted}" '
        'text-anchor="middle">Evaluation filing events, n (log scale)</text>'
    )
    # 1% of baseline log loss reference
    ref = 0.01 * baseline
    parts.append(
        f'<line x1="{left}" x2="{left + pw}" y1="{y(ref):.1f}" y2="{y(ref):.1f}" '
        f'stroke="{muted}" stroke-width="1" stroke-dasharray="4 4"/>'
    )
    parts.append(
        f'<text x="{left + pw + 8}" y="{y(ref) + 4:.1f}" font-size="11" fill="{muted}">'
        "1% of baseline log loss</text>"
    )
    pts = " ".join(f"{x(g['n']):.1f},{y(g['mde80']):.1f}" for g in proj["grid"])
    parts.append(
        f'<polyline points="{pts}" fill="none" stroke="{s1}" stroke-width="2" '
        'stroke-linejoin="round"/>'
    )
    for g in proj["grid"]:
        parts.append(
            f'<circle cx="{x(g["n"]):.1f}" cy="{y(g["mde80"]):.1f}" r="4" fill="{s1}" '
            f'stroke="#fcfcfb" stroke-width="2"><title>n = {g["n"]:,}: analytic MDE80 = '
            f'{g["mde80"]:.4f} nats</title></circle>'
        )
    base_sims = [c for c in sims if c["name"] == "gauss_rho0" and c["sim_mde80"] is not None]
    for c in base_sims:
        cx, cy = x(c["n"]), y(c["sim_mde80"])
        parts.append(
            f'<rect x="{cx - 5:.1f}" y="{cy - 5:.1f}" width="10" height="10" fill="{s2}" '
            f'stroke="#fcfcfb" stroke-width="2" transform="rotate(45 {cx:.1f} {cy:.1f})">'
            f'<title>n = {c["n"]:,}: simulated MDE80 = {c["sim_mde80"]:.4f} nats '
            f'(R = {c["replications"]})</title></rect>'
        )
    last = proj["grid"][-1]
    parts.append(
        f'<text x="{x(last["n"]) + 10:.1f}" y="{y(last["mde80"]) + 4:.1f}" font-size="11" '
        f'fill="{ink}">Analytic (1/sqrt n)</text>'
    )
    # legend
    lx, ly = left + pw + 8, top + 8
    parts.append(
        f'<line x1="{lx}" x2="{lx + 18}" y1="{ly}" y2="{ly}" stroke="{s1}" stroke-width="2"/>'
    )
    parts.append(f'<circle cx="{lx + 9}" cy="{ly}" r="4" fill="{s1}"/>')
    parts.append(
        f'<text x="{lx + 24}" y="{ly + 4}" font-size="11" fill="{ink}">Analytic</text>'
    )
    parts.append(
        f'<rect x="{lx + 4}" y="{ly + 15}" width="10" height="10" fill="{s2}" '
        f'transform="rotate(45 {lx + 9} {ly + 20})"/>'
    )
    parts.append(
        f'<text x="{lx + 24}" y="{ly + 24}" font-size="11" fill="{ink}">Simulated '
        "(production</text>"
    )
    parts.append(
        f'<text x="{lx + 24}" y="{ly + 38}" font-size="11" fill="{ink}">bootstrap)</text>'
    )
    head = proj["grid"][0]
    parts.append(
        f'<text x="{x(head["n"]) + 10:.1f}" y="{y(head["mde80"]) - 10:.1f}" font-size="11" '
        f'fill="{ink}">n = 197: {head["mde80"]:.4f}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# -------------------------------------------------------------------- main --


def build(smoke: bool = False, root: Path = ROOT) -> dict[str, Any]:
    inputs = load_inputs(root)
    head = inputs[HEADLINE]
    se_ref = se_from_ci(head["ci_low"], head["ci_high"])
    analytic = analytic_table(inputs)
    proj = projection(se_ref, head["n"], head["baseline_log_loss"])
    cells = SMOKE_CELLS if smoke else FULL_CELLS
    k_grid = SMOKE_K_GRID if smoke else K_GRID
    sims = run_simulation(se_ref, head["n"], cells, k_grid)
    mde_ref = mde80(se_ref)
    base197 = next(c for c in sims if c["name"] == "gauss_rho0" and c["n"] == head["n"])
    sim_mde = base197["sim_mde80"]
    agree = sim_mde is not None and abs(sim_mde / mde_ref - 1) <= 0.15
    size0 = base197["size_two_sided_at_k0"]
    return {
        "study": "power / minimum-detectable-effect calibration, filing-text comparison",
        "protocol": "research/power/PROTOCOL.md",
        "mode": "smoke" if smoke else "full",
        "inputs": {f"{p}/{t}": v for (p, t), v in inputs.items()},
        "constants": {
            "z_95": Z_95,
            "z_80": Z_80,
            "mde_multiplier": MDE_MULTIPLIER,
            "ci_width_in_se": CI_WIDTH_IN_SE,
            "master_seed": MASTER_SEED,
            "n_resamples": N_RESAMPLES,
            "block_size": BLOCK_SIZE,
            "k_grid": list(k_grid),
            "n_grid": list(N_GRID),
            "production_function": "quant_sentiment.historical_text_study_v2."
            "_block_bootstrap_delta_ci",
        },
        "headline": {
            "period": HEADLINE[0],
            "target": HEADLINE[1],
            "n": head["n"],
            "se_hat": se_ref,
            "analytic_mde80": mde_ref,
            "analytic_mde80_share_of_baseline": mde_ref / head["baseline_log_loss"],
            "baseline_log_loss": head["baseline_log_loss"],
            "realized_ci_low": head["ci_low"],
            "realized_ci_high": head["ci_high"],
            "sim_mde80_base_cell": sim_mde,
            "sim_over_analytic": None if sim_mde is None else sim_mde / mde_ref,
            "rule2_simulation_agrees_within_15pct": agree,
            "headline_mde80": mde_ref if agree or sim_mde is None else sim_mde,
            "headline_source": "analytic" if agree or sim_mde is None else "simulated",
            "base_cell_size_at_k0": size0,
            "rule3_size_flag_anticonservative": bool(size0 is not None and size0 > 0.075),
        },
        "analytic": analytic,
        "projection": proj,
        "simulation": sims,
        "baseline_skill": baseline_skill(inputs, root),
        "illustrative_translation": translate_mde(mde_ref),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-svg", type=Path)
    parser.add_argument("--smoke", action="store_true", help="fast reduced run (tests)")
    args = parser.parse_args(argv)
    started = time.time()
    result = build(smoke=args.smoke)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_svg is not None:
        args.out_svg.parent.mkdir(parents=True, exist_ok=True)
        args.out_svg.write_text(
            render_svg(result["projection"], result["simulation"],
                       result["headline"]["baseline_log_loss"]),
            encoding="utf-8",
        )
    print(json.dumps(result["headline"], indent=2))
    print(f"runtime_seconds={time.time() - started:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
