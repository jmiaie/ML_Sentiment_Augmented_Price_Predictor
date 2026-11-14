# Power / minimum-detectable-effect study: filing-text comparison (frozen design)

Status: **FROZEN** at the commit that adds this file (design freeze, 2026-10-20). That commit
contains this protocol, the code and the tests. It is pushed before the full computation runs.
Results and the report arrive in a later commit. Any later change to this design is recorded
below under "Amendments", dated and with its reason.

This is a calibration of the existing, published design. It is not a new evaluation of 2025
data, it does not retune anything, and it changes no frozen artifact, config, manifest, result
JSON or publication-pack file.

## 1. Question

The authoritative historical study reports, for the primary target (next-session excess-return
direction) in the 2025 evaluation block, a combined-minus-market log-loss delta of +0.0041
(positive = the combined model is worse), with block-bootstrap 95% interval
[-0.0048506, +0.0108001] (500 resamples, block size 10, seed 0), n = 197. The bootstrap
interval includes zero, so the study failed to show that filing text improves the forecast.

This study asks: **how large a log-loss improvement could that design have detected?** It also
asks whether the comparison model (market-only) had any skill over the majority-class baseline.

## 2. Inventory of committed artifacts (done before writing this protocol)

| Artifact | Per-observation predictions? | Per-observation losses? | What it does hold |
|---|---|---|---|
| `results/historical_text_v2/*_{formation_dev,validation,historical_evaluation}.json` (authoritative, 6 files) | no | no | period metrics per model, headline delta, bootstrap CI (low, high, mean, resamples, block size), n |
| `results/historical_text/*.json` (superseded v1, 3 files) | no | no | aggregate metrics only; not used here |
| `publication/sentiment-study/tables/target-comparison.csv` | no | no | delta and CI per period and target |
| `publication/sentiment-study/tables/period-metrics.csv` | no | no | log loss, Brier, balanced accuracy, ROC AUC, accuracy, n per model |
| `data/` | no | no | manifests only; the price and filing corpora are not committed |

No per-observation predictions or losses are committed, and the inputs needed to regenerate
them are not in the repository (rule 4 forbids network acquisition). The preferred method
(resampling the committed per-observation losses with planted shifts) is therefore not
available. The design falls back to the analytic method, and checks it by synthetic
simulation through the production bootstrap.

## 3. Method

### 3.1 Inputs (committed artifacts only)

The six authoritative v2 result JSONs. For each file the code records its sha256 and checks it
against `artifact_sha256` in `research/experiment-ledger.csv`. If any hash disagrees, the run
fails. Values used: `key_metrics.headline_bootstrap_delta_log_loss_ci95.{ci_low, ci_high,
mean_delta, n_resamples, block_size}`, `key_metrics.n_eval_rows` and `period_metrics.*`.

### 3.2 Headline: analytic MDE from the committed interval

- SE_hat = (ci_high - ci_low) / 3.92 (a normal 95% interval spans 2 x 1.96 SE).
- MDE80 = (1.96 + 0.84) x SE_hat = 2.80 x SE_hat. This is the smallest true improvement that a
  two-sided 5%-size test would detect 80% of the time.
- The MDE is reported in log-loss units (nats per observation), and as a share of the
  majority-baseline log loss for the same block (primary 2025: 0.6911031).
- Headline cell: primary target, `historical_evaluation` (2025), n = 197. The same calculation
  is reported for every other committed (period, target) interval as a secondary table:
  `formation_dev` n = 20, `validation` (2024) n = 210, both targets.

Assumptions: (a) the bootstrap distribution of the mean delta is close to normal, so the
interval width maps onto an SE; (b) the SE does not depend on the true shift (a location
shift); (c) the observed interval is a fair estimate of the design's SE. In fact the interval
is somewhat asymmetric about its mean (lower arm 0.0089, upper arm 0.0067), and it comes from
19 blocks and 500 resamples, so the SE itself is uncertain. Assumption (a) is checked in 3.4.

### 3.3 Projection over n

n grid: {197, 500, 1000, 5000, 20000}. Scaling assumption: SE(n) = SE_hat x sqrt(197 / n), so
MDE80(n) = 2.80 x SE(n). This holds if more observations have the same per-row variance and the
same short-range block dependence (block size 10). Caveats: more data would come from more
years or more issuers. That can change heterogeneity, clustering of filings in earnings
seasons, and the models themselves (more training data). Dependence longer than the block
would make the true SE shrink more slowly than 1/sqrt(n).

Also reported: the n at which MDE80 equals 1% and 0.5% of the baseline log loss,
n* = 197 x (MDE80(197) / target)^2.

### 3.4 Check: synthetic simulation through the production bootstrap (rule 3)

The production statistic is `quant_sentiment.historical_text_study_v2._block_bootstrap_delta_ci`
(moving-block bootstrap, n // 10 blocks of 10 consecutive rows, 500 resamples, 2.5/97.5
percentiles). The decision rule implied by the published reading is "the 95% interval excludes
zero". The simulation measures the actual size and power of that rule:

- Synthetic per-row loss differences d_t = mu + sigma x u_t, t = 1..n. The production function
  is called with `losses_model3 = d` and `losses_model1 = 0` (so its `diffs` equals d exactly),
  `n_resamples = 500`, `block_size = 10`.
- u_t is a stationary process with unit marginal variance:
  - `gauss_rho0` (base): iid N(0, 1);
  - `gauss_rho0.3`, `gauss_rho0.6`: Gaussian AR(1) with coefficient rho (u_1 ~ N(0, 1),
    u_t = rho u_{t-1} + sqrt(1 - rho^2) e_t);
  - `t4_rho0`: iid Student-t with 4 degrees of freedom, scaled to unit variance (heavy tails).
- sigma is set so that the true SD of the sample mean at n = 197 equals SE_hat (primary, 2025):
  sigma = SE_hat / sqrt(v(197, rho) / 197), with v(n, rho) = 1 + 2 sum_{k=1}^{n-1} (1 - k/n) rho^k.
  At other n the same sigma and rho are used, so SE(n) = sigma x sqrt(v(n, rho) / n).
- Planted shifts (negative = combined model better): mu = -k x SE(n),
  k in {0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.8, 3.0, 3.5, 4.0}. The same noise draw and the same
  bootstrap seed are used for every k within a replication (common random numbers).
- Power at k = share of replications with ci_high < 0 (improvement detected). Size at k = 0 =
  share with ci_high < 0 or ci_low > 0 (nominal 5%). The share with ci_low > 0 is recorded at
  every k.
- Simulated MDE80 = k at which power reaches 0.80, by linear interpolation between grid points,
  times SE(n). If power never reaches 0.80 on the grid, the result is reported as "> 4 SE".
- Cells: `gauss_rho0` at n in {197, 500, 1000}; `gauss_rho0.3`, `gauss_rho0.6` and `t4_rho0` at
  n = 197. n = 5000 and n = 20000 are analytic only. Simulating them through the production
  bootstrap is slow, and with hundreds of blocks the bootstrap is near its normal limit.
- Replications: R = 2000 per cell at n = 197, R = 1000 at n = 500 and n = 1000. The Monte Carlo
  SE of a power near 0.8 is about 0.009 at R = 2000 and 0.013 at R = 1000.
- Also recorded per cell: mean bootstrap-implied SE ((ci_high - ci_low) / 3.92) relative to the
  true SE(n), as a check of the bootstrap's own calibration.

Seeds: master seed 20261020. Each (cell, replication) gets
`numpy.random.SeedSequence(20261020, spawn_key=(cell_index, rep))`. The noise is drawn from that
generator, and the production bootstrap seed is the next draw `integers(0, 2**31 - 1)` from the
same generator. The cells are indexed in the order listed above.

Smoke version (in the test suite, < 30 s): the base cell at n = 197, R = 40, k in {0, 2.8};
plus unit tests for the analytic formulas, the AR(1) variance factor, input-hash verification
and the translation-equivariance of the production bootstrap.

### 3.5 Baseline-skill comparison (from committed tables)

From `publication/sentiment-study/tables/period-metrics.csv`, for every (period, target): the
market-only minus majority-baseline differences in log loss, Brier score, balanced accuracy and
ROC AUC. For the market-only AUC, the null SE under no skill is the Hanley-McNeil /
Mann-Whitney value, SE0 = sqrt((n1 + n0 + 1) / (12 n1 n0)). It assumes independent rows, which
is a stated simplification. n1 = round(observed_positive_rate x n), using the baseline's
calibration table in the result JSON. Reported: z = (AUC - 0.5) / SE0 and the AUC MDE80 =
0.5 + 2.80 x SE0.

### 3.6 Illustrative translation of the log-loss MDE

The combined model has to beat a market-only model that sits at the baseline. So an
improvement of MDE80 over market-only is roughly the same as an absolute improvement of MDE80
over a constant forecast. For a calibrated forecast p = 0.5 + s, with s ~ N(0, tau^2) clipped
to [-0.49, 0.49] and base rate 0.5, the code solves (by bisection on a fine grid, with no Monte
Carlo) for the tau whose expected log-loss reduction against the constant 0.5 forecast equals
MDE80(197). It also reports the ROC AUC of that calibrated forecast. This is only a
translation into plainer units, under stated simplifications (base rate 0.5 rather than 0.46,
Gaussian spread). It is not an estimate.

## 4. Interpretation rules (written before any result)

1. **Headline MDE** is the analytic MDE80 at n = 197 (primary, 2025), in nats and as a share of
   baseline log loss.
2. **Simulation agreement.** If the simulated MDE80 of the base cell at n = 197 is within +/-15%
   of the analytic value, the analytic MDE is confirmed and stays the headline. If not, both are
   reported and the simulated value is the headline, because it measures the production decision
   rule. This rule is fixed here, so applying it needs no amendment.
3. **Size.** If the base-cell rejection rate at k = 0 is above 7.5% (nominal 5%, allowing for
   Monte Carlo error and few blocks), the report flags the published percentile block bootstrap
   as anti-conservative at n = 197. This is a finding, not a fix. Production code is not changed.
4. **What the null does and does not rule out.** The report states: (a) the design had about
   80% power only for improvements of at least MDE80; improvements below that are not excluded
   by the design; (b) separately, the realized interval excludes improvements larger than
   |ci_low| = 0.0049 nats at the 95% level (a statement about this sample, not about the
   design's power). No equivalence or "no effect" claim is made.
5. **Baseline skill.** The market-only model is called "without demonstrable skill" in a
   (period, target) cell when |log-loss difference vs baseline| < MDE80 for that cell's n and the
   AUC z-score is below 1.96. Otherwise the report gives the numbers and says skill is suggested
   in that cell, without claiming significance beyond the stated approximations.
6. Nothing in this study changes the published 2025 conclusion. It only calibrates how much that
   conclusion can carry.

## 5. Outputs

- `results/power/sentiment_mde.json`: all inputs with hashes, analytic table, projection, every
  simulation cell (seeds, R, power per k, size, bootstrap-SE ratio, simulated MDE), baseline-skill
  table, illustrative translation, runtime environment.
- `results/power/sentiment_mde_vs_n.svg`: MDE80 against n (log-log), analytic line plus
  simulated points. Hand-written SVG, because the locked environment has no plotting library.
- `research/power/REPORT.md`: summary paragraph, question, design, results, interpretation,
  limitations, reproduction command.

Reproduction (full run):

```bash
python research/power/sentiment_mde.py \
  --out-json results/power/sentiment_mde.json \
  --out-svg results/power/sentiment_mde_vs_n.svg
```

## Amendments

None.
