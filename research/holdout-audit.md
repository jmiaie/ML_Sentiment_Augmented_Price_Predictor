# Holdout audit — Sentiment Directive #9 (2025 calendar year)

**Repository:** `jmiaie/ML_Sentiment_Augmented_Price_Predictor`  
**Branch:** `research/historical-text-validation`  
**Audit date (PT):** 2026-09-15  
**Holdout window under D9:** calendar year **2025** (2025-01-01 through 2025-12-31)  
**Dataset ID:** `edgar_8k_yf_megacap_daily_2015_2025_v1`

## Verdict

**CLEAR** — no evidence that calendar-year **2025** market data or filing text was
previously inspected, tuned against, or used for empirical evaluation /
performance claims in this repository.

## Scope searched

| Surface | Method | Finding |
|---|---|---|
| Working tree (`src`, `tests`, `research`, `artifacts`, README) | recursive text search for `2025` / `holdout` / empirical metrics | Scaffold / copyright / synthetic methodology only; no empirical 2025 holdout metrics. |
| Artifacts | `artifacts/synthetic_methodology_validation.json` | Explicitly labeled **Synthetic methodology validation** — not historical. |
| Research docs | `research/sentiment-incremental-predictive-value.md`, `model-risk.md`, `portfolio-rationalization.md` | Methodology framing; historical results pending reproducible PIT dataset. |
| Tests | `tests/` | Unit/synthetic fixtures for calendar, features, modeling, validation — no real-market 2025 packs. |
| Prior branches / PRs (session) | D5 PR #1 methodology harness; empty Copilot PR #3 | No historical OOS 2025 evaluation artifacts. |

## Classification rules applied

- **CLEAR:** holdout calendar window not used for model selection, hyperparameter tuning, benchmark cherry-picking, or reported historical performance.
- **PREVIOUSLY INSPECTED:** any committed notebook/result/config that evaluates or plots 2025 returns/sentiment metrics for research decisions.

Copyright / changelog year strings and synthetic generator dates do **not** count as empirical holdout inspection.

## Data scope (explicit; not expanded mid-study)

**Reproducible subset:** SEC EDGAR **8-K** primary documents for **AAPL / MSFT / AMZN** plus **yfinance** daily prices for the same symbols over `[2015-01-01, 2026-01-01)`. Sentiment scores from embedded lexicon `sentiment_d9_financial_polarity_v1` applied to real filing text. Point-in-time availability = EDGAR `acceptanceDateTime`. Direction labels from prices only.

**Not in scope:** full EDGAR corpus, 10-K/10-Q panels, paid news APIs, transformer embeddings, invented filing text, invented labels. Synthetic-only runs are **not** a substitute for the D9 historical claim.

## Restrictions until FINAL CONFIGURATION FROZEN

1. Do **not** evaluate models on 2025 for final claims.
2. Development / validation uses **2015–2023** (formation/dev) and **2024** (validation) only, once data is frozen.
3. Pre-registered experiment config remains **`not-yet-frozen-for-holdout`** until development+validation complete and a tracker `FINAL CONFIGURATION FROZEN` record exists on Issue #3.

## Sign-off

| Field | Value |
|---|---|
| Holdout status | **CLEAR** |
| Prior empirical 2025 evaluation artifacts | **null** (none found) |
| Ready for acquisition + pre-registration | **yes** |

## Post-freeze holdout execution

YAML status set to `frozen-for-holdout` after formation+validation. Tracker
**FINAL CONFIGURATION FROZEN — Sentiment D9-D** recorded on Issue #3. Holdout
experiment `sentiment_hist_text_v1_holdout_2025` executed once under frozen config.
No retune after freeze.

## Post-execution code defects found (2026-09-17) — two, partial, scoped corrections

An independent code audit (commissioned by this program, empirically verified against
the actual committed artifacts and code, not accepted on description alone) found two
real defects in the code that produced the three already-committed result artifacts
(`sentiment_hist_text_v1_{dev_formation,val_2024,holdout_2025}.json`), including the
already-executed 2025 holdout. **Both defects are fixed in the source going forward.
Neither has been used to regenerate the already-committed artifacts** — no local raw
data exists to re-run this study in this environment (in addition to the deeper
question, below, of whether re-running is even appropriate once 2025 has already been
observed once).

1. **Pre-freeze holdout-year leak via an unscoped summary field.** `key_metrics` in
   `run_period_ablation` (`src/quant_sentiment/historical_text_study.py`) computed
   every field correctly against the period-sliced `formation_frame`/`eval_frame`
   *except* `n_sentiment_events_in_frame`, which summed
   `full_frame["sentiment_event_count_5d"]` — the entire unsliced 2015–2025 series —
   regardless of which period's artifact was being written. Confirmed directly: this
   field reads `344` identically in all three artifacts (`dev_formation`, `val_2024`,
   `holdout_2025`), while the correctly-scoped `n_eval_rows` field on the same three
   artifacts correctly varies (`2264`, `252`, `249`). This means the `dev_formation`
   and `val_2024` artifacts — both generated *before* the holdout freeze, per the
   runner's own structural holdout-refusal gate — already carried a number with 2025
   filings baked into it, a genuine (if narrow) violation of "no 2025 access before
   freeze." **Scope:** this field is pure reporting metadata; it is never read by any
   model-fitting, model-selection, or metric-computation code path, so none of the
   headline accuracy/log-loss/Brier/ROC-AUC numbers in any of the three artifacts are
   affected. Fixed by scoping the field to `eval_frame` and renaming it
   `n_sentiment_events_in_eval` so its contract is unambiguous.
2. **Walk-forward embargo/gap formula off by one at the frozen config's own
   (label_horizon=1, embargo=0) values.** `build_walk_forward_plan`
   (`src/quant_sentiment/validation.py`) computed
   `gap = max(label_horizon - 1, 0) + embargo`, which is `0` at these exact values —
   the values the frozen config actually uses (`walk_forward_gap: 0` in every
   committed artifact). At gap=0, the last training row's own forward-looking label
   and the first evaluation row's own feature reduce to the algebraically identical
   expression `close[t+1]/close[t] - 1` for the same `t` — confirmed numerically, not
   just by index arithmetic. The repo's own `labels_are_non_overlapping` self-check
   (`>= label_horizon`) did not catch this, since `1 >= 1` is `True` even though an
   exact overlap exists; the existing test suite only ever exercised
   `(label_horizon=2, embargo=1)`, which happens to satisfy the correct bound by
   coincidence of its own chosen embargo. **Scope/materiality:** because the labels
   here are two-endpoint ratios (not cumulative/path-dependent), the leak is a single
   shared price bar at each of the 109 walk-forward fold boundaries used to select the
   locked `C` hyperparameter — not a multi-day overlapping window, and it recurs
   identically and symmetrically across all three tested `C` values at every fold, so
   it should not have differentially favored any one `C`. It is not expected to
   materially bias the already-reported accuracy/log-loss/Brier/ROC-AUC numbers, but
   it is a genuine, reproducible violation of the documented "gap prevents overlapping
   forward labels from leaking across splits" contract (`research/model-risk.md`,
   `research/sentiment-incremental-predictive-value.md`) at exactly the values this
   study actually froze and ran under. Fixed to `gap = label_horizon + embargo`, and
   the self-check tightened from `>=` to strict `>` to match its own name and close
   the same class of off-by-one for any future (label_horizon, embargo) combination.

**The open question this program cannot resolve unilaterally:** this study's 2025
holdout has already been observed once, under code carrying both defects above.
Re-running it under the corrected code — even though defect 2 could in principle
shift which fold-level rows enter each `C`-selection split, and therefore in
principle the selected `C` and the fitted model — would mean observing 2025 a second
time, which is exactly what this program's one-time-holdout discipline exists to
prevent. This is the same category of dilemma already on record for D9-C
(`options-volatility-risk-lab`)'s VaR fix. Left here for independent/owner review to
resolve, not decided by this fix.

## v2 AUTHORITATIVE rebuild: code/tests complete, real acquisition pending (2026-09-17)

The v1 study above remains preserved, unmodified, as EXPLORATORY /
NON-CONFORMING (per Directive #9's own classification — 3 issuers, 8-K only,
non-LM embedded lexicon, day-level sample unit with manufactured non-event
days, raw rather than excess-return targets, no proper NYSE calendar). A new
AUTHORITATIVE v2 was built from Directive #9's own D9-D spec text: the real
Loughran-McDonald Master Dictionary (via the `pysentiment2` package, never
vendored into this repo), a real NYSE trading calendar (`pandas_market_calendars`
— holidays, early closes, time zones), event-level filing observations (no
manufactured non-event days), excess-return-over-SPY targets (primary
1-session, secondary 5-session), a purged/session-distance-embargoed
walk-forward split (not a row-count gap — necessary once samples are pooled
across 12 issuers with irregular, asynchronous filing timing), and the
Model 0/1/2/3 confirmatory framework with the spec's own C grid. See
`configs/experiments/sentiment_historical_text_study_v2.yaml` for the frozen
(on creation, per that config's own freeze basis) configuration.

**All new code is unit-tested on synthetic fixtures and passing**
(`tests/test_nyse_calendar.py`, `test_lm_dictionary.py`,
`test_market_features_v2.py`, `test_walk_forward_v2.py`,
`test_event_frame_v2.py`, `test_historical_text_study_v2.py` — 45 new tests;
66 total in this repo's suite, all green). **The real 12-issuer SEC EDGAR
10-K/10-Q/8-K corpus has not been acquired** — this session's own egress
proxy blocks `data.sec.gov`/`www.sec.gov` (confirmed 403). No DEV/VAL/2025
execution has happened under v2, and no result artifacts or report exist
yet for it. `docs/D9_D_ACQUISITION_HANDOFF.md` documents exactly what a
network-capable environment needs to run
(`scripts/acquire_sec_filings_12issuer_daily.py`) and hand back.

Per Directive #9's own D9-D spec text, once real data lands: AAPL/MSFT/AMZN's
2025 outcomes were already inspected once under v1, and those three issuers
are part of the authoritative 12-issuer universe — so the conforming D9-D
2025 result must be labeled **`PREVIOUSLY INSPECTED / HISTORICAL EVALUATION`**,
never an untouched final holdout. The remaining nine issuers may not be
isolated and presented as a new untouched confirmatory sample. No shift to
2026 (reserved program-wide).
