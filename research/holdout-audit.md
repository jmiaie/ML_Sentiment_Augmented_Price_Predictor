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
