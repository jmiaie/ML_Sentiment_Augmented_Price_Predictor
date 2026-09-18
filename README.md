# ML_Sentiment_Augmented_Price_Predictor

This repository is a lean validation framework for one question:

> Does point-in-time sentiment add incremental predictive information beyond market-only features for short-horizon asset returns?

The current implementation is intentionally modest and reproducible:

- no paid APIs or credentials
- no network access, financial downloads, transformer downloads, or GPU requirements in CI
- no claims of historical sentiment alpha
- deterministic synthetic workflows for methodology validation

Historical D9 study (branch `research/historical-text-validation`, merged to `main` 2026-09-18 via PR #4, merge commit `0cafe8e`):
reproducible megacap EDGAR 8-K subset + yfinance prices with PIT lexicon scores.
See `research/holdout-audit.md`, `data/manifests/`, and `results/historical_text/`.
Synthetic methodology artifacts remain non-substitutes for historical claims.


## Scope

The package under `src/quant_sentiment` implements:

- a point-in-time sentiment event schema with publication, ingestion, availability, timezone, model version, confidence, and `effective_trading_timestamp`
- deterministic US market-session alignment rules for before-open, intraday, after-close, weekend, and holiday mapping
- leakage-safe trailing market and sentiment feature generation
- decision-time-safe labels with a primary next-session direction target and a documented two-session secondary horizon
- expanding-window walk-forward validation with horizon-driven embargo/gap handling
- ablations on identical out-of-sample periods:
  - majority baseline
  - market-only logistic regression
  - sentiment-only logistic regression
  - combined logistic regression
- classification metrics, log loss, Brier score, and calibration-table data
- deterministic synthetic methodology validation artifacts

## Market-session rules

The testable default calendar uses US regular-hours assumptions:

- session open: 09:30 America/New_York
- session close: 16:00 America/New_York
- before open: effective at same-day session open
- intraday: effective immediately
- after close: effective at next trading-session open
- weekend or configured holiday: effective at next trading-session open

## Quick start

```bash
python -m pip install -e .[dev]
ruff check .
mypy src tests
pytest
python -m quant_sentiment.synthetic_validation \
  --output artifacts/synthetic_methodology_validation.json
```

The generated artifact is explicitly labeled `Synthetic methodology validation` and should not be interpreted as historical evidence.

## Repository outputs

- `artifacts/synthetic_methodology_validation.json`: generated synthetic ablation results
- `research/sentiment-incremental-predictive-value.md`: research framing and limitations
- `research/model-risk.md`: model-risk and leakage-control notes
- `research/portfolio-rationalization.md`: factual repository positioning and hub-update recommendation

## Non-goals in this rebuild

The repository does not currently claim or implement:

- Docker support
- SHAP explainability
- XGBoost
- real historical sentiment datasets
- economic-performance headline metrics such as Sharpe

Those would require additional reproducible evidence and are intentionally omitted here.
