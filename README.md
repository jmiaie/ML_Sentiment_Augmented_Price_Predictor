# ML_Sentiment_Augmented_Price_Predictor

## Result

**Sentiment features did not add measurable predictive value over market-only features** on the pre-registered historical evaluation (12 large-cap issuers, SEC EDGAR 10-K/10-Q/8-K text, 2025 evaluation period, primary next-session excess-return direction target):

- n = 197 evaluation filing events
- combined-minus-market log-loss delta: **+0.0041** (block-bootstrap 95% interval [-0.0049, +0.0108]; positive means the combined model was worse)
- combined-minus-market balanced-accuracy delta: **-0.027**
- The sample is underpowered: it cannot rule out small effects in either direction.

Source: `results/historical_text_v2/sentiment_hist_text_v2_primary_historical_evaluation.json`. Write-ups: [case study](publication/sentiment-study/CASE-STUDY.md), [technical paper](publication/sentiment-study/TECHNICAL-PAPER.md).

Despite the repository name, this is an evaluation of incremental information, not a price predictor.

This repository is a lean validation framework for one question:

> Does point-in-time sentiment add incremental predictive information beyond market-only features for short-horizon asset returns?

The current implementation is intentionally modest and reproducible:

- no paid APIs or credentials
- no network access, financial downloads, transformer downloads, or GPU requirements in CI
- no claims of historical sentiment alpha (the historical study found none)
- deterministic synthetic workflows for methodology validation

The historical study evaluates lexicon (Loughran-McDonald) sentiment from SEC EDGAR filings against market-only features; see the [case study](publication/sentiment-study/CASE-STUDY.md) for the full write-up.

- Authoritative results (v2): `results/historical_text_v2/`
- Superseded exploratory results (v1: 3 issuers, 8-K only; preserved, not a source for the result above): `results/historical_text/`
- Data manifests: `data/manifests/`

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
python -m pip install -e ".[dev,data]"
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
- `research/holdout-audit.md`: history of the 2025 evaluation window and leakage fixes
- `research/portfolio-rationalization.md`: factual repository positioning and hub-update recommendation

## Limitations

- **The 2025 evaluation window was not an untouched holdout.** An earlier exploratory v1 study had already evaluated 2025 for three of the twelve issuers (AAPL, MSFT, AMZN), under code later found to contain a purge off-by-one. The authoritative v2 study re-used the 2025 window, and its result is therefore labeled `PREVIOUSLY INSPECTED / HISTORICAL EVALUATION` rather than a confirmatory out-of-sample test (see `research/holdout-audit.md`).
- n = 197 evaluation events is too few to detect small effects; a null result here is not evidence that filing text carries no information.
- Lexicon-based scores only; no transformer or news-feed sentiment.

## Non-goals

The repository does not claim or implement:

- Docker support
- SHAP explainability
- XGBoost
- paid or proprietary sentiment datasets (the historical study uses public SEC EDGAR filings and a public dictionary only)
- economic-performance headline metrics such as Sharpe

Those would require additional reproducible evidence and are intentionally omitted here.
