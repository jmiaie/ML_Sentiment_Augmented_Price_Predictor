# Sentiment incremental predictive value

## Research question

Does point-in-time sentiment add incremental predictive information beyond market-only features for short-horizon asset returns?

## Current answer

This repository can answer that question only at the **methodology** level today.

- The code enforces point-in-time event availability.
- The validation design is chronological and leakage-aware.
- The synthetic experiment injects known incremental sentiment signal and checks that the combined model can recover it out of sample.

Historical results pending reproducible point-in-time dataset.

## Data status

The committed artifact uses deterministic synthetic data only. No paid APIs, credentials, internet downloads, transformer downloads, or financial downloads are required in CI.

## Event schema

Each sentiment event carries:

- source
- event ID
- entity
- symbol
- publication timestamp
- ingestion timestamp
- availability timestamp
- timezone
- sentiment score
- model version
- confidence
- `effective_trading_timestamp`

`effective_trading_timestamp` is computed using explicit US session rules:

- before open -> same-day open
- intraday -> immediate
- after close -> next trading-day open
- weekend/holiday -> next trading-day open

## Feature and label design

### Market features

- 1-day return
- 3-day return
- 3-day realized volatility
- 3-day volume z-score

### Sentiment features

- 5-day event count
- 5-day mean sentiment
- 5-day confidence-weighted mean sentiment
- 5-day mean confidence

### Labels

- primary: `next_session_direction`
- secondary: `next_two_session_direction`

Both labels begin strictly after the feature cutoff.

## Validation design

- expanding-window walk-forward validation for tuning
- untouched final chronological test window when enough data exist
- horizon-driven gap/embargo between train and OOS windows
- imputers/scalers/models fit only on the relevant train slices

## Baselines and ablation

All evaluated on identical out-of-sample windows:

1. majority baseline
2. market-only logistic regression
3. sentiment-only logistic regression
4. combined logistic regression

Reported metrics include accuracy, balanced accuracy, precision, recall, F1, ROC AUC when legitimate, log loss, Brier score, and calibration-table data.

## Synthetic methodology validation

The generated artifact is explicitly labeled `Synthetic methodology validation`.

Interpretation:

- a positive combined-model result supports that the pipeline can detect injected incremental information when it exists
- it does **not** establish real-market predictive value
- it does **not** justify economic or Sharpe claims
