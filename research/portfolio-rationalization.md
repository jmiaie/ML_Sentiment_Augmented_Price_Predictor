# Portfolio rationalization report

## Repository role

`ML_Sentiment_Augmented_Price_Predictor` is now positioned as a reproducible methodology harness for testing whether point-in-time sentiment adds incremental predictive information beyond market-only features.

## What this repository now demonstrates

- point-in-time event handling
- leakage-safe feature and label construction
- expanding-window walk-forward validation
- ablation between market-only, sentiment-only, and combined logistic baselines
- deterministic synthetic methodology validation

## What it does not currently demonstrate

- real historical sentiment alpha
- production data ingestion
- economic performance claims
- model portability across assets or regimes

## Overlap / duplicate assessment

No specific named overlap candidates were supplied inside this repository-local directive, and this rebuild does not archive, delete, rename, or modify any external repository or hub entry.

The safest current interpretation is:

- keep this repository if a dedicated sentiment-methodology example is useful in the portfolio
- do not present it as proven historical alpha research until a reproducible point-in-time dataset is added

## Suggested hub update

Use language close to the following:

> **ML_Sentiment_Augmented_Price_Predictor** — synthetic methodology harness for evaluating whether point-in-time sentiment could add incremental predictive information beyond market-only features for short-horizon returns; historical results pending a reproducible point-in-time dataset.


## Sibling repository names (2026-09-30)

Canonical code: **this repo** (`jmiaie/ML_Sentiment_Augmented_Price_Predictor`).

- `ML_Sentiment_Augmented_Price_Predictor_private` — empty placeholder (initial commit only); not a fork of results.
- `ML_Sentiment_Augmented_Price_Predictor_public` — empty placeholder; not a portfolio showcase of this harness.

Hub narrative: treat as a chapter under `quant-research-portfolio`, not a free-standing alpha product. See also `docs/STATUS.md`.
