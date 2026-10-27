# Portfolio rationalization report

## Repository role

`ML_Sentiment_Augmented_Price_Predictor` is now positioned as a reproducible methodology harness for testing whether point-in-time sentiment adds incremental predictive information beyond market-only features.

## What this repository now demonstrates

- point-in-time event handling
- leakage-safe feature and label construction
- expanding-window walk-forward validation
- ablation between market-only, sentiment-only, and combined logistic baselines
- deterministic synthetic methodology validation
- a pre-specified historical evaluation on real SEC filing text (n = 197; see
  `publication/sentiment-study/`), which failed to demonstrate incremental predictive value

## What it does not currently demonstrate

- historical sentiment alpha (the historical evaluation is a null, and too small to rule out a small effect)
- production data ingestion
- economic performance claims
- model portability across assets or regimes

## Overlap / duplicate assessment

No specific named overlap candidates were supplied inside this repository-local review, and this rebuild does not archive, delete, rename, or modify any external repository or hub entry.

The safest current interpretation is:

- keep this repository if a dedicated sentiment-methodology example is useful in the portfolio
- do not present it as historical alpha research: the historical evaluation is a null

## Suggested hub update

Use language close to the following:

> **ML_Sentiment_Augmented_Price_Predictor** — leakage-safe harness and pre-specified historical study of whether SEC filing-text sentiment adds information beyond market-only features for short-horizon returns. Result: failure to demonstrate (combined-minus-market Δ log loss +0.0041, positive = worse; 95% CI [−0.0049, +0.0108], n = 197).

*(Updated 2026-10-27: the original wording, written before the historical study, described the repository as synthetic-only.)*


## Sibling repository names (2026-09-30)

Canonical code: **this repo** (`jmiaie/ML_Sentiment_Augmented_Price_Predictor`).

- A non-public name-hold twin — empty placeholder (initial commit only); not a fork of results.
- `ML_Sentiment_Augmented_Price_Predictor_public` — empty placeholder; not a portfolio showcase of this harness.

Hub narrative: treat as a chapter under `quant-research-portfolio-public`, not a free-standing alpha product. See also `docs/STATUS.md`.
