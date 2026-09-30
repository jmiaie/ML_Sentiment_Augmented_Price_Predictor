# Status — ML_Sentiment_Augmented_Price_Predictor

**Updated:** 2026-09-30 (PT)  
**Maturity:** MVP research harness (not a live trading system)

## Canonical home

**This repository** (`jmiaie/ML_Sentiment_Augmented_Price_Predictor`) is the **canonical** code and results tree.

Sibling placeholders (do **not** treat as products):

| Repo | Role | Notes |
|------|------|-------|
| [`ML_Sentiment_Augmented_Price_Predictor_private`](https://github.com/jmiaie/ML_Sentiment_Augmented_Price_Predictor_private) | Empty name-hold (Jan 2026 initial commit only) | size ≈ 0; no methodology code |
| [`ML_Sentiment_Augmented_Price_Predictor_public`](https://github.com/jmiaie/ML_Sentiment_Augmented_Price_Predictor_public) | Empty name-hold | size ≈ 0; no methodology code |

Portfolio narrative home for quant chapters: [`quant-research-portfolio`](https://github.com/jmiaie/quant-research-portfolio) (private hub). This repo is a methodology chapter, not a standalone alpha product.

## What is proven here

- Leakage-safe point-in-time schema + walk-forward ablations
- Deterministic **synthetic** methodology validation (CI / offline)
- One pre-registered historical text study (SEC EDGAR lexicon features) that **did not** find measurable incremental predictive value over market-only features — see root README Result section and `results/historical_text_v2/`

## What is **not** claimed

- Live trading performance or “sentiment alpha”
- That the repository name implies a shipped price predictor
- That empty `_private` / `_public` twins contain alternate results

## Offline quickstart

```bash
python -m pip install -e ".[dev]"
pytest -m "not slow"
python -m quant_sentiment.synthetic_validation \
  --output artifacts/synthetic_methodology_validation.json
```

No paid APIs, network downloads, or GPU required for the synthetic path. Optional `.[data]` extras are for offline data tooling only when explicitly run.

## Next actions

1. Keep twins archived or README-pointer-only (avoid dual maintenance)
2. Optional: one-line hub entry under quant-research-portfolio linking this RESULT
3. Leave draft admin PR #10 closed-or-draft unless Jeff wants formal D10 sign-off
