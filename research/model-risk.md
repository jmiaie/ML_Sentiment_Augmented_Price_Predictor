# Model risk and leakage controls

## Status

This repository currently validates methodology on deterministic synthetic data only. Historical results pending reproducible point-in-time dataset.

## Principal risks

1. **Timestamp integrity risk**  
   Publication, ingestion, and availability timestamps must remain ordered and timezone-aware. Any downstream dataset that collapses these fields into a single naive timestamp can reintroduce look-ahead bias.

2. **Session-mapping risk**  
   Sentiment events are mapped using explicit US regular-hours rules. Real production datasets may require richer holiday/half-day handling than the deterministic calendar used here for tests.

3. **Feature leakage risk**  
   Sentiment aggregation is cutoff-based and strictly filters on `effective_trading_timestamp <= decision_timestamp`. Trailing market features are backward-looking only, and tests verify that adding future events or extreme future prices does not alter prior feature rows.

4. **Label overlap risk**  
   Walk-forward planning uses a horizon-driven gap/embargo between train and validation/test windows so overlapping forward labels cannot leak across splits.

5. **Calibration and class-balance risk**  
   Probability metrics and calibration-table data are included, but synthetic class balance is not a substitute for real-market class balance or regime changes.

6. **Economic over-interpretation risk**  
   The rebuild intentionally avoids headline economic claims. No Sharpe, alpha, or monetization claims should be made from the committed artifacts.

## Controls implemented in this repository

- timezone-aware event schema
- explicit `effective_trading_timestamp`
- deterministic session-alignment rules with tests for after-close and weekend/holiday behavior
- decision-time-safe next-session primary target and documented two-session secondary target
- expanding-window validation plus untouched chronological test window
- preprocessing fit only within the relevant train windows via scikit-learn pipelines

## Required future evidence before stronger claims

- reproducible point-in-time sentiment source with licensing clarity
- versioned market dataset with documented corporate-action handling
- broader asset and regime coverage
- explicit economic assumptions if cost-aware evaluation is added later
