# Post-publication notes on the historical study (2026-11-14)

These notes qualify statements in the publication pack (`publication/sentiment-study/`). The
frozen pack, result files, source code and hashes are **unchanged**, and nothing was re-run. The
headline conclusion stands: the study **failed to demonstrate** that filing-text features add
predictive value beyond market-only features.

## 1. The comparison model itself had no skill on the primary target

On the 2025 evaluation (197 events, primary target), the market-only model's log loss is 0.69108
and the majority-class baseline's is 0.69110 (`publication/sentiment-study/tables/period-metrics.csv`).
The market-only model's ROC AUC is 0.51. So "text adds nothing beyond market features" is, on this
target, "text adds nothing beyond a model that is itself no better than a constant". On the secondary
target the market-only model does beat the baseline (0.6834 vs 0.6927).

## 2. The "interval excluding zero" on the development internal test block rests on 20 rows

The technical paper's abstract cites a development internal-test difference of +0.0203 "with an
interval excluding zero". That block has **20 evaluation rows**. With the study's block size of 10,
the bootstrap resamples two blocks. An interval from two blocks is not reliable evidence in either
direction, and it should not be read as support for "text makes the model worse".

## 3. Limits of the block bootstrap (`_block_bootstrap_delta_ci`)

- If a block has no more rows than the block size, every resample is the same sample and the
  interval has zero width. No evaluated block in the study is that small, but the function does
  not guard against it.
- Blocks are consecutive **rows** (filing events in session order), not sessions, so several
  events on one day can fall into different blocks.
- A separate power study (open pull request "power / minimum-detectable-effect study") finds the
  interval rejects a true null about 7.8% of the time at n = 197, against a nominal 5%. The
  published 2025 interval includes zero, so this does not change the conclusion; it means the
  interval is somewhat too narrow.

## 4. "Direction is consistent across all six block/target combinations"

Adding uninformative features to a regularised logistic model is expected to raise out-of-sample
log loss slightly. A consistent small positive difference is what the null hypothesis predicts, so
the consistency is not additional evidence about filing text.

## 5. Calendar helper behaviour outside its date range

`NyseCalendar.next_trading_day` returns the first session for a date before the schedule start,
and `last_session_on_or_before` returns the last session for a date after the schedule end. Neither
raises. All study dates lie inside the schedule, so results are unaffected, but the helpers should
not be reused outside that range without a guard.

## 6. Scope reminder

Twelve mega-cap issuers and a one-day horizon are an unfavourable place to look for a filing-text
effect, and the sample can detect only an improvement of about 1.6% in log loss or larger. A
larger sample (hundreds of issuers) is needed before anything can be ruled out.
