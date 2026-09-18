# Does Point-in-Time Financial Text Add Incremental Predictive Information?

### A Historical Walk-Forward Study of SEC Filing-Derived and Market Features

Draft publication document. Lane D, Directive #9 / D10. Micap AI LLC.

---

## Abstract

We ask one question: does a logistic classifier given Loughran-McDonald
filing-text features plus market features predict a short-horizon excess-return
direction better than the same classifier given market features alone? The
comparison, its metric, its horizon and its regularisation were frozen before this
D9-D v2 historical-evaluation run was executed; the period itself remains
classified `PREVIOUSLY INSPECTED / HISTORICAL EVALUATION`, and had been inspected
before this run.

On the 2025 historical-evaluation block, the pre-specified comparison did not
demonstrate an improvement from adding filing-text features. For the primary target
(next-session excess-return direction) the combined model's log loss exceeded the
market-only model's by `+0.0040902` (block-bootstrap 95% interval
`[-0.0048506, +0.0108001]`, 500 resamples, block size 10). For the secondary
target (five-session excess-return direction) the corresponding difference was
`+0.0031305` (interval `[-0.0047166, +0.0118433]`). Both intervals include zero.
Both point estimates are positive, i.e. the text feature did not lower log loss.

The corresponding pre-2025 record runs in the same direction: on the development
internal test block the difference was `+0.02029985` with an interval excluding
zero, and on the 2024 validation block `+0.00405322` with an interval including
zero. Across every period and both targets, the model using text features was
never the best of the four models evaluated.

Three claims are deliberately **not** made. First, this study does not establish
incremental predictive information from filing text; it also does not establish
that filing text is uninformative, because the intervals are wide relative to the
effects at issue. Second, the 2025 period is **`PREVIOUSLY INSPECTED / HISTORICAL
EVALUATION`** — it is not a holdout, clean or otherwise, and must not be described
as one. Third, no risk-adjusted or excess-return performance measure is reported
anywhere in this pack, because no such quantity was computed.

## 1. Scope

This document reports what the frozen D9-D experiment produced. It is a
publication and communication artifact. It performs no fitting, no data
acquisition, no re-selection of any hyperparameter, and no re-execution of the
2025 evaluation. Every number below is read from an artifact whose SHA-256 is
recorded in `RESULT-SOURCE-MAP.md` and re-verified by `publication_pack.py check`.

## 2. Data

Twelve large, currently-listed issuers (AAPL, MSFT, AMZN, GOOGL, NVDA, JPM, XOM,
JNJ, PG, WMT, HD, KO) with SPY as market context. Filing text covers `10-K`,
`10-Q` and `8-K`. The corpus as recorded by the accepted artifact:

- 2,349 filing events read; 2,296 rows in the assembled frame; 12 issuers present.
- Form mix: 1,781 `8-K`, 391 `10-Q`, 124 `10-K`.
- Rows dropped: 52 for insufficient market history, 1 for an insufficient forward
  window, 0 duplicates, 0 empty texts, 0 missing price sessions.
- Frame span: `2015-04-02` to `2025-12-15`.

Frozen-input verification ran before any fitting and is recorded in each artifact:
2,361 filing files verified (2,349 of them text files), 13 price files verified,
12 issuers verified, both manifests reporting `DATA FROZEN`, freeze timestamp
`2026-09-17T20:46:08Z`, `verification_mode: frozen_bytes_sha256`.

Text features are derived from the Loughran-McDonald Master Dictionary
(`lm_master_dictionary_2026_v1`, resolved at runtime from the `pysentiment2`
package; not vendored here): negative, positive, uncertainty, litigious and
constraining fractions, net tone, and log word count. Market features are ten
column-strict trailing-only series (issuer 1/5/20-session returns, 20- and
60-session realised volatility, a 20-session volume z-score, and the SPY
equivalents). FinBERT is not used and is not confirmatory for anything here.

This universe is a static, present-day selection of large issuers. It carries
survivorship and selection bias, and the frozen configuration says so explicitly.
That bias is a property of the accepted evidence, disclosed rather than repaired —
repairing it would require reacquiring data, which D10 forbids.

## 3. Targets

- Primary: `next_session_excess_return_direction` — 1-session horizon, 1-session
  embargo, defined as issuer close-to-close return minus SPY close-to-close return
  being positive.
- Secondary: `five_session_forward_excess_return_direction` — 5-session horizon,
  5-session embargo, same excess-return construction.

Excess-over-SPY construction matters: a model cannot score by learning the market.
Target-aware period slicing is applied by a single shared code path so that a row
is admitted to a period only if its entire target window lies inside that period.

## 4. Models and the pre-specified comparison

Four models, all on the same feature/target pipeline:

- `model0_majority_baseline` — unconditional majority class; no regularisation.
- `model1_market_only` — logistic regression on market features.
- `model2_text_only` — logistic regression on dictionary-derived text features.
- `model3_market_text_combined` — logistic regression on both.

The headline comparison is `model3` minus `model1` in log loss, lower is better.
A positive difference therefore means the text features made the combined model
worse than market features alone. Balanced accuracy, Brier score and ROC AUC for
the logistic models are recorded in the artifacts; log loss carries the headline
because it was fixed as the headline metric before evaluation.

## 5. Evaluation geometry

Geometry is read from the artifacts, not asserted here:

- Training block `2015-04-02` → `2024-12-20`, 2,099 training rows for the
  primary target; 2,098 for the secondary target, whose 5-session embargo
  removes one further row.
- Purge cutoff session `2025-01-03` at ordinal 3,775 (primary target);
  `2024-12-27` at ordinal 3,771 (secondary target). Last training target end
  ordinal 3,768 is the same for both targets and sits before either cutoff.
- First effective evaluation session `2025-01-06` (ordinal 3,776) for both
  targets; evaluation block `2025-01-06` → `2025-12-15`, 197 evaluation rows,
  12 unique issuers (identical for both targets).
- Embargo: 1 session (primary), 5 sessions (secondary), applied by actual NYSE
  session distance rather than row count, because pooled multi-issuer event rows
  can share or span sessions.
- Training/evaluation row intersection: **0**. The artifact records
  `train_eval_intersection_count` = 0 and `train_windows_close_before_eval_start`
  = true.

## 6. Pre-specification and single execution

Regularisation `C` for all three logistic models, both targets, is `0.01`, taken
verbatim from `walk_forward.final_selected_c` in the frozen configuration
`configs/experiments/sentiment_historical_text_study_v2.yaml`
(SHA-256 `bd702bf5f58311b80ad68dc0fc9bb89bbcd11479ddd49c480e7fd6beacdd80f8`).
It was selected on purged-formation training plus the 2024 validation block only,
by `--select-final-c`; the selection wrote no other result artifact. The 2025
evaluation then refused to start unless every logistic model carried a value for
both targets, and never re-selected `C` on 2025 data
(`constraints.no_retune_after_freeze`).

`C = 0.01` sits at the lower edge of the pre-specified grid. That is disclosed as
a limitation. It is not permission to widen the grid after seeing results; doing
so would be exactly the kind of post-hoc tuning the freeze exists to prevent.

The 2025 historical evaluation was executed exactly once, in one invocation
covering both targets. The binding label for that period is
**`PREVIOUSLY INSPECTED / HISTORICAL EVALUATION`**.

## 7. Results — 2025 historical evaluation

Primary target, log loss by model (measured, unrounded in the artifact):

| model | log loss |
| --- | --- |
| `model0_majority_baseline` | 0.6911031425 |
| `model1_market_only` | 0.6910789538 |
| `model2_text_only` | 0.6943506742 |
| `model3_market_text_combined` | 0.6951691553 |

Headline difference `model3 − model1` = `+0.0040902`, block-bootstrap 95%
interval `[-0.0048506, +0.0108001]` (500 resamples, block 10). Difference in
balanced accuracy: `-0.0274725275`.

Secondary target:

| model | log loss |
| --- | --- |
| `model0_majority_baseline` | 0.6926514193 |
| `model1_market_only` | 0.6833713863 |
| `model2_text_only` | 0.6959961447 |
| `model3_market_text_combined` | 0.6865018901 |

Headline difference `model3 − model1` = `+0.0031305`, interval
`[-0.0047166, +0.0118433]`. Difference in balanced accuracy: `+0.0154282766`.

Reading, stated plainly:

1. The market-only model was the best of the four in **both** targets.
2. Adding text features moved log loss in the wrong direction in both targets,
   and removed most of the market-only model's margin over the majority baseline
   in the primary target.
3. Every interval includes zero, so a small negative, null or small positive
   contribution of the text features is all compatible with these data. The data
   discriminate between the competing models less sharply than the point
   estimates alone suggest.
4. This is a null/negative result and is reported as one. It is reported because
   it was pre-specified, not because it is agreeable.

## 8. Pre-2025 record

Same comparison, same metric, earlier blocks. Training sizes differ per block, so
these are not interchangeable replicates.

| block | target | Δ log loss (m3−m1) | 95% interval | n_eval | n_train |
| --- | --- | --- | --- | --- | --- |
| development internal test | primary | `+0.02029985` | `[+0.00545864, +0.03545251]` | 20 | 1,864 |
| development internal test | secondary | `+0.01558910` | `[+0.01066213, +0.05154456]` | 20 | 1,845 |
| 2024 validation | primary | `+0.00405322` | `[-0.00279798, +0.01162724]` | 210 | 1,889 |
| 2024 validation | secondary | `+0.00400243` | `[-0.00305854, +0.01036718]` | 210 | 1,889 |

Direction is consistent across all six block/target combinations in this pack: the
text-augmented model is worse on log loss. On the development internal test block
the interval excludes zero and the effect is largest; on the two larger evaluation
blocks the interval includes zero. The 20-row development test set is small and
its interval should not be treated as a stable estimate of anything.

## 9. Disclosures and limitations

1. **The reported fold-delta distribution is identical across the development and
   validation runs.** In both artifacts, `headline_fold_delta_log_loss_distribution`
   contains the same 90-element list, element for element, even though those two
   runs trained on different numbers of rows (1,864 and 1,889). A field that does
   not vary with its period's training geometry cannot be read as a period-specific
   statistic. The artifacts also record `walk_forward_fold_count` = 90 in all
   periods, including a block with 20 evaluation rows, which points the same way:
   this is formation geometry, not a per-period measurement. Disclosed as a
   provenance observation; the artifacts were not modified to "fix" it, because
   D10 forbids altering accepted evidence.
2. **The fold-delta distribution is empty in the 2025 artifacts.** Under the
   frozen-`C` path the tuner is skipped, so no per-fold deltas are computed. This
   is expected behaviour for a frozen-`C` evaluation, not missing output.
3. **`C` at the grid edge** (§6).
4. **Universe bias** (§2).
5. **Interval widths.** 197 evaluation rows for 2025, 20 for the development
   internal test block. Effects of this size are not resolvable at these sample
   sizes.
6. **The configuration's named required report, `research/historical-textual-signal-study.md`,
   does not exist at this HEAD.** The frozen configuration requires such a report;
   the file is absent (HTTP 404 against the repository at the accepted head). This
   pack is the first document to attempt it. No statement in this pack is sourced
   from that absent file, and its absence is not evidence about any result.
7. **v1 material is preserved and excluded.** Three superseded v1 result artifacts
   remain in the tree, hashed in `RESULT-SOURCE-MAP.md`, and are cited nowhere as
   a source. Their numbers are not reused here.
8. **Single seed.** `seed: 0`. No seed-robustness study was run, and none is
   claimed.

## 10. What this study does not claim

- It does not claim filing text is uninformative. Intervals include zero and are
  wide; that is compatible with a small real effect in either direction.
- It does not claim the market-only model is predictive in any tradeable sense.
  A log-loss ordering among four classifiers on 197 rows is not a strategy
  result, and no position sizing, cost model, capacity analysis or risk-adjusted
  return measure was computed anywhere in this experiment.
- It does not claim the 2025 period is a clean or independent holdout.
- It does not claim any causal mechanism — not for the absence of an effect, and
  not for the direction of the point estimates.
- It does not generalise beyond twelve large-cap US issuers, `10-K`/`10-Q`/`8-K`
  text, a Loughran-McDonald dictionary representation, and horizons of one and
  five sessions.

## 11. Reproduction and provenance

Everything in this pack is verifiable offline:

```
python publication/sentiment-study/scripts/publication_pack.py build     # regenerate tables + figure
python publication/sentiment-study/scripts/publication_pack.py check     # artifact hashes, reproducibility, citations
python publication/sentiment-study/scripts/publication_pack.py hashcheck # fail-closed hash integrity
python publication/sentiment-study/scripts/claim_crosscheck.py          # independent re-derivation of every claim
```

`build` performs no network access. `check`, `hashcheck` and the cross-check read
only local files. The CI workflow runs the same commands at full history depth.

Accepted evidence, for cross-reference:

- primary 2025 artifact — `6ed9197eb2d51cd68341a7fc1a9790eb61d6fa9ad10fda9e148a8da5120cca36`
- secondary 2025 artifact — `b207ed7a5c7923f6bdb0217bdc37963b5e81b37a2dc34788b0696ef822557d12`
- frozen configuration — `bd702bf5f58311b80ad68dc0fc9bb89bbcd11479ddd49c480e7fd6beacdd80f8`
- producing code head — `0fa6cfd3e124f36eea3383af7da2f71b6c43934f`

The 2025 artifacts record `0fa6cfd…` as their producing head; the pre-2025
artifacts record `e55cb09e8367053c24270743e8b8d96f5fe1e375`, theirs. Artifacts were
not rewritten, so the difference is expected and is not drift.
