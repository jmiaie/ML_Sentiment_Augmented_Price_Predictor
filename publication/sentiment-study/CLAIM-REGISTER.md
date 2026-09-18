# CLAIM-REGISTER — D10-D (sentiment-study)

Every substantive claim this pack makes, with its evidence citation and its status.
Status vocabulary is fixed: **SUPPORTED** (measured, cited), **DISCLOSED**
(true and stated as a limitation), **NOT CLAIMED** (explicitly outside the pack),
**KNOWN LIMITATION** (a gap in the accepted evidence, disclosed not repaired).

Citations refer to `RESULT-SOURCE-MAP.md`.

| id | claim | evidence | status |
| --- | --- | --- | --- |
| CL-01 | 2025 primary headline Δ log loss (m3−m1) is `+0.0040902` | C-01 | SUPPORTED |
| CL-02 | 2025 primary bootstrap 95% interval includes zero | C-01 | SUPPORTED |
| CL-03 | 2025 secondary headline Δ log loss is `+0.0031305` | C-02 | SUPPORTED |
| CL-04 | 2025 secondary bootstrap 95% interval includes zero | C-02 | SUPPORTED |
| CL-05 | Bootstrap used 500 resamples at block size 10 | C-01, C-02 | SUPPORTED |
| CL-06 | Market-only was the lowest-log-loss model of the four in both 2025 targets | C-01, C-02 | SUPPORTED |
| CL-07 | The text-augmented model had higher log loss than market-only in all six period×target blocks | C-01–C-06 | SUPPORTED |
| CL-08 | The 2025 evaluation block is 197 rows across 12 issuers | C-01, C-02 | SUPPORTED |
| CL-09 | Development internal test block is 20 rows; 2024 validation block is 210 rows | C-03–C-06 | SUPPORTED |
| CL-10 | The development-block interval excludes zero; validation and 2025 intervals include zero | C-01–C-06 | SUPPORTED |
| CL-11 | `C` = 0.01 for every logistic model and both targets, read from the frozen configuration | C-07, C-08 | SUPPORTED |
| CL-12 | `C` sits at the lower edge of the pre-specified grid | C-08 | DISCLOSED |
| CL-13 | `C` was selected on pre-2025 data only and not re-selected on 2025 | C-07, C-08, C-13 | SUPPORTED |
| CL-14 | The 2025 evaluation ran exactly once, one invocation, covering both targets | C-01, C-02, C-11, C-15 | SUPPORTED |
| CL-15 | Training/evaluation row intersection is zero | C-01, C-02 | SUPPORTED |
| CL-16 | All training target windows end before the purge cutoff | C-01, C-02 | SUPPORTED |
| CL-17 | Embargo is 1 session (primary) and 5 sessions (secondary), applied by session distance | C-01, C-02, C-08 | SUPPORTED |
| CL-18 | Corpus: 2,349 filing events read, 2,296 frame rows, 12 issuers | C-01 | SUPPORTED |
| CL-19 | Form mix is 1,781 `8-K`, 391 `10-Q`, 124 `10-K` | C-01 | SUPPORTED |
| CL-20 | Rows dropped: 52 insufficient market history, 1 insufficient forward window, 0 otherwise | C-01 | SUPPORTED |
| CL-21 | Frozen-input verification passed before fitting, 2,361 filing files and 13 price files verified, both manifests `DATA FROZEN` | C-01, C-14 | SUPPORTED |
| CL-22 | The binding 2025 label is `PREVIOUSLY INSPECTED / HISTORICAL EVALUATION` | C-01, C-02, C-12 | SUPPORTED |
| CL-23 | The fold-delta distribution is identical element-for-element across the development and validation artifacts (90 elements each) despite different training sizes | C-03, C-05 | DISCLOSED |
| CL-24 | `walk_forward_fold_count` is 90 in every period, including one with 20 evaluation rows | C-01–C-06 | SUPPORTED |
| CL-25 | The 2025 fold-delta distribution is empty because the frozen-`C` path skips the tuner | C-01, C-02, C-13 | SUPPORTED |
| CL-26 | The configuration's named required report is absent at the accepted head | C-08 (requirement), repository at `9184eff7` (absence) | KNOWN LIMITATION |
| CL-27 | Three superseded v1 result artifacts remain, preserved and cited as sources nowhere | C-16, C-17, C-18 | SUPPORTED |
| CL-28 | The universe is a static present-day selection carrying survivorship/selection bias | C-08 | DISCLOSED |
| CL-29 | FinBERT was not used and is not confirmatory | C-08 | SUPPORTED |
| CL-30 | No risk-adjusted, excess-return or cost-adjusted performance measure was computed | C-01–C-08 | NOT CLAIMED |
| CL-31 | No tradeable strategy, position sizing or capacity result is asserted | C-01–C-08 | NOT CLAIMED |
| CL-32 | No causal mechanism is asserted for either the direction or the absence of an effect | C-01–C-08 | NOT CLAIMED |
| CL-33 | Results are not generalised beyond 12 large-cap US issuers, three filing forms, a dictionary representation, and 1- and 5-session horizons | C-08 | NOT CLAIMED |
| CL-34 | The pack modifies no accepted D9 artifact; artifact hashes at the accepted head equal the values this pack declares | C-01–C-18 | SUPPORTED |
| CL-35 | The pack performs no network access, no fitting, no C selection and no re-execution | C-15, pack scripts | SUPPORTED |
| CL-36 | The 2025 period is not described as a holdout of any kind | C-01, C-02, C-12 | SUPPORTED |

## Claim classes deliberately absent

No claim in this pack rests on a number that appears only in prose. Any claim whose
supporting value is not carried by a cited artifact is not listed above because it
is not made. If a reviewer finds a numeric statement in this pack and cannot find
it in the corresponding artifact, that is a defect and should be reported as one.
