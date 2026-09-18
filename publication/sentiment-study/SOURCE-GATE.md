# SOURCE-GATE — D10-D (sentiment-study)

The fourteen fields below are the authoritative source-gate field contract defined by
Directive #10. They are populated here from accepted D9-D evidence. Sections printed
after the fourteen fields are pack detail, not part of the field contract, and are
not numbered as fields.

---

**Field 1 — Repository.** `jmiaie/ML_Sentiment_Augmented_Price_Predictor`.

**Field 2 — Source PR / branch.** The accepted evidence branch is `research/d9d-cap-lift`,
whose tip is the accepted HEAD named in Field 3. This pack is a separate publication
branch, `publication/sentiment-study`, created from that tip, and its output is a draft
pull request only. The stale sentiment pull request #4 is NOT the source branch and
remains untouched.

**Field 3 — Accepted HEAD.** `9184eff7571f8911f410632df8a06601c98311bc`.

**Field 4 — Experiment ID.** `sentiment_hist_text_v2`.

**Field 5 — Dataset IDs.** `sec_filings_12issuer_2015_2025_v1` (SEC filing text,
`10-K`/`10-Q`/`8-K`, twelve issuers); `yf_sentiment_equities_daily_2015_2025_v1` (daily
equity and SPY prices); and the frozen language input `lm_master_dictionary_2026_v1`,
the Loughran-McDonald Master Dictionary, resolved at runtime from the pinned
`pysentiment2` package and not vendored in this repository.

**Field 6 — Dataset SHA / frozen identity.** Filings canonical
`3b2941870391b8584afaad46417cf3e6a0fe03509238fc6ac21958bb4c90f7f4`;
prices canonical
`6ca6e433983fb5f629d084d0964229d5b9c91cbee0756f686598bea6c90d4cb2`.
Both are manifest-declared: each manifest reports `DATA FROZEN` at freeze timestamp
`2026-09-17T20:46:08Z`, and each canonical is verified by requiring the exact value to
be physically present in the hash-verified manifest for that dataset and in both
hash-verified 2025 artifacts. Raw filing text and price CSVs are not committed to this
repository, so the two canonicals cannot be recomputed from the tree; that is a
disclosure, not an oversight.

**Field 7 — 2025 / holdout status.** `PREVIOUSLY INSPECTED / HISTORICAL EVALUATION`.
The period is not a holdout of any kind and must not be described as one.

**Field 8 — Final config SHA.** `configs/experiments/sentiment_historical_text_study_v2.yaml`,
SHA-256 `bd702bf5f58311b80ad68dc0fc9bb89bbcd11479ddd49c480e7fd6beacdd80f8`,
`config_status: frozen-final`. The configuration is not edited by this pack: the 2025
artifacts record this exact hash as the configuration they executed under, so editing
the file would break the artifacts' own provenance.

**Field 9 — Primary artifact path.**
`results/historical_text_v2/sentiment_hist_text_v2_primary_historical_evaluation.json`,
secondary target
`results/historical_text_v2/sentiment_hist_text_v2_secondary_historical_evaluation.json`.

**Field 10 — Result artifact SHA.** Primary
`6ed9197eb2d51cd68341a7fc1a9790eb61d6fa9ad10fda9e148a8da5120cca36`;
secondary
`b207ed7a5c7923f6bdb0217bdc37963b5e81b37a2dc34788b0696ef822557d12`.
Both are unchanged by this pack and were not re-derived.

**Field 11 — Independent review status.** *Directive #9 Final Four-Stream Independent
Program Audit (Grokbot, 2026-09-17 evening PT) — PROGRAM SIGN-OFF: YES; P0=0; P1=0;
HISTORICAL EMPIRICAL VALIDATION COMPLETE / ACCEPTED.* This is an EXTERNAL program-audit
citation, accepted by Phase 0 as the authorising sign-off. It is not something this
publication pack independently re-derived — the pack re-derives only the numbers it
prints. It is not contingent on any later central-hub issue edit, and no such edit is
made or requested by this pack.

**Field 12 — Primary finding.** On the 2025 historical-evaluation block the primary
target point estimate is the `model3 − model1` log-loss delta `+0.0040902`, with a
block-bootstrap 95% interval of `[-0.0048506, +0.0108001]` and a balanced-accuracy
delta of `-0.0274725275`. Secondary target: delta `+0.0031305`, interval
`[-0.0047166, +0.0118433]`, balanced-accuracy delta `+0.0154282766`.

**Field 13 — Primary null / negative finding.** Loughran-McDonald filing-text features
did not demonstrate incremental predictive value beyond market-only features under the
pre-specified headline log-loss comparison. This is a failure to demonstrate, not a
demonstration of absence, and it is not strengthened anywhere in this pack: filing text
is not claimed to be useless, text is not claimed to be harmful, market-only is not
claimed to be proven predictive, and a zero effect is not claimed to be proven.

**Field 14 — Primary limitation.** At minimum: (a) the 2025 period is
`PREVIOUSLY INSPECTED / HISTORICAL EVALUATION`, not a holdout; (b) n=197 in 2025 and the
uncertainty intervals are wide, so effects of this size are not resolvable at this
sample size; (c) the universe is a static, present-day-selected twelve-issuer set
carrying survivorship and selection bias; (d) `C` sits at the lower edge of the
pre-specified grid, which is a disclosure and never a licence to widen the grid after
seeing results; (e) the frozen configuration's named required report
`research/historical-textual-signal-study.md` is absent at the accepted HEAD, so it is a
source for no prose in this pack.

---

## Pack detail — not part of the fourteen-field contract

**Producing code head.** `0fa6cfd3e124f36eea3383af7da2f71b6c43934f`, the commit whose
tree generated the 2025 artifacts. The pre-2025 artifacts record
`e55cb09e8367053c24270743e8b8d96f5fe1e375`, their own producing head. Artifacts are not
rewritten, so the two heads legitimately differ by period.

**Empirical re-execution policy.** NONE. The 2025 evaluation was executed exactly once.
This pack performs no training, no fitting, no C selection, no data acquisition, and no
network access of any kind.

**Publication artifact set.** `SOURCE-GATE.md`, `TECHNICAL-PAPER.md`, `CASE-STUDY.md`,
`RESULT-SOURCE-MAP.md`, `CLAIM-REGISTER.md`, `QUANT-RED-TEAM.md`, `CLAIM-RED-TEAM.md`,
`CITATION-RED-TEAM.md`, `D10-STATUS.md`, `reproducibility.json`,
`scripts/publication_pack.py`, `scripts/claim_crosscheck.py`, three generated tables and
one generated figure.

**Verifiers and gates.** `publication_pack.py check` (declared artifact hashes, generated-byte
reproducibility, citation resolution), `publication_pack.py hashcheck` (hash-integrity,
fail-closed, offline), `claim_crosscheck.py` (independent re-derivation of the pack's
claims from the artifacts, including every quantitative cell printed in §7 and §8 of the
paper), the `--selftest` regression set for the overclaim guard, plus the
`publication-pack` CI workflow at full history depth and the repository's own `ci`
workflow (ruff, mypy, pytest). All gates fail closed.

**Prohibitions honoured.** No retraining, retuning, data reacquisition, rerun of 2025,
alteration of any accepted artifact, overwrite of superseded evidence, hypothesis change,
universe change, target change, historical-evaluation label change, suppression of null or
negative results, invented metric, invented statistical claim, invented risk-adjusted or
excess-return performance figure, merge of any pull request, external publication, D11
work, modification of the central `quant-research-portfolio` repository, modification of
central Issue #3, or update to any central hub headline or performance claim. The stale
sentiment pull request #4 is untouched.

**Reviewer instructions.** Verify the two 2025 artifact hashes independently; re-derive
the headline log-loss difference from `period_metrics` and compare it to the recorded
headline value; confirm the bootstrap interval brackets zero for both targets; confirm the
binding label below appears verbatim; confirm the training/evaluation geometry (zero row
intersection, purge before evaluation start, embargo of one session) from the artifacts
rather than from this prose; and re-run `claim_crosscheck.py` to confirm that every
quantitative cell printed in §7 and §8 of the paper is derived from an artifact.

**Binding 2025 label:** `PREVIOUSLY INSPECTED / HISTORICAL EVALUATION`. It is not an
untouched holdout and must not be described as one.
