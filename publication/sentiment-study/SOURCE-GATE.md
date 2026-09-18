# SOURCE-GATE — D10-D (sentiment-study)

The source gate records, in fixed fields, what this publication is sourced from and
what authorises it to exist. No program-defined schema for this document exists in
any reachable repository; the field list below was therefore authored for this pack
and is fixed once and for all here, so a reviewer can check the same 14 fields for
every lane rather than a differently-shaped document each time.

---

**Field 1 — Directive and lane.** Directive #9 (D9 research program) with D10
publication output. Lane D. This pack is the D10-D publication pack.

**Field 2 — Repository, branch and head.** `jmiaie/ML_Sentiment_Augmented_Price_Predictor`,
branch `publication/sentiment-study`, created from the accepted evidence head
`9184eff7571f8911f410632df8a06601c98311bc`. Draft pull request only; no merge.

**Field 3 — Accepted evidence input.** The single 2025 historical evaluation
produces exactly two artifacts:

- primary target — `results/historical_text_v2/sentiment_hist_text_v2_primary_historical_evaluation.json`,
  SHA-256 `6ed9197eb2d51cd68341a7fc1a9790eb61d6fa9ad10fda9e148a8da5120cca36`
- secondary target — `results/historical_text_v2/sentiment_hist_text_v2_secondary_historical_evaluation.json`,
  SHA-256 `b207ed7a5c7923f6bdb0217bdc37963b5e81b37a2dc34788b0696ef822557d12`

Both are unchanged by this pack and were not re-derived.

**Field 4 — Experiment identifier.** `sentiment_hist_text_v2`.

**Field 5 — Frozen configuration.** `configs/experiments/sentiment_historical_text_study_v2.yaml`,
SHA-256 `bd702bf5f58311b80ad68dc0fc9bb89bbcd11479ddd49c480e7fd6beacdd80f8`,
`config_status: frozen-final`. The configuration is not edited by this pack: the
2025 artifacts record this exact hash as the configuration they executed under, so
editing the file would break the artifacts' own provenance.

**Field 6 — Datasets.** Filings `sec_filings_12issuer_2015_2025_v1`, prices
`yf_sentiment_equities_daily_2015_2025_v1`, sentiment dictionary
`lm_master_dictionary_2026_v1`. Both manifests report `DATA FROZEN`; freeze
timestamp `2026-09-17T20:46:08Z`. Canonical dataset hashes
`3b2941870391b8584afaad46417cf3e6a0fe03509238fc6ac21958bb4c90f7f4` (filings) and
`6ca6e433983fb5f629d084d0964229d5b9c91cbee0756f686598bea6c90d4cb2` (prices) are
manifest-declared and are verified by requiring the exact value to be physically
present in the hash-verified manifest and in both hash-verified 2025 artifacts.
Raw filings text and price CSVs are not committed to this repository, so those two
hashes cannot be recomputed from the tree; that is a disclosure, not an oversight.

**Field 7 — Producing code head.** `0fa6cfd3e124f36eea3383af7da2f71b6c43934f`
(the commit whose tree generated the 2025 artifacts). The pre-2025 artifacts record
`e55cb09e8367053c24270743e8b8d96f5fe1e375`, their own producing head. Artifacts are
not rewritten, so the two heads legitimately differ by period.

**Field 8 — Empirical re-execution policy.** NONE. The 2025 evaluation was executed
exactly once. This pack performs no training, no fitting, no C selection, no data
acquisition, and no network access of any kind.

**Field 9 — Publication artifact set.** This pack: `SOURCE-GATE.md`,
`TECHNICAL-PAPER.md`, `CASE-STUDY.md`, `RESULT-SOURCE-MAP.md`, `CLAIM-REGISTER.md`,
`QUANT-RED-TEAM.md`, `CLAIM-RED-TEAM.md`, `CITATION-RED-TEAM.md`, `D10-STATUS.md`,
`reproducibility.json`, `scripts/publication_pack.py`, `scripts/claim_crosscheck.py`,
three generated tables and one generated figure.

**Field 10 — Verifier and gates.** `publication_pack.py check` (declared artifact
hashes, generated-byte reproducibility, citation resolution), `publication_pack.py
hashcheck` (hash-integrity, fail-closed, offline), `claim_crosscheck.py`
(independent re-derivation of the pack's claims from the artifacts), plus the
`publication-pack` CI workflow at full history depth and the repository's own
`ci` workflow (ruff, mypy, pytest). All gates fail closed.

**Field 11 — Program sign-off (REQUIRED).** *Directive #9 Final Four-Stream
Independent Program Audit (Grokbot, 2026-09-17 evening PT) — PROGRAM SIGN-OFF: YES;
P0=0; P1=0; HISTORICAL EMPIRICAL VALIDATION COMPLETE / ACCEPTED.* This is the
authorising sign-off. It is not contingent on any later central-hub issue edit, and
no such edit is made or requested by this pack.

**Field 12 — Prohibitions honoured.** No retraining, retuning, data reacquisition,
rerun of 2025, alteration of any accepted artifact, overwrite of superseded
evidence, hypothesis change, universe change, target change, historical-evaluation
label change, suppression of null or negative results, invented metric, invented
statistical claim, invented risk-adjusted or excess-return performance figure,
merge of any pull request, external publication, D11 work, modification of the
central `quant-research-portfolio` repository, modification of central Issue #3, or
update to any central hub headline or performance claim. The stale sentiment pull
request #4 is untouched.

**Field 13 — Known limitations and gaps.** Static present-day-selected 12-issuer
universe carrying survivorship/selection bias; a 197-row 2025 evaluation set, so
interval widths are material; the frozen configuration's C sits at the lower edge
of the pre-specified grid (a disclosure, never a licence to widen the grid
post hoc); the configuration's named required report
`research/historical-textual-signal-study.md` is absent at this HEAD and is not a
source for any prose here; the reported fold-delta distributions are identical
across the development and validation runs, so they must not be read as
period-specific statistics; v1 material is preserved, non-conforming, and excluded
from every table.

**Field 14 — Reviewer instructions.** Verify the two 2025 artifact hashes
independently; re-derive the headline log-loss difference from `period_metrics` and
compare it to the recorded headline value; confirm the bootstrap interval brackets
zero for both targets; confirm the binding label below appears verbatim; confirm the
training/evaluation geometry (zero row intersection, purge before evaluation start,
embargo of one session) from the artifacts rather than from this prose.

**Binding 2025 label:** `PREVIOUSLY INSPECTED / HISTORICAL EVALUATION`. It is not an
untouched holdout and must not be described as one.
