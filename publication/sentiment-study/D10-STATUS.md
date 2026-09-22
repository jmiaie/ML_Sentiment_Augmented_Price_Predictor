# D10-STATUS — lane D (sentiment-study)

**Lane status: READY FOR INDEPENDENT D10 REVIEW.**
No merge. No external publication. No D11. No central control-plane edits.

*(Authoring-phase status — 2026-09-18. Superseded as a statement of current state: the pack is now integrated on `main`. See "Post-review integration status" at the top of this file.)*

## Post-review integration status

This publication pack was originally authored and reviewed under a
no-merge / stop-at-independent-review instruction. That language is preserved
below as a historical record of the authoring phase.

The pack has subsequently been integrated into `main`. This integration does
not, by itself, constitute Directive #10 program sign-off.

Current lifecycle status:
INTEGRATED ON MAIN / FINAL D10 PROGRAM SIGN-OFF PENDING.

| Integration record | Value |
| --- | --- |
| Accepted D9 head | `9184eff7571f8911f410632df8a06601c98311bc` (evidence head) — accepted producing-code head `0fa6cfd3e124f36eea3383af7da2f71b6c43934f` |
| Cleared publication head / accepted publication ancestor | `457586331f1dcdf92dee312a3cae623542406364` (cleared ancestor; PR #6 head `3972c43941219a47415bd621aff19bd8a576e9af`) |
| `main` head at the reconciliation baseline (frozen 2026-09-21; a reference point, not a permanently-current value — verify with `git ls-remote <repo> refs/heads/main`) |
| Integration path | Pack authored on `publication/sentiment-study`; PR **#5** merged it into `research/d9d-cap-lift` (`beff3d1b7c`); PR **#4** merged the D9-D study branch into `main` (`0cafe8ef`); PR **#6** merged `publication/sentiment-study` → `main` after conflict resolution (`77a2390`, 2026-09-18T20:32:14Z). `main`'s head **is** that merge commit. |
| Relevant pull requests | #4 (D9-D study, merged), #5 (pack, merged), #6 (`main` integration, merged); #7, #2 remain open (PR hygiene inventory) |
| Exact-head CI evidence | At exact `main` merge head `77a2390c`: `ci` run `35391992114` (success) — https://github.com/jmiaie/ML_Sentiment_Augmented_Price_Predictor/actions/runs/35391992114 ; `publication-pack` run `35391992514` (success) — https://github.com/jmiaie/ML_Sentiment_Augmented_Price_Predictor/actions/runs/35391992514 . |
| Exact-head CI evidence — remediation branch | `reconcile/d10-d-lifecycle`. Both workflows run on every push to this branch, so the current head's runs are listed at https://github.com/jmiaie/ML_Sentiment_Augmented_Price_Predictor/actions?query=branch%3Areconcile%2Fd10-d-lifecycle (this documentation-only push triggers both). Most recent completed runs, at commit `37ff44daca` — the commit immediately preceding this edit: `ci` run `35660934496` (success) — https://github.com/jmiaie/ML_Sentiment_Augmented_Price_Predictor/actions/runs/35660934496 ; `publication-pack` run `35660934499` (success) — https://github.com/jmiaie/ML_Sentiment_Augmented_Price_Predictor/actions/runs/35660934499 . |
| Diff from accepted D9 is publication-only | **No — publication pack plus D9-D producing-path code.** `git diff --name-status 9184eff7 77a2390c` yields the pack and `.github/workflows/publication-pack.yml`, **plus** `scripts/acquire_edgar_8k_yf_megacap_daily.py` (M), `src/quant_sentiment/edgar_filings.py` / `hashing.py` / `market_data_io.py` (A), `src/quant_sentiment/sec_http.py` (M), `tests/test_sec_http.py` (A), `README.md` (M) and `src/quant_sentiment.egg-info/*` (M) — attributed to commit `ca185e4` (“D9-D: pre-registration fix, full v1 decoupling, env-sourced SEC_USER_AGENT, provenance”, PR #4). **No file under `results/`, `configs/` or `data/` changed**: no accepted result artifact, configuration, dataset manifest, experiment identity, or ledger row was altered, and no rerun was performed. |
| Disclosed integration nuance | (1) The pack directory at `main` is **not** byte-identical to the cleared ancestor `457586331`: `git diff 457586331 77a2390c -- publication/sentiment-study/` returns exactly one file, `scripts/publication_pack.py`, from commit `24259e8` (“lint: reflow pack script to line-length=100 (formatting only)”, 5 insertions / 2 deletions, both re-wraps of existing string literals) — no semantic change to any hash, citation, or regeneration gate. PR #6's merge message states the pack directory remained byte-identical to the cleared tip: that holds for the merge *resolution*, but not literally for the final pack directory versus `457586331` as of `24259e8`. (2) `main` joins two D9-D lines — the evidence line (`0fa6cfd3` → `9184eff7`) and the producing-code/decoupling line (`ca185e4`, PR #4). |

**Current program state.** D9: COMPLETE / ACCEPTED. D10: TECHNICALLY
INTEGRATED / FORMAL SIGN-OFF PENDING. D11: PARTIALLY STARTED THROUGH THE
PUBLIC HUB / NOT FORMALLY ACTIVATED. D12: DRAFTED / BLOCKED BY D11 HIRING
EVIDENCE. D13: DRAFTED / NOT YET JUSTIFIED.

**Final D10 program sign-off remains PENDING.** No authoritative
`DIRECTIVE #10 PUBLICATION PACK SIGN-OFF: YES` has been issued for this pack.
A D9 program sign-off is not a D10 program sign-off. This section records
integration state only: it is not a sign-off, and it does not strengthen,
weaken, or restate any finding, number, or claim in the pack.

### How to read the rest of this directory

Every "no merge", "no pull request merged", "draft PR only", "not on `main`",
"not from `main`", "no external publication", and "READY FOR INDEPENDENT
(D10) REVIEW" statement preserved below, or elsewhere in this directory, is
**authoring-phase language** kept deliberately as the contemporaneous record
(append-only history; the historical record is not rewritten). Where such a
statement could be read as describing the *current* lifecycle state, this
section supersedes it; the statement itself is left unedited. The
machine-readable `reproducibility.json` field `merge` is likewise left
byte-unchanged on purpose, so the pack's own hash and regeneration gates stay
valid at the recorded tip.

*Repository visibility note:* the host repository is public, so this pack is
world-readable on `main`. No PyPI/npm release, website deployment, or other
external-service publication was performed.

---

*Post-review integration section added 2026-09-21 as documentation-only
reconciliation. No empirical artifact, configuration, dataset manifest,
experiment identity, ledger row, number, or finding was changed; no
rerun, retune, or reacquisition was performed.*

## What this lane produced

A publication pack for the accepted D9-D sentiment/filing-text evidence, on branch
`publication/sentiment-study`, which was created from the exact accepted evidence
head. The pack is draft-pull-request material only.

Contents: `SOURCE-GATE.md`, `TECHNICAL-PAPER.md`, `CASE-STUDY.md`,
`RESULT-SOURCE-MAP.md`, `CLAIM-REGISTER.md`, `QUANT-RED-TEAM.md`,
`CLAIM-RED-TEAM.md`, `CITATION-RED-TEAM.md`, `D10-STATUS.md`,
`reproducibility.json`, two scripts, three generated tables and one generated
figure.

## Gates applied

All gates are offline and fail closed. Exact commands and their results are in the
pull request's CI run and in the return package; the counts are deliberately not
restated here, because a document that quotes its own verifier's output is a
document that changes the thing it is measuring.

| gate | what it verifies | result |
| --- | --- | --- |
| `publication_pack.py build` | tables and figure are regenerated from accepted artifacts only, no network | PASS |
| `publication_pack.py check` | declared artifact hashes match disk, generated bytes are reproducible, every citation resolves | PASS |
| `publication_pack.py hashcheck` | every published hash is either a declared source artifact, a generated file, or a manifest-anchored canonical value — fail-closed | PASS |
| `claim_crosscheck.py` | every pack claim is re-derived independently of the generator | PASS |
| repository `ci` (ruff, mypy, pytest) | the pack's scripts satisfy the repository's own quality gates | PASS |

`hashcheck` rejects a hash that is not anchored in a named, hash-verified file. This
is the specific failure mode it exists to catch: a hash whose tail was written by
hand rather than measured. Both the generator and the cross-check were exercised
against a deliberately fabricated value to confirm the gate fails closed.

### Self-caught defects during authoring

Recorded so a reviewer does not have to rediscover them:

1. The citation table initially carried four generated-file hashes that were typed
   rather than measured. They were replaced with measured values, and `hashcheck`
   now rejects any unanchored hash mechanically.
2. The hash-anchoring check initially referenced its failure list before that list
   was initialised, so it would have raised instead of reporting. Fixed; a negative
   test confirms the gate now fails closed and reports cleanly.
3. The overclaim guard was first written as plain negation-aware substring
   matching, then found to raise a false positive on a quoted red-team attack.
   It was rebuilt with QUOTE-EXEMPT handling: a token entirely inside a balanced
   double-quoted span is treated as quoted text being named or attacked, while
   ordinary prose stays strictly negation-governed. It ships with `--selftest`
   and six regression cases, including the trap case: a stray "not" elsewhere in
   the sentence must not whitelist a real overclaim.
4. The development-block headline delta was first written into the paper as
   `+0.0202998`. The measured value is `+0.02029985`. The cross-checker's prose
   assertion caught the disagreement, and the table and abstract were regenerated
   from the artifact rather than corrected by hand.
5. The cross-checker read `max_train_target_end_ordinal` from the wrong nesting
   level and raised a `KeyError` on its first complete run. Nothing had been
   passing silently: the assertion was never reached. Fixed to read the ordinal
   from the key-metrics level.
6. The cross-checker asserted that form counts sum to filing events read. The
   correct model is that form counts sum to frame rows, 2,296, which equal events
   read, 2,349, minus every documented skip. Both numbers were already right in
   the paper; the checker's arithmetic was wrong and was fixed.

## Empirical status

- The 2025 evaluation was executed exactly once, before this pack existed, in one
  invocation covering both targets. This lane re-executed nothing.
- No accepted artifact was modified. The declared hashes of the accepted evidence
  equal the bytes on disk at the accepted head, verified by `check`.
- No configuration, hypothesis, universe, target or period label was changed.
- The binding 2025 label remains `PREVIOUSLY INSPECTED / HISTORICAL EVALUATION`.

## Findings by severity

- **P0: none.**
- **P1: none.**
- **P2 (disclosed, not repaired):**
  1. The reported fold-delta distribution is identical element-for-element across
     the development and validation artifacts despite different training sizes, and
     `walk_forward_fold_count` is reported as a formation plan rather than a
     per-period measurement. Disclosed as a provenance observation. Artifacts were
     not modified to correct it.
  2. The frozen configuration's `C` sits at the lower edge of its grid. Disclosed
     as a limitation, with the explicit note that this is not a licence to widen
     the grid after the fact.
  3. The configuration's named required report is absent at the accepted head. This
     pack is the first attempt at it; the absence is disclosed and no prose here is
     sourced from it.
  4. The 2025 evaluation block is 197 rows, so the intervals are wide relative to
     the effects under discussion.

## Independent audit remediation (2026-09-18)

An independent audit of the live draft pull request found one stale document
contract and four wording and threshold defects. All five are repaired here. The
audit is recorded because an audit trail that shows only the wins is not one.

1. **P1-1 — the source gate used an invented field list.** `SOURCE-GATE.md` had
   claimed no program-defined schema existed and had authored its own fourteen
   fields. It now uses the authoritative fourteen-field contract defined by
   Directive #10, and the claim that no schema exists is removed. The former extra
   sections are retained as unnumbered pack detail after the fourteen fields.
2. **P1-2 — wording implied the 2025 period was fresh.** `CASE-STUDY.md` said the
   specification was fixed "before 2025 was touched", and that the result came from
   a comparison fixed "before the evaluation period was read". Both are replaced
   with study-specific execution wording that scopes the freeze to this v2
   evaluation run while the period stays classified `PREVIOUSLY INSPECTED /
   HISTORICAL EVALUATION`. The paper's abstract is scoped the same way.
3. **P1-3 — the abstract overstated the null.** "On the 2025 evaluation period the
   answer is no" is replaced by the pre-specified-comparison wording: the
   comparison did not demonstrate an improvement from adding filing-text features.
   The exact deltas and the zero-spanning intervals follow unchanged.
4. **P2-1 — an undefined threshold.** `CASE-STUDY.md` described what the intervals
   rule out as "a large, reliable improvement", but this study defines no
   minimum-effect threshold, so the word was doing substantive work after the fact.
   The sentence now states the compatibility claim and the actual bounds.
5. **P2-2 — a stale red-team finding.** `CLAIM-RED-TEAM.md` CRT-4 still said the §7
   and §8 values were not independently contract-checked. Resolved by extending the
   check rather than by restating the gap: `claim_crosscheck.py` now compares every
   quantitative cell printed in §7 and §8 — the model log losses, the headline
   delta, both interval bounds, the balanced-accuracy delta, `n_eval` and `n_train`
   — against the accepted artifacts.

## Prohibitions — confirmations

- No retraining, retuning, data reacquisition, or re-execution of the 2025
  evaluation. No change to hypotheses, universes, targets or period labels.
- No accepted D9 artifact altered; superseded evidence preserved as-is.
- No metric, statistical claim, or performance figure invented. Null results
  reported as null. No risk-adjusted or excess-return performance number is
  reported, because none was computed.
- No pull request merged. No external publication. No D11 work started.
- The central `quant-research-portfolio` repository and central Issue #3 are
  untouched, as is the stale sentiment pull request #4. The other lane's
  publications were not touched.

## For the independent reviewer

1. Re-hash the two accepted 2025 artifacts and compare against the declared values
   in `SOURCE-GATE.md` and `RESULT-SOURCE-MAP.md`.
2. Re-derive the headline log-loss differences from `period_metrics` in the
   artifacts and compare with the paper's stated values.
3. Confirm both intervals include zero and that the pack says so without softening
   it.
4. Confirm the training/evaluation geometry — zero row intersection, training
   target windows ending before the purge cutoff, session-distance embargo — from
   the artifacts directly, not from this prose.
5. Run the four commands in `TECHNICAL-PAPER.md` §11 and confirm all four gates
   pass with no network access.
6. Confirm the fold-distribution carry-over disclosure (P2.1) is present and is
   not described as a defect that was fixed.
7. Confirm the binding 2025 label appears verbatim and is not softened anywhere.

A new specific defect, demonstrated from live artifacts, reopens this lane. Style
preferences and additional wish-list analyses do not.
