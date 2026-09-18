# CLAIM-RED-TEAM — D10-D (sentiment-study)

Adversarial review of what this pack *says*, as distinct from what it measured. The
job here is to find a sentence that outruns its evidence, a label that got softened,
or a number that appears in prose but in no artifact.

Disposition vocabulary: **UPHELD**, **REPAIRED**, **REJECTED**.

---

**CRT-1 — "Text-augmented prediction failed." Is that an overclaim?**
Yes, in the direction of drama, and the pack avoids it. The paper reports a
measured log-loss difference with an interval that includes zero, and states that
the data do not establish incremental predictive information from filing text while
also not establishing that the text is uninformative. Disposition: **REJECTED** as
wording; the measured formulation is used throughout.

**CRT-2 — "Does the pack anywhere claim the text features are harmful? Or that
sentiment never works?"**
No such sentence exists. The pack states that the combined model's log loss was
higher and that the point estimates run that way in every block, then declines the
inference: a higher log loss on 197 rows with a zero-spanning interval is not
evidence of harm. Disposition: **REJECTED**.

**CRT-3 — "Is the 2025 period described as an untouched holdout, or as pristine?"**
This pack does not assert an untouched holdout, and never labels it pristine. It
is described as `PREVIOUSLY INSPECTED / HISTORICAL EVALUATION` in the source gate,
the paper, the status document and the claim register, and the pack states
explicitly that the period is not a holdout of any kind. An automated guard in the
cross-checker fails the build when an overclaim token appears in unquoted prose
without a negation directly governing it.
Disposition: **REJECTED**, with a mechanical guard rather than a promise.

**CRT-4 — "Does any prose number fail to appear in an artifact?"**
This was checked the hard way. `claim_crosscheck.py` re-derives the pack's claims
from the artifacts rather than from the generator's intermediates, and the
citation table is re-hashed by `check`. One class of prose number is *not*
independently contract-checked: the log-loss values in §7's two small tables and
the pre-2025 block values in §8. Those were read from the artifacts when the
document was written, but nothing mechanical prevents a later edit from changing
them. Disposition: **UPHELD as a residual gap**, disclosed here rather than left
for a reviewer to find.

**CRT-5 — "The headline values in §7: are they re-derived or copied?"**
Re-derived. The cross-checker reads the headline log-loss difference out of each
2025 artifact and asserts that the paper states the measured value rounded to eight
decimals, in addition to asserting that the paper carries both full artifact
hashes, the frozen config hash, and the binding period label verbatim. A mismatch
fails the build. Disposition: **REJECTED** as a defect.

**CRT-6 — "Is 'the market-only model was best of the four' stated as a finding
about markets?"**
It is stated as an ordering of four log-loss values on one 197-row block, and §10
states that this is not a strategy result and that no tradeable claim is made.
Disposition: **REJECTED** as an overreach, but worth flagging to a reader: the
ordering is the single most quotable number in the pack and the least
generalizable. Disposition: **UPHELD as a reading risk**, disclosed in §10.

**CRT-7 — "Did the pack soften the negative development-block result?"**
No. It is given as the largest effect in the pack, with the interval that excludes
zero, and immediately discounted for sample size. Understating it would have been
the easier edit and it was not taken. Disposition: **REJECTED**.

**CRT-8 — "Does 'no risk-adjusted performance measure was computed' risk being read
as 'a performance measure was computed and withheld'?"**
Slight risk. §10 phrases it as a statement about what the experiment contains: no
position sizing, no cost model, no capacity analysis, and no risk-adjusted return
measure entered the experiment. Disposition: **UPHELD as a wording risk**, accepted;
the framing is "not computed", repeated in the claim register as a not-claimed item.

**CRT-9 — "Self-review is not review. Who checks the checker?"**
Partly fair. Three independent mechanisms exist: the generator verifies its own
outputs byte-for-byte, the cross-checker is a deliberately separate script with no
shared helpers or constants with the generator so a bug in one cannot mask a false
claim in the other, and CI runs both at full history depth. What is genuinely
self-assessed is this document and the other red-team documents. Disposition:
**UPHELD as a structural limitation**, disclosed; the mitigation is that every
red-team claim is tied to a command a reviewer can run, and that the lane stops at
READY FOR INDEPENDENT D10 REVIEW rather than self-certifying.

**CRT-10 — "Did the pack quietly fix, clean up or re-freeze anything to make the
story tidier?"**
No. Two things would have been easy to tidy and were deliberately left alone: the
fold-delta carry-over (P2.1) and the `C` boundary solution (P2.2). Both are
disclosed as accepted-evidence properties. The only edits this lane made were to
its own pack text and its own scripts. Disposition: **REJECTED**, testable by
comparing declared artifact hashes against disk.

**CRT-11 — "Does the pack claim the required study report exists?"**
No. The configuration names `research/historical-textual-signal-study.md` as the
required report; that file is absent at the accepted head and the pack says so,
records it as a known limitation, and sources no prose from it. Disposition:
**REJECTED**.

**CRT-12 — "The abstract states the answer is 'no'. Is that the same overclaim as
CRT-1?"**
It is a statement about the pre-specified comparison, immediately qualified by the
same paragraph's intervals, and by an explicit paragraph of claims not made
including the absence-versus-null distinction. The word "no" answers "did the
combined model do better" — not "is text worthless". Disposition: **UPHELD as a
phrasing risk worth a reviewer's eye**, accepted because the qualification is
adjacent and explicit.

## Summary

One residual gap is upheld and disclosed (CRT-4, the un-gated §7/§8 prose values),
one wording risk is upheld and accepted (CRT-8), one structural limitation is
upheld and disclosed (CRT-9), and one phrasing risk is flagged for the reviewer
(CRT-12). No claim in this pack was found to outrun its evidence, and none was
found stated without a citation.
