# QUANT-RED-TEAM — D10-D (sentiment-study)

Adversarial quantitative review of this pack. The purpose is to attack the study's
conclusions and the pack's honesty before an external reviewer does. Each item
records the attack, the disposition, and where the pack discloses it.

Disposition vocabulary: **UPHELD** (attack lands; pack already discloses it),
**REPAIRED** (attack landed on a defect in this pack; fixed),
**REJECTED** (attack does not survive contact with the evidence).

---

**QRT-1 — "One year, 197 rows, and you are drawing conclusions about a feature
class?"**
Attack lands on the strength of the claim, not its direction. Intervals are wide:
for the primary target the interval spans roughly −0.0049 to +0.0108, so it
brackets both a small real benefit and a small real harm. Disposition: **UPHELD**,
disclosed in `TECHNICAL-PAPER.md` §7 and §9.5. The pack states the negative
finding, states that the intervals include zero, and refuses to convert a null
into a demonstration of absence.

**QRT-2 — "The 20-row development block has an interval excluding zero. Is that
being sold as evidence?"**
It is reported and immediately discounted: `TECHNICAL-PAPER.md` §8 says the 20-row
block is small and its interval should not be treated as a stable estimate.
Disposition: **UPHELD** with disclosure.

**QRT-3 — "Six block×target comparisons all pointing the same way is a
multiple-comparison artefact dressed as consistency."**
Partly fair. Only one comparison was pre-specified as the headline (2025, primary
target, log loss, m3−m1). The statement "higher log loss in all six blocks" is a
post-hoc aggregate of the record, not a pre-specified test, and the pack does not
attach a p-value or interval to that aggregate. Disposition: **UPHELD**; the paper
presents the six numbers as a record, not as a test.

**QRT-4 — "Model 1 barely beats the majority baseline. Should the headline be m3
versus m0?"**
No, and this is a place where the pack deliberately does not take the easier
comparison. m1-vs-m0 is not the question posed; the question is whether text adds
information beyond market features, which requires m1 as the comparator. The
baseline is reported so a reader can see that the market-only margin is small.
Disposition: **REJECTED** as a change; the baseline is disclosed.

**QRT-5 — "The fold-delta distribution is identical across two runs with different
training sizes. That is a red flag."**
Attack lands, and it was found independently while authoring this pack. In both the
development and validation artifacts, `headline_fold_delta_log_loss_distribution`
carries the same 90-element list, element for element, although those runs trained
on 1,864 and 1,889 rows respectively. `walk_forward_fold_count` also reads 90 in
every period, including a block with 20 evaluation rows. A field that does not move
with its period's training geometry cannot be read as a period-specific statistic.
Disposition: **UPHELD**, disclosed at `TECHNICAL-PAPER.md` §9.1 and `D10-STATUS.md`
P2.1. The artifacts were not altered to correct it — that would be a prohibited
change to accepted evidence — and no published number in this pack depends on that
field.

**QRT-6 — "The fold-delta list is empty in the 2025 artifacts. Is output missing?"**
No. Under the frozen-`C` path the tuner is skipped, so no per-fold deltas are
computed and the field is legitimately empty. Disposition: **REJECTED**, with the
reason disclosed at §9.2 so a reader does not have to guess.

**QRT-7 — "C = 0.01 sits at the grid edge. The selection is a boundary solution,
therefore untrustworthy."**
The value is at the lower edge of the pre-specified grid, which the pack discloses
as a limitation. What the attack cannot establish is that the reported comparison
is tuned: the same `C` applies to the market-only, text-only and combined models
alike, so a boundary solution cannot manufacture the *difference* between the
combined and market-only models. Disposition: **UPHELD as a disclosure**
(`TECHNICAL-PAPER.md` §6), **REJECTED as an explanation of the result**.

**QRT-8 — "Static present-day universe. Survivorship bias."**
True, and stated in the frozen configuration itself, which the pack quotes.
Disposition: **UPHELD**, `TECHNICAL-PAPER.md` §2; the pack states plainly that
repairing it would require reacquiring data, which D10 forbids.

**QRT-9 — "One seed. No seed-robustness grid."**
Correct, and disclosed at §9.8. A seed grid would be new empirical work, outside
this lane's authority and prohibited by D10. Disposition: **UPHELD** as a
limitation.

**QRT-10 — "No transaction costs, no capacity, no risk-adjusted return. So the
study says nothing about making money."**
Correct, and the pack says so in terms rather than leaving it implied: §10 states
that no position sizing, cost model, capacity analysis or risk-adjusted return
measure was computed anywhere in the experiment. Disposition: **UPHELD**; the
limitation is stated as a not-claimed item rather than buried.

**QRT-11 — "Text-only (m2) is the worst model. Doesn't that prove the text
features are noise?"**
No. m2 also lacks the market features that carry whatever signal exists, so a poor
m2 is expected regardless of whether the text features contain information. The m2
result is reported and is not used to support any claim about the text features.
Disposition: **REJECTED** as an inference; the raw numbers are reported so a reader
can see them.

**QRT-12 — "The secondary target's interval is narrower in the wrong direction —
is the horizon choice cherry-picked?"**
Both horizons were fixed in the frozen configuration, both are reported, and both
point the same way. Disposition: **REJECTED**.

**QRT-13 — "Is the 2025 period really untouched?"**
It is not, and the pack refuses the word. The binding label is
`PREVIOUSLY INSPECTED / HISTORICAL EVALUATION`, carried verbatim in the source
gate, the paper and the status document. Disposition: **UPHELD as the correct and
mandatory framing.**

## Residual risk accepted by this lane

The strongest surviving criticism is sample size (QRT-1) combined with a single
seed (QRT-9) and a boundary `C` (QRT-7). Together they mean the point estimates in
this pack should be read as a direction, not a magnitude, and the pack says so
throughout. No attempt is made to convert this record into a stronger claim, and
no additional empirical work is performed to shore it up, because both are outside
this lane's authority.
