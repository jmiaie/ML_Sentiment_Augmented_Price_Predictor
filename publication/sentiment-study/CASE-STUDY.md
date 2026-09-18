# CASE-STUDY — D10-D (sentiment-study)

## The decision this study was built to inform

Before spending engineering time on a filing-text signal, the question worth
answering is narrow and answerable: **given everything already known about price,
does the text of a SEC filing improve a short-horizon prediction of whether an
issuer beats the market?**

Not "is sentiment interesting". Not "can text be turned into features". Those are
easy to answer yes to and they do not settle a budget question. The question that
settles it is whether text features improve a fixed, pre-specified comparison
against the same model given market features alone.

## What was done

Twelve large-cap issuers, their `10-K`/`10-Q`/`8-K` filings from 2015 onward, and
a Loughran-McDonald dictionary representation of the filing text. Two targets, both
constructed as excess return over SPY — a one-session horizon and a five-session
horizon. Four logistic models: majority baseline, market-only, text-only, and
market-plus-text. Log loss as the headline metric. Regularisation strength selected
on data ending in 2024, written into a frozen configuration, and then read
verbatim by a single 2025 evaluation run.

The parts that make the result worth anything are unglamorous. The target is
excess over SPY, so a model cannot score by learning the market. Period slicing is
target-aware, so a row is admitted to a period only when its whole target window
lies inside that period. Training and evaluation are separated by a purge by NYSE
session distance plus an embargo, not by a row count. Frozen inputs are verified
by content hash before any fitting, fail-closed. And the comparison, the metric and
the regularisation were fixed before 2025 was touched.

## What came back

The market-only model was the best of the four in both targets. Adding filing text
made log loss worse, not better — on the 2025 period by `+0.0040902` for the
one-session target and `+0.0031305` for the five-session target, with bootstrap
intervals that include zero in both cases. The pre-2025 record points the same way
in all four earlier block/target combinations.

A second reading matters as much as the first: the intervals are wide. The data do
not cleanly separate "text adds nothing" from "text adds a little, in either
direction". What they rule out — on this universe, with this representation, at
these horizons — is a large, reliable improvement.

## What a decision-maker should take from it

1. **Do not fund a filing-text feature layer on the strength of this evidence.**
   In the one comparison that was fixed in advance, the text features did not pay
   for themselves in prediction error.
2. **Do not conclude that filing text is worthless.** A null result at 197
   evaluation rows with wide intervals is not a demonstration of absence. Anyone
   arguing the opposite direction from this same evidence is overreading it.
3. **Treat the negative result as the deliverable.** The value here is that the
   answer was produced by a comparison fixed before the evaluation period was
   read, with the regularisation chosen on pre-2025 data and the binding period
   label recorded as `PREVIOUSLY INSPECTED / HISTORICAL EVALUATION`. A result
   obtained that way is reusable. A result obtained by trying representations until
   one of them worked is not, no matter how good the number looks.
4. **Beware the null that is really a specification.** A margin over a majority
   baseline is not evidence of edge. Here the market-only model beat the baseline
   by a hair on one target and the combined model gave most of that back. A
   log-loss ordering among four classifiers over one year is not a trading result;
   no costs, no sizing, no capacity and no risk-adjusted return measure entered
   this experiment at all.

## Why publish a negative result

Because the alternative is a literature and a market full of survivors. A study
that reports only the specification that worked teaches nothing about whether the
next attempt will work, and it quietly transfers the cost of the falsification to
whoever tries next. This pack reports one narrow, pre-specified comparison, its
provenance, and its limitations — including the places where the accepted evidence
is less clean than a reader might assume.

## Provenance of this document

Every number above is re-derivable from the artifacts cited in
`RESULT-SOURCE-MAP.md` by running the commands in `TECHNICAL-PAPER.md` §11. This
document introduces no number that is not also carried by a generated table.
