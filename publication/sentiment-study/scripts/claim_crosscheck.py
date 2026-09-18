#!/usr/bin/env python3
"""D10-D independent claim cross-check.

Separate from publication_pack.py ON PURPOSE: no shared helpers, no shared
tables, no shared constants. This script re-derives the pack's claims from the
accepted artifacts, the append-only ledger and the raw pack prose, and fails
closed on any disagreement. It is the executable form of CLAIM-RED-TEAM.md.

Offline. Reads only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import cast

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
DOCS = ["TECHNICAL-PAPER.md", "CASE-STUDY.md", "D10-STATUS.md", "CLAIM-REGISTER.md"]
PROSE = ["TECHNICAL-PAPER.md", "CASE-STUDY.md", "D10-STATUS.md", "QUANT-RED-TEAM.md",
         "CLAIM-RED-TEAM.md", "CITATION-RED-TEAM.md", "SOURCE-GATE.md"]

LABEL_2025 = "PREVIOUSLY INSPECTED / HISTORICAL EVALUATION"
HEAD = "0fa6cfd3e124f36eea3383af7da2f71b6c43934f"
CONFIG_SHA = "bd702bf5f58311b80ad68dc0fc9bb89bbcd11479ddd49c480e7fd6beacdd80f8"
PRIMARY_2025 = "6ed9197eb2d51cd68341a7fc1a9790eb61d6fa9ad10fda9e148a8da5120cca36"
SECONDARY_2025 = "b207ed7a5c7923f6bdb0217bdc37963b5e81b37a2dc34788b0696ef822557d12"
PERIODS = ("formation_dev", "validation", "historical_evaluation")
TARGETS = ("primary", "secondary")
MODELS = ("model0_majority_baseline", "model1_market_only", "model2_text_only",
          "model3_market_text_combined")
BANNED = ["proves", "outperform", "adds value", "demonstrates incremental value",
          "confirms that text", "Sharpe", "alpha", "economic significance",
          "untouched holdout", "new holdout", "pristine", "correctly remained"]
# An overclaim token is allowed only inside an explicit negation, so the pack can
# state plainly what it does NOT claim ("this is not an untouched holdout") without
# opening the door to an unnegated overclaim. A bare occurrence still fails.
NEG_CUES = ("no ", "not ", "never ", "without ", "neither", "nor ", "none",
            "cannot", "can not", "must not", "rather than", "instead of")

# QUOTE-EXEMPT HANDLING. Two rules keep the guard strict while still letting the
# pack name the exact phrases it refuses to assert:
#   1. a prohibited token entirely inside a balanced double-quoted span on one
#      line is quoted text or a quoted attack the pack is refuting, not an
#      assertion, so it is skipped;
#   2. ordinary prose is scanned strictly and a bare token still fails -- a token
#      is accepted only when a negation cue DIRECTLY governs it.
# The exemption exists precisely so red-team attack strings are not misclassified
# as author assertions. It is deliberately NOT "any negation cue anywhere on the
# line": a stray "not" elsewhere in a sentence must not whitelist a real overclaim.
QUOTED = re.compile(r'"[^"\n]*"')
# Punctuation/coordination that ends the clause a cue governs. A cue on the far
# side of one of these does not reach the token, so "this is not unusual; the
# study uses an untouched holdout" still fails.
CLAUSE_BREAKS = (";", ",", ":", " but ", " and ", " -- ", " - ", ". ")
MAX_WORDS_AFTER_CUE = 2
# A printed number, with its sign. Used to read a table cell back out of the paper.
NUM = re.compile(r"[+-]?\d+(?:\.\d+)?")


def violations(text: str) -> list[tuple[str, int]]:
    """Prohibited tokens in `text` that read as author assertions.

    Quoted spans are excluded (see QUOTE-EXEMPT HANDLING above); everything else
    must be negation-governed to be accepted. An unbalanced quote forms no span
    and therefore hides nothing: malformed quoting fails closed rather than
    weakening the guard.
    """
    quoted = [(m.start(), m.end()) for m in QUOTED.finditer(text)]
    flagged: list[tuple[str, int]] = []
    for token in BANNED:
        for m in re.finditer(re.escape(token), text, re.IGNORECASE):
            if any(s <= m.start() and m.end() <= e for s, e in quoted):
                continue
            window = text[max(0, m.start() - 40):m.start()].lower()
            negated = False
            for cue in NEG_CUES:
                i = window.rfind(cue)
                if i == -1:
                    continue
                between = window[i + len(cue):]
                if (len(between.split()) <= MAX_WORDS_AFTER_CUE
                        and not any(b in between for b in CLAUSE_BREAKS)):
                    negated = True
                    break
            if not negated:
                flagged.append((token, m.start()))
    return flagged


# A fail-closed gate that cannot fail is not a gate. These cases pin both
# directions of the guard, including the stray-negation trap.
REGRESSIONS = (
    ('The phrase "untouched holdout" is prohibited here.', False, "quoted example"),
    ('Attack tested: "pristine holdout".', False, "quoted attack"),
    ("This is not an untouched holdout.", False, "negated assertion"),
    ("This is an untouched holdout.", True, "bare overclaim"),
    ("This is a pristine holdout.", True, "bare overclaim"),
    ("This is not unusual; the study uses an untouched holdout.", True,
     "stray 'not' elsewhere on the line must not whitelist"),
)


def selftest() -> int:
    bad = 0
    for text, should_flag, note in REGRESSIONS:
        flagged = bool(violations(text))
        if flagged != should_flag:
            print(f"SELFTEST FAIL: expected flag={should_flag} got {flagged} "
                  f"({note}): {text!r}")
            bad += 1
    if bad:
        print(f"overclaim guard selftest FAILED: {bad}/{len(REGRESSIONS)} cases wrong")
        return 1
    print(f"overclaim guard selftest OK: {len(REGRESSIONS)} regression cases "
          f"({sum(1 for r in REGRESSIONS if r[1])} must-fail, "
          f"{sum(1 for r in REGRESSIONS if not r[1])} must-pass)")
    return 0

CHECKS = 0


def ok(cond: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        print(f"ASSERT FAIL: {label}")
        raise SystemExit(1)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(rel: str) -> dict:
    return cast(dict, json.loads((ROOT / rel).read_text(encoding="utf-8")))


def art(target: str, period: str) -> dict:
    return load(f"results/historical_text_v2/sentiment_hist_text_v2_{target}_{period}.json")


def main() -> int:
    # ---- artifact hashes agree with the append-only ledger's own declaration ----
    ledger = {}
    with (ROOT / "research/experiment-ledger.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ledger[row["experiment_id"]] = row
    for period in PERIODS:
        for target in TARGETS:
            rel = f"results/historical_text_v2/sentiment_hist_text_v2_{target}_{period}.json"
            exp = f"sentiment_hist_text_v2_{target}_{period}"
            ok(exp in ledger, f"ledger row exists for {exp}")
            ok(ledger[exp]["artifact_sha256"] == sha(ROOT / rel),
               f"ledger artifact_sha256 matches disk for {exp}")

    # ---- per-period independent re-derivation of every headline number ----
    for period in PERIODS:
        for target in TARGETS:
            doc = art(target, period)
            k = doc["result"]["key_metrics"]
            m = doc["result"]["period_metrics"]
            m3 = m["model3_market_text_combined"]["log_loss"]
            m1 = m["model1_market_only"]["log_loss"]
            delta = m3 - m1
            ok(abs(delta - k["headline_delta_log_loss_model3_minus_model1"]) < 1e-12,
               f"headline delta_log_loss recomputes for {target}/{period}")
            bdelta = (m["model3_market_text_combined"]["balanced_accuracy"]
                      - m["model1_market_only"]["balanced_accuracy"])
            ok(abs(bdelta - k["headline_delta_balanced_accuracy_model3_minus_model1"]) < 1e-12,
               f"headline delta_balanced_accuracy recomputes for {target}/{period}")
            ci = k["headline_bootstrap_delta_log_loss_ci95"]
            ok(ci["ci_low"] < ci["mean_delta"] < ci["ci_high"],
               f"bootstrap CI brackets its own mean for {target}/{period}")
            ok(ci["n_resamples"] == 500, f"bootstrap resamples == 500 for {target}/{period}")
            for model in MODELS:
                ok(m[model]["samples"] == k["n_eval_rows"],
                   f"{model} sample count == n_eval_rows for {target}/{period}")
            ok(k["train_eval_intersection_count"] == 0,
               f"zero train/eval row intersection for {target}/{period}")
            ok(k["train_windows_close_before_eval_start"] is True,
               f"training windows close before eval start for {target}/{period}")
            ok(k["walk_forward_fold_count"] == 90,
               f"walk_forward_fold_count == 90 for {target}/{period}")

    # ---- 2025 block: geometry, C provenance, label, corpus arithmetic ----
    for target in TARGETS:
        doc = art(target, "historical_evaluation")
        k = doc["result"]["key_metrics"]
        t = k["pre_evaluation_training"]
        ok(doc["historical_evaluation_label"] == LABEL_2025, f"2025 label exact for {target}")
        ok(doc["c_source"] == "frozen_config_final_selected_c", f"c_source frozen for {target}")
        ok(doc["config_status"] == "frozen-final", f"config frozen-final for {target}")
        ok(doc["config_sha256"] == CONFIG_SHA, f"2025 run cites frozen config sha for {target}")
        ok(doc["code_git_head"] == HEAD, f"2025 run cites producing code head for {target}")
        ok(k["max_train_target_end_ordinal"] < t["purge_cutoff_ordinal"],
           f"train target windows end before purge cutoff for {target}")
        ok(t["first_eval_effective_session_ordinal"]
           == t["purge_cutoff_ordinal"] + t["embargo_sessions"],
           f"embargo gap == embargo_sessions for {target}")
        ok(k["headline_fold_delta_log_loss_distribution"] == [],
           f"2025 fold-delta distribution N/A under frozen-C for {target}")
        for model in ("model1_market_only", "model2_text_only", "model3_market_text_combined"):
            ok(k["selected_regularization"][model] == 0.01, f"C == 0.01 for {model} {target}")
        ok(k["selected_regularization"]["model0_majority_baseline"] is None,
           f"baseline has no C for {target}")

    # ---- EVERY quantitative cell printed in §7 and §8 is derived from an artifact.
    # §7: the four model log losses, the headline delta, both interval bounds and the
    # balanced-accuracy delta, per target. §8: the delta, both interval bounds, n_eval
    # and n_train, for each of the four block/target rows. A printed cell that no
    # longer matches its artifact fails the build, so no cell can drift unseen.
    paper_text = (PACK / "TECHNICAL-PAPER.md").read_text(encoding="utf-8")
    sec7 = paper_text.split("## 7. Results", 1)[1].split("## 8.", 1)[0]
    sec8 = paper_text.split("## 8. Pre-2025 record", 1)[1].split("## 9.", 1)[0]
    sec7_blocks = {
        "primary": sec7.split("Primary target", 1)[1].split("Secondary target", 1)[0],
        "secondary": sec7.split("Secondary target", 1)[1],
    }

    def row_tokens(section: str, key: tuple, where: str) -> list[str]:
        """The numbers printed in the table row whose leading cells equal `key`."""
        for line in section.splitlines():
            cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
            if tuple(cells[:len(key)]) == key:
                return NUM.findall(" ".join(cells[len(key):]).replace(",", ""))
        ok(False, f"{where}: table row {key} not found")
        raise SystemExit(1)

    def printed_is(tokens: list[str], expected: list[tuple[float, int]], where: str) -> None:
        """Each printed number equals its artifact value at the precision printed."""
        ok(len(tokens) == len(expected),
           f"{where}: {len(tokens)} numbers printed, {len(expected)} expected")
        for token, (value, dp) in zip(tokens, expected, strict=True):
            ok(abs(float(token) - value) <= 0.5 * 10 ** -dp,
               f"{where}: printed {token} vs artifact {value:.{dp}f}")

    for target, block in sec7_blocks.items():
        result = art(target, "historical_evaluation")["result"]
        km = result["key_metrics"]
        ci = km["headline_bootstrap_delta_log_loss_ci95"]
        for model in MODELS:
            printed_is(row_tokens(block, (model,), f"§7 {target}"),
                       [(result["period_metrics"][model]["log_loss"], 10)],
                       f"§7 {target} {model} log loss")
        prose = " ".join(line for line in block.splitlines()
                         if not line.lstrip().startswith("|"))
        for label, value, dp in (
            ("headline delta", km["headline_delta_log_loss_model3_minus_model1"], 7),
            ("interval low", ci["ci_low"], 7),
            ("interval high", ci["ci_high"], 7),
            ("balanced-accuracy delta",
             km["headline_delta_balanced_accuracy_model3_minus_model1"], 10),
        ):
            ok(f"{value:+.{dp}f}" in prose,
               f"§7 {target} {label} printed as {value:+.{dp}f}")

    for period, block_label in (("formation_dev", "development internal test"),
                                ("validation", "2024 validation")):
        for target in TARGETS:
            pkm = art(target, period)["result"]["key_metrics"]
            pci = pkm["headline_bootstrap_delta_log_loss_ci95"]
            printed_is(row_tokens(sec8, (block_label, target), f"§8 {block_label}"),
                       [(pkm["headline_delta_log_loss_model3_minus_model1"], 8),
                        (pci["ci_low"], 8), (pci["ci_high"], 8),
                        (float(pkm["n_eval_rows"]), 0), (float(pkm["n_train_rows"]), 0)],
                       f"§8 {target}/{period}")

    # ---- P2.1 fold-distribution carry-over: machine-verified disclosure ----
    for target in TARGETS:
        dev = art(target, "formation_dev")["result"]["key_metrics"][
            "headline_fold_delta_log_loss_distribution"]
        val = art(target, "validation")["result"]["key_metrics"][
            "headline_fold_delta_log_loss_distribution"]
        ok(len(dev) == 90 and dev == val,
           f"fold-delta distribution identical across dev/validation for {target} (P2.1)")

    corpus = art("primary", "historical_evaluation")["corpus"]
    forms = sum(corpus["counts_by_form"].values())
    ok(forms == corpus["n_frame_rows"], "form counts sum to frame rows")
    skips = sum(corpus["skip_counts"].values())
    ok(corpus["n_frame_rows"] == corpus["n_filing_events_read"] - skips,
       "frame rows == events read minus every documented skip")
    ok(corpus["n_issuers_in_frame"] == 12, "12 issuers present in the frame")

    # ---- the null direction itself, and that it is NOT oversold ----
    for target in TARGETS:
        k = art(target, "historical_evaluation")["result"]["key_metrics"]
        ci = k["headline_bootstrap_delta_log_loss_ci95"]
        ok(k["headline_delta_log_loss_model3_minus_model1"] > 0,
           f"text does not lower log loss in 2025 for {target}")
        ok(ci["ci_low"] < 0 < ci["ci_high"],
           f"2025 bootstrap CI includes zero for {target}")

    # ---- the disclosed fold-distribution carry-over is real (not a typo) ----
    for target in TARGETS:
        fdev = art(target, "formation_dev")["result"]["key_metrics"][
            "headline_fold_delta_log_loss_distribution"]
        val = art(target, "validation")["result"]["key_metrics"][
            "headline_fold_delta_log_loss_distribution"]
        ok(len(fdev) == len(val) == 90, f"fold distributions have 90 elements for {target}")
        ok(fdev == val, f"fold distribution identical across formation_dev/validation for {target}")

    # ---- v1 lineage stays preserved, superseded, and out of the tables ----
    v1 = load("results/historical_text/sentiment_hist_text_v1_holdout_2025.json")
    ok(v1["config_status"] == "frozen-for-holdout", "v1 artifacts remain preserved as-is")
    ok("sentiment_hist_text_v1_holdout_2025" in ledger, "v1 ledger row preserved")
    table = (PACK / "tables/period-metrics.csv").read_text(encoding="utf-8")
    ok("v1" not in table, "superseded v1 numbers are not presented as evidence in the tables")

    # ---- the required report path really is absent, and the pack says so ----
    config_text = (ROOT / "configs/experiments/sentiment_historical_text_study_v2.yaml").read_text(
        encoding="utf-8")
    ok("path: research/historical-textual-signal-study.md" in config_text,
       "frozen config still names the required report path")
    ok(not (ROOT / "research/historical-textual-signal-study.md").is_file(),
       "required report path is absent at this HEAD (disclosed limitation)")

    # ---- prose: numbers present, overclaims absent ----
    paper = (PACK / "TECHNICAL-PAPER.md").read_text(encoding="utf-8")
    ok(LABEL_2025 in paper, "paper carries the binding 2025 label verbatim")
    for target in TARGETS:
        delta = art(target, "historical_evaluation")["result"]["key_metrics"][
            "headline_delta_log_loss_model3_minus_model1"]
        ok(repr(round(delta, 8)) in paper, f"paper states the measured headline delta for {target}")
    ok(PRIMARY_2025 in paper, "paper cites the primary 2025 artifact hash")
    ok(SECONDARY_2025 in paper, "paper cites the secondary 2025 artifact hash")
    ok(CONFIG_SHA in paper, "paper cites the frozen config hash")
    for name in PROSE:
        text = (PACK / name).read_text(encoding="utf-8")
        for token, pos in violations(text):
            ok(False, f"{name}: overclaim token '{token}' asserted at offset {pos}")
        for match in re.finditer(r"significantly", text, re.IGNORECASE):
            window = text[max(0, match.start() - 40):match.start()].lower()
            ok(
                any(cue in window for cue in ("not ", "no ", "without ")),
                f"{name}: 'significantly' only in a negative claim",
            )

    for name in DOCS:
        ok((PACK / name).is_file(), f"deliverable present: {name}")

    print(f"claim cross-check OK: {CHECKS} independent assertions passed")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main())
