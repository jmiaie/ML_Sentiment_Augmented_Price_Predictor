#!/usr/bin/env python3
"""Authoritative D9-D (v2) historical text study runner.

Consumes the v2 acquisition artifacts (12-issuer 10-K/10-Q/8-K corpus + issuer
and SPY daily prices), builds the real event-level frame, and runs the
pre-registered purged walk-forward study for formation/DEV and 2024 validation.

Examples:
  # Pre-freeze: DEV + 2024 validation only. Cannot touch 2025.
  python scripts/run_historical_text_study_v2.py

  # 2025 HISTORICAL EVALUATION -- only with a frozen config + frozen corpus.
  python scripts/run_historical_text_study_v2.py \\\\
      --periods historical_evaluation --allow-2025

  # PRE-2025 only: select the final C on the 2024 validation window and print
  # the block to paste into walk_forward.final_selected_c. Runs no period study.
  python scripts/run_historical_text_study_v2.py --select-final-c

Fail-closed gates (each exits 2):
  * asking for historical_evaluation without --allow-2025
  * --allow-2025 while config status is not ``frozen-final``
  * --allow-2025 while ``walk_forward.final_selected_c`` lacks a complete
    pre-2025-selected C for either target
  * --select-final-c combined with --allow-2025 (selection is PRE-2025 only)
  * ANY run -- 2025 or not -- while EITHER manifest (filings or prices) is not
    exactly ``DATA FROZEN``, or is missing its canonical dataset hash (a
    ``DATA FROZEN (PARTIAL: N failed)`` label never passes)
C is never re-selected on the 2025 window: the frozen PER-TARGET
``final_selected_c`` values are handed to the engine as ``fixed_c``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quant_sentiment.event_frame_v2 import FilingEvent, build_event_frame
from quant_sentiment.historical_text_study_v2 import (
    FINAL_C_SELECTION_MAX_SESSION,
    MODEL_SPECS,
    PRIMARY_EMBARGO_SESSIONS,
    PRIMARY_TARGET_COLUMN,
    SECONDARY_EMBARGO_SESSIONS,
    SECONDARY_TARGET_COLUMN,
    PeriodSpec,
    run_period_study,
    select_final_c_on_validation,
    write_json_artifact,
)
from quant_sentiment.lm_dictionary import LM_DATASET_ID
from quant_sentiment.nyse_calendar import NyseCalendar

FROZEN_CONFIG_STATUS = "frozen-final"
HOLDOUT_PERIOD_NAME = "historical_evaluation"
FROZEN_MANIFEST_STATUS = "DATA FROZEN"
PRE_2025_PERIODS = ("formation_dev", "validation")
ALL_PERIODS = (*PRE_2025_PERIODS, HOLDOUT_PERIOD_NAME)

TARGETS = (
    ("primary", PRIMARY_TARGET_COLUMN, PRIMARY_EMBARGO_SESSIONS),
    ("secondary", SECONDARY_TARGET_COLUMN, SECONDARY_EMBARGO_SESSIONS),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_head(root: Path) -> str | None:
    """Best-effort producing-commit id; None when git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return manifest


def _frozen_c_for_target(
    experiment: dict[str, Any], target_name: str
) -> dict[str, float | None] | None:
    """The frozen, pre-2025-selected C for one target; None if not selected yet.

    ``walk_forward.final_selected_c`` holds one dictionary per target, because
    the primary (1-session) and secondary (5-session) targets have different
    horizons and embargoes and are selected independently. A target is only
    usable once every logistic model has a value; model 0 is the majority
    baseline and has no C.
    """
    table = experiment.get("walk_forward", {}).get("final_selected_c")
    if not isinstance(table, dict):
        return None
    entry = table.get(target_name)
    if not isinstance(entry, dict) or any(model not in entry for model in MODEL_SPECS):
        return None
    if any(entry[model] is None for model in MODEL_SPECS if MODEL_SPECS[model]):
        return None
    return {model: (None if value is None else float(value)) for model, value in entry.items()}


def _load_events(filings_dir: Path, symbols: list[str]) -> tuple[list[FilingEvent], dict[str, int]]:
    """Read the acquisition's per-issuer index CSVs into FilingEvent rows.

    Only ``status == OK`` rows become events; anything else was a failed
    acquisition and would silently shrink the corpus.
    """
    events: list[FilingEvent] = []
    not_ok: dict[str, int] = {}
    for symbol in symbols:
        index_path = filings_dir / "filings" / f"{symbol}_index.csv"
        with index_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "OK":
                    not_ok[symbol] = not_ok.get(symbol, 0) + 1
                    continue
                events.append(
                    FilingEvent(
                        cik=row["cik"],
                        ticker=row["ticker"],
                        company=row["company"],
                        accession=row["accession"],
                        form=row["form"],
                        filing_date=row["filing_date"],
                        acceptance_datetime=pd.Timestamp(row["acceptance_datetime"]),
                        text=(filings_dir / row["text_relpath"]).read_text(
                            encoding="utf-8", errors="replace"
                        ),
                    )
                )
    return events, not_ok


def _load_price_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="Date", parse_dates=True)
    return frame.sort_index()


def _append_ledger(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id",
        "repo",
        "branch",
        "dataset_id",
        "config_path",
        "status",
        "period_name",
        "period_start",
        "period_end",
        "primary_symbol",
        "seed",
        "artifact_path",
        "artifact_sha256",
        "key_metrics_json",
        "notes",
        "created_utc",
    ]
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/sentiment_historical_text_study_v2.yaml"),
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results/historical_text_v2"))
    parser.add_argument("--ledger", type=Path, default=Path("research/experiment-ledger.csv"))
    parser.add_argument("--branch", default="research/d9d-cap-lift")
    parser.add_argument(
        "--periods",
        default="formation_dev,validation",
        help=f"comma-separated subset of {ALL_PERIODS}",
    )
    parser.add_argument("--allow-2025", action="store_true")
    parser.add_argument(
        "--select-final-c",
        action="store_true",
        help=(
            "run the PRE-2025 final-C selection (purged formation train -> 2024 "
            "validation) and write the selection report; runs no period study and "
            "never reads 2025"
        ),
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    config_path = args.config if args.config.is_absolute() else root / args.config
    experiment: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_text = config_path.read_text(encoding="utf-8")
    status = str(experiment.get("status", ""))
    experiment_id = str(experiment["experiment_id"])
    dataset_ids: dict[str, str] = dict(experiment["dataset_ids"])

    periods_requested = [p.strip() for p in args.periods.split(",") if p.strip()]
    unknown = [p for p in periods_requested if p not in ALL_PERIODS]
    if unknown:
        print(f"ERROR: unknown --periods {unknown}; allowed {ALL_PERIODS}", file=sys.stderr)
        return 2
    wants_2025 = HOLDOUT_PERIOD_NAME in periods_requested

    # ---- gates (fail closed, before any data is read) ----
    if wants_2025 and not args.allow_2025:
        print(
            f"ERROR: --periods {HOLDOUT_PERIOD_NAME} requires --allow-2025 "
            "(it is a HISTORICAL EVALUATION, gated once only, after the final "
            "configuration freeze).",
            file=sys.stderr,
        )
        return 2
    if args.allow_2025 and status != FROZEN_CONFIG_STATUS:
        print(
            f"ERROR: --allow-2025 requires config status {FROZEN_CONFIG_STATUS!r} "
            f"(got {status!r}) in {config_path}",
            file=sys.stderr,
        )
        return 2
    if args.select_final_c and args.allow_2025:
        print(
            "ERROR: --select-final-c is a PRE-2025 operation and cannot be combined "
            "with --allow-2025.",
            file=sys.stderr,
        )
        return 2
    if args.allow_2025:
        unselected = [
            name for name, _, _ in TARGETS if _frozen_c_for_target(experiment, name) is None
        ]
        if unselected:
            print(
                "ERROR: --allow-2025 requires walk_forward.final_selected_c to hold a "
                "complete pre-2025-selected C for every target; missing or incomplete "
                f"for {unselected}. Without it the 2025 run would re-select C on its "
                "own window, which no_retune_after_freeze forbids.",
                file=sys.stderr,
            )
            return 2

    raw_dir = args.raw_dir or (root / "data" / "raw")
    manifests_dir = raw_dir.parent / "manifests"
    filings_manifest_path = manifests_dir / f"{dataset_ids['filings']}.json"
    prices_manifest_path = manifests_dir / f"{dataset_ids['prices']}.json"
    for manifest_path in (filings_manifest_path, prices_manifest_path):
        if not manifest_path.exists():
            print(f"ERROR: missing manifest {manifest_path}", file=sys.stderr)
            return 2

    filings_manifest = _load_manifest(filings_manifest_path)
    prices_manifest = _load_manifest(prices_manifest_path)
    filings_status = str(filings_manifest.get("status", ""))
    prices_status = str(prices_manifest.get("status", ""))

    # Every real run reads a frozen corpus -- not only the 2025 HISTORICAL
    # EVALUATION. Both datasets are gated: a DATA FROZEN filings manifest over a
    # VALIDATED prices manifest is a half-frozen corpus, and the price series
    # feed every feature and every target.
    for label, current_status, manifest in (
        ("filings", filings_status, filings_manifest),
        ("prices", prices_status, prices_manifest),
    ):
        if current_status != FROZEN_MANIFEST_STATUS:
            print(
                f"ERROR: {label} manifest status {current_status!r} is not "
                f"{FROZEN_MANIFEST_STATUS!r}. Every real empirical run requires BOTH "
                "datasets DATA FROZEN (VALIDATED and 'DATA FROZEN (PARTIAL: n failed)' "
                "are both refused).",
                file=sys.stderr,
            )
            return 2
        if not (manifest.get("sha256") or {}).get("dataset_canonical"):
            print(
                f"ERROR: {label} manifest has no sha256.dataset_canonical; a frozen "
                "snapshot must carry its canonical dataset hash.",
                file=sys.stderr,
            )
            return 2

    periods_cfg = experiment["periods"]
    period_specs = {
        name: PeriodSpec(
            name,
            periods_cfg[name]["start"],
            periods_cfg[name]["end_inclusive"],
        )
        for name in ALL_PERIODS
    }
    formation = period_specs["formation_dev"]

    # ---- corpus ----
    symbols = [str(s) for s in experiment["universe"]["symbols"]]
    market_symbol = str(experiment["universe"]["market_context_symbol"])
    filings_dir = raw_dir / dataset_ids["filings"]
    prices_dir = raw_dir / dataset_ids["prices"]

    events, not_ok_rows = _load_events(filings_dir, symbols)
    issuer_price_frames = {
        symbol: _load_price_frame(prices_dir / f"{symbol}.csv") for symbol in symbols
    }
    spy_price_frame = _load_price_frame(prices_dir / f"{market_symbol}.csv")
    calendar = NyseCalendar()
    frame, skip_counts = build_event_frame(events, issuer_price_frames, spy_price_frame, calendar)

    counts_by_form = (
        frame["form"].value_counts().to_dict() if not frame.empty else {}
    )
    corpus = {
        "n_filing_events_read": len(events),
        "n_index_rows_not_ok": not_ok_rows,
        "n_frame_rows": int(frame.shape[0]),
        "n_issuers_in_frame": int(frame["ticker"].nunique()) if not frame.empty else 0,
        "counts_by_form": {str(k): int(v) for k, v in counts_by_form.items()},
        "skip_counts": skip_counts,
        "filings_dataset_id": dataset_ids.get("filings"),
        "filings_manifest_status": filings_status,
        "filings_dataset_canonical_sha256": (filings_manifest.get("sha256") or {}).get(
            "dataset_canonical"
        ),
        "prices_dataset_id": dataset_ids.get("prices"),
        "prices_manifest_status": prices_status,
        "prices_dataset_canonical_sha256": (prices_manifest.get("sha256") or {}).get(
            "dataset_canonical"
        ),
        "dictionary_dataset_id_config": dataset_ids.get("dictionary"),
        "dictionary_dataset_id_runtime": LM_DATASET_ID,
        "dictionary_dataset_id_matches": dataset_ids.get("dictionary") == LM_DATASET_ID,
    }
    if not corpus["dictionary_dataset_id_matches"]:
        print(
            f"WARNING: config dictionary id {dataset_ids.get('dictionary')!r} != runtime "
            f"{LM_DATASET_ID!r}",
            file=sys.stderr,
        )
    if not frame.empty:
        corpus["frame_first_session"] = str(frame["effective_session"].iloc[0].date())
        corpus["frame_last_session"] = str(frame["effective_session"].iloc[-1].date())

    # ---- jobs ----
    walk_forward = experiment["walk_forward"]
    results_dir = args.results_dir if args.results_dir.is_absolute() else root / args.results_dir
    ledger_path = args.ledger if args.ledger.is_absolute() else root / args.ledger
    rel_config = (
        str(config_path.relative_to(root))
        if config_path.is_relative_to(root)
        else str(config_path)
    )
    git_head = _git_head(root)

    if args.select_final_c:
        selection_targets: dict[str, Any] = {}
        for target_name, target_column, embargo_sessions in TARGETS:
            selection_targets[target_name] = select_final_c_on_validation(
                frame,
                formation=formation,
                validation=period_specs["validation"],
                target_column=target_column,
                embargo_sessions=embargo_sessions,
                initial_train_size=int(walk_forward["initial_train_size"]),
                validation_size=int(walk_forward["validation_size"]),
                step_size=int(walk_forward["step_size"]),
                formation_internal_test_size=int(walk_forward["formation_internal_test_size"]),
            )
        selection_artifact: dict[str, Any] = {
            "experiment_id": experiment_id,
            "dataset_ids": dataset_ids,
            "config_path": rel_config,
            "config_status": status,
            "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
            "branch": args.branch,
            "code_git_head": git_head,
            "corpus": corpus,
            "selection_max_session": FINAL_C_SELECTION_MAX_SESSION,
            "targets": selection_targets,
            "notes": (
                "PRE-2025 final-C selection only. This mode produces no DEV, no 2024 "
                "and no 2025 evaluation numbers; it exists so the reviewed selection "
                "can be pasted into walk_forward.final_selected_c."
            ),
        }
        selection_path = results_dir / f"{experiment_id}_final_c_selection.json"
        selection_digest = write_json_artifact(selection_path, selection_artifact)
        print(f"wrote {selection_path} sha256={selection_digest[:16]}")
        print("walk_forward.final_selected_c:")
        print(
            yaml.safe_dump(
                {name: entry["selected_c"] for name, entry in selection_targets.items()},
                sort_keys=False,
            )
        )
        return 0

    # Per-target: the two targets have different horizons and embargoes, so their
    # C was selected independently. One shared dictionary would silently apply the
    # primary target's selection to the secondary target.
    fixed_c_by_target: dict[str, dict[str, float | None]] = {
        name: _frozen_c_for_target(experiment, name) or {} for name, _, _ in TARGETS
    }

    for target_name, target_column, embargo_sessions in TARGETS:
        for period_name in periods_requested:
            eval_period = period_specs[period_name]
            run_id = f"{experiment_id}_{target_name}_{period_name}"
            print(f"Running {run_id} eval={eval_period.name} target={target_column} ...")
            payload = run_period_study(
                frame,
                formation=formation,
                eval_period=eval_period,
                target_column=target_column,
                embargo_sessions=embargo_sessions,
                allow_holdout=args.allow_2025,
                fixed_c=fixed_c_by_target[target_name] if args.allow_2025 else None,
                initial_train_size=int(walk_forward["initial_train_size"]),
                validation_size=int(walk_forward["validation_size"]),
                formation_internal_test_size=int(walk_forward["formation_internal_test_size"]),
                step_size=int(walk_forward["step_size"]),
            )
            artifact: dict[str, Any] = {
                "experiment_id": run_id,
                "dataset_ids": dataset_ids,
                "config_path": rel_config,
                "config_status": status,
                "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
                "branch": args.branch,
                "code_git_head": git_head,
                "target_name": target_name,
                "period_selected": period_name,
                "corpus": corpus,
                "result": payload,
            }
            if period_name == HOLDOUT_PERIOD_NAME:
                artifact["historical_evaluation_label"] = str(
                    periods_cfg.get("historical_evaluation_label", "")
                )
                artifact["c_source"] = "frozen_config_final_selected_c"
            out_path = results_dir / f"{run_id}.json"
            digest = write_json_artifact(out_path, artifact)

            key_metrics = payload.get("key_metrics") or {}
            metrics_for_ledger = {k: v for k, v in key_metrics.items()}
            _append_ledger(
                ledger_path,
                {
                    "experiment_id": run_id,
                    "repo": "ML_Sentiment_Augmented_Price_Predictor",
                    "branch": args.branch,
                    "dataset_id": dataset_ids["filings"],
                    "config_path": rel_config,
                    "status": status,
                    "period_name": eval_period.name,
                    "period_start": eval_period.start,
                    "period_end": eval_period.end_inclusive,
                    "primary_symbol": "pooled_12_issuer",
                    "seed": str(experiment.get("seed", "")),
                    "artifact_path": (
                        str(out_path.relative_to(root))
                        if out_path.is_relative_to(root)
                        else str(out_path)
                    ),
                    "artifact_sha256": digest,
                    "key_metrics_json": json.dumps(metrics_for_ledger, sort_keys=True, default=str),
                    "notes": str(payload.get("notes", "")),
                    "created_utc": _utc_now(),
                },
            )
            print(
                f"  wrote {out_path} sha256={digest[:16]} n_eval={key_metrics.get('n_eval_rows')} "
                f"model3_ll={key_metrics.get('model3_log_loss')} "
                f"model1_ll={key_metrics.get('model1_log_loss')} "
                f"delta={key_metrics.get('headline_delta_log_loss_model3_minus_model1')} "
                f"c_selection={key_metrics.get('c_selection')}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
