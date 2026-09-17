#!/usr/bin/env python3
"""Authoritative D9-D (v2) historical text study runner.

Consumes the v2 acquisition artifacts (12-issuer 10-K/10-Q/8-K corpus + issuer
and SPY daily prices), builds the real event-level frame, and runs the
pre-registered purged walk-forward study for formation/DEV and 2024 validation.

Examples:
  # Pre-freeze: DEV + 2024 validation only. Cannot touch 2025.
  python scripts/run_historical_text_study_v2.py

  # 2025 HISTORICAL EVALUATION -- only with a frozen config + frozen corpus.
  python scripts/run_historical_text_study_v2.py \\
      --periods historical_evaluation --allow-2025

Fail-closed gates (each exits 2):
  * asking for historical_evaluation without --allow-2025
  * --allow-2025 while config status is not ``frozen-final``
  * --allow-2025 while ``walk_forward.final_selected_c`` is unpopulated
  * --allow-2025 while the filings manifest is not ``DATA FROZEN``
  * any run whose filings manifest status is something else (a
    ``DATA FROZEN (PARTIAL: N failed)`` label never passes)
C is never re-selected on the 2025 window: the frozen ``final_selected_c``
values are handed to the engine as ``fixed_c``.
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
    PRIMARY_EMBARGO_SESSIONS,
    PRIMARY_TARGET_COLUMN,
    SECONDARY_EMBARGO_SESSIONS,
    SECONDARY_TARGET_COLUMN,
    PeriodSpec,
    run_period_study,
    write_json_artifact,
)
from quant_sentiment.lm_dictionary import LM_DATASET_ID
from quant_sentiment.nyse_calendar import NyseCalendar

FROZEN_CONFIG_STATUS = "frozen-final"
HOLDOUT_PERIOD_NAME = "historical_evaluation"
ACCEPTED_MANIFEST_STATUSES = ("VALIDATED", "DATA FROZEN")
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
    return json.loads(path.read_text(encoding="utf-8"))


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
    frozen_c = experiment.get("walk_forward", {}).get("final_selected_c")
    if args.allow_2025 and not frozen_c:
        print(
            "ERROR: --allow-2025 requires walk_forward.final_selected_c to be "
            "populated (pre-2025-selected C). Without it the 2025 run would "
            "re-select C on its own window, which no_retune_after_freeze forbids.",
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
    if filings_status not in ACCEPTED_MANIFEST_STATUSES:
        print(
            f"ERROR: filings manifest status {filings_status!r} is not one of "
            f"{ACCEPTED_MANIFEST_STATUSES} (a PARTIAL freeze is never usable)",
            file=sys.stderr,
        )
        return 2
    if args.allow_2025 and filings_status != FROZEN_MANIFEST_STATUS:
        print(
            f"ERROR: --allow-2025 requires the filings corpus to be "
            f"{FROZEN_MANIFEST_STATUS!r} (got {filings_status!r}). Freeze the "
            "uncapped corpus before the 2025 HISTORICAL EVALUATION.",
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
        "filings_manifest_status": filings_status,
        "filings_dataset_canonical_sha256": (filings_manifest.get("sha256") or {}).get(
            "dataset_canonical"
        ),
        "prices_manifest_status": str(prices_manifest.get("status", "")),
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

    fixed_c: dict[str, float | None] | None = None
    if args.allow_2025:
        fixed_c = {
            str(model): (None if value is None else float(value))
            for model, value in dict(frozen_c).items()
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
                fixed_c=fixed_c,
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
