#!/usr/bin/env python3
"""Check whether a fresh Yahoo price acquisition reproduces the frozen price manifest.

The published study's raw CSVs are not stored in the repository, only manifests with
SHA-256 hashes. This script re-downloads the Yahoo daily prices with the same functions the
repository's acquisition script uses (``quant_sentiment.acquisition``), with the manifest's
recorded parameters, into a temporary directory outside the repository, and compares the
result with the frozen manifest: per-file SHA-256, row count, first and last date and
missing-value count.

Yahoo restates adjusted prices, so a fresh fetch is expected NOT to match byte for byte. A
difference is reported as "DATA DRIFT (not a reproduction failure of the analysis)" and
described, never treated as an error. Only a failed fetch makes the exit code non-zero.

Not covered: the SEC EDGAR datasets need a contact header kept as a repository secret
(SEC_USER_AGENT). They are listed in the report as not checked and are never fetched.

Safety properties:

* Frozen manifests are never written: the price download functions are called directly, not
  the SEC acquisition script (which would rewrite data/manifests/ and fetch SEC data), and
  the manifest directory is hashed before and after to prove nothing changed.
* Fetched CSVs live in a temporary directory that is removed on exit. Only the JSON report
  (hashes, counts, dates, versions; no price data) is meant to be kept.
* Nothing is committed, pushed or uploaded by this script.

Exit codes: 0 every dataset was fetched and compared (match or drift); 2 any fetch failed or
a safety check tripped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"

CANONICAL_KEY = "dataset_canonical"
OHLCV_COLS = ("Open", "High", "Low", "Close", "Volume")
ACTION_COLS = ("Dividends", "Stock Splits", "Capital Gains")

CLASS_MATCH = "MATCH"
CLASS_DRIFT = "DATA DRIFT (not a reproduction failure of the analysis)"
CLASS_FAILED = "FETCH FAILED"
CLASS_UNVERIFIABLE = "UNVERIFIABLE (manifest records no hashes)"

DRIFT_EXPLANATION = (
    "Yahoo restates split- and dividend-adjusted prices (auto_adjust=True), so a later "
    "fetch can differ from the frozen bytes even when nothing is wrong. The size of any "
    "price difference is not quantified here because the frozen raw CSVs are not stored "
    "in the repository."
)

# One entry per frozen dataset that can be re-acquired without credentials.
DATASETS: tuple[dict[str, Any], ...] = (
    {
        "manifest": "yf_sentiment_equities_daily_2015_2025_v1.json",
        "kind": "yfinance_inprocess",
        "via": "quant_sentiment.acquisition",
    },
)

NOT_CHECKED: tuple[dict[str, str], ...] = (
    {
        "dataset": "edgar_8k_yf_megacap_daily_2015_2025_v1",
        "reason": "needs SEC_USER_AGENT secret",
    },
    {
        "dataset": "sec_filings_12issuer_2015_2025_v1",
        "reason": "needs SEC_USER_AGENT secret",
    },
)


class FetchError(RuntimeError):
    """A fresh acquisition could not be completed (network, script or safety failure)."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- hashing


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_dataset_hash(file_hashes: dict[str, str]) -> str:
    """Same formula as the acquisition scripts: sorted ``name:sha256`` lines."""
    payload = "\n".join(f"{k}:{v}" for k, v in sorted(file_hashes.items())) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def snapshot_dir(path: Path) -> dict[str, str]:
    """Map every file name in a directory to its SHA-256 (empty if it does not exist)."""
    if not path.is_dir():
        return {}
    return {p.name: sha256_file(p) for p in sorted(path.iterdir()) if p.is_file()}


# ------------------------------------------------------------------- CSV inspection


def _is_missing(cell: str) -> bool:
    return cell.strip().lower() in ("", "nan", "none", ".")


def _nonzero(cell: str) -> bool:
    if _is_missing(cell):
        return False
    try:
        return float(cell) != 0.0
    except ValueError:
        return True


def summarize_csv(path: Path) -> dict[str, Any]:
    """Row count, first/last date, missing OHLCV cells, action counts and columns of a CSV."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None) or []
        rows = [r for r in reader if r]
    dates = [r[0].strip()[:10] for r in rows]
    summary: dict[str, Any] = {
        "rows": len(rows),
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "columns": sorted(header[1:]),
        "missing_values": 0,
        "actions": None,
    }
    idx = {name: i for i, name in enumerate(header)}
    ohlcv = [idx[c] for c in OHLCV_COLS if c in idx]
    summary["missing_values"] = sum(
        1 for r in rows for i in ohlcv if i >= len(r) or _is_missing(r[i])
    )
    actions: dict[str, Any] = {}
    for col in ACTION_COLS:
        key = col.lower().replace(" ", "_")
        if col in idx:
            i = idx[col]
            actions[key] = {
                "present": True,
                "nonzero_count": sum(1 for r in rows if i < len(r) and _nonzero(r[i])),
            }
        else:
            actions[key] = {"present": False, "nonzero_count": None}
    summary["actions"] = actions
    return summary


# ------------------------------------------------------------------ manifest access


def _per_name(manifest: dict[str, Any], key: str, name: str, *fallbacks: str) -> Any:
    """Look up a recorded value for one symbol/series across the manifest layouts."""
    value = manifest.get(key)
    if isinstance(value, dict):
        return value.get(name)
    if value is not None:
        return value
    for fb in fallbacks:
        if manifest.get(fb) is not None:
            return manifest[fb]
    return None


def expected_metadata(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Recorded rows, dates, missing count, actions and columns for one symbol or series."""
    cols = None
    for key in ("per_symbol_columns", "columns"):
        block = manifest.get(key)
        if isinstance(block, dict) and block.get(name) is not None:
            cols = sorted(block[name])
    actions = manifest.get("actions")
    return {
        "rows": _per_name(manifest, "row_counts", name, "row_count_uniform", "row_count"),
        "first_date": _per_name(manifest, "actual_start", name, "actual_start_uniform"),
        "last_date": _per_name(manifest, "actual_end", name, "actual_end_uniform"),
        "missing_values": _per_name(
            manifest,
            "missing_counts",
            name,
            "missing_ohlcv_cells_uniform",
            "missing_value_count",
        ),
        "actions": actions.get(name) if isinstance(actions, dict) else None,
        "columns": cols,
    }


def expected_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    sha = manifest.get("sha256") or {}
    return {k: v for k, v in sha.items() if k != CANONICAL_KEY and isinstance(v, str)}


def source_version_of(manifest: dict[str, Any], package: str = "yfinance") -> str | None:
    sv = manifest.get("source_version")
    if isinstance(sv, dict):
        return sv.get(package)
    return sv if isinstance(sv, str) else None


def build_spec(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Acquisition parameters taken from the frozen manifest, not from script defaults."""
    params = manifest.get("parameters") or {}
    start = params.get("start") or manifest["requested_start"]
    end = params.get("end") or manifest["requested_end_exclusive"]
    spec: dict[str, Any] = {
        "kind": config["kind"],
        "dataset_id": manifest["dataset_id"],
        "start": start,
        "end": end,
        "interval": params.get("interval") or manifest.get("interval") or "1d",
    }
    spec["symbols"] = list(manifest.get("requested_universe") or manifest["symbols"])
    return spec


def spec_cache_key(spec: dict[str, Any]) -> str:
    keep = {k: v for k, v in spec.items() if k != "dataset_id"}
    return json.dumps(keep, sort_keys=True)


# -------------------------------------------------------------------------- fetching


def fetch_prices(spec: dict[str, Any], out_dir: Path, repo_root: Path) -> dict[str, Any]:
    """Download prices with the repository's own functions; no manifest is ever written."""
    manifests = repo_root / "data" / "manifests"
    try:
        out_dir.resolve().relative_to(repo_root.resolve())
    except ValueError:
        pass
    else:
        raise FetchError(f"raw output directory {out_dir} is inside the repository")
    src = str(repo_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from quant_sentiment.acquisition import download_prices, validate_prices, write_csv
    except ImportError as exc:
        raise FetchError(f"cannot import quant_sentiment.acquisition: {exc}") from exc

    before = snapshot_dir(manifests)
    retrieval = utc_now()
    failed: list[str] = []
    for symbol in spec["symbols"]:
        try:
            frame = download_prices(symbol, start=spec["start"], end=spec["end"])
            validate_prices(symbol, frame)
            write_csv(out_dir / f"{symbol}.csv", frame)
        except Exception as exc:  # network, empty frame or validation failure
            failed.append(f"{symbol}: {type(exc).__name__}: {exc}")
    if snapshot_dir(manifests) != before:
        raise FetchError("data/manifests changed during the check; aborting")
    if failed:
        raise FetchError(f"{len(failed)} symbol(s) failed: {failed[:5]}")
    return {
        "retrieval_timestamp_utc": retrieval,
        "excluded": [],
        "script": "quant_sentiment.acquisition",
    }


def fetch_dataset(
    spec: dict[str, Any], out_dir: Path, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    return fetch_prices(spec, out_dir, repo_root)


# ------------------------------------------------------------------------ comparison


def _check(expected: Any, fresh: Any) -> dict[str, Any]:
    return {
        "expected": expected,
        "fresh": fresh,
        "match": None if expected is None else expected == fresh,
    }


def compare_file(
    name: str, manifest: dict[str, Any], expected_sha: str | None, fresh_dir: Path
) -> dict[str, Any]:
    stem = name[: -len(".csv")] if name.endswith(".csv") else name
    exp = expected_metadata(manifest, stem)
    path = fresh_dir / name
    if not path.is_file():
        return {
            "file": name,
            "fresh_file_present": False,
            "match": False,
            "match_basis": "fresh file missing",
            "sha256": {"expected": expected_sha, "fresh": None, "match": False},
        }
    fresh = summarize_csv(path)
    fresh_sha = sha256_file(path)
    sha = _check(expected_sha, fresh_sha)
    result: dict[str, Any] = {
        "file": name,
        "fresh_file_present": True,
        "match": sha["match"],
        "match_basis": "sha256" if expected_sha else "not recorded (no per-file hash in manifest)",
        "sha256": sha,
        "rows": _check(exp["rows"], fresh["rows"]),
        "first_date": _check(exp["first_date"], fresh["first_date"]),
        "last_date": _check(exp["last_date"], fresh["last_date"]),
        "missing_values": _check(exp["missing_values"], fresh["missing_values"]),
        "columns": _check(exp["columns"], fresh["columns"]),
    }
    if exp["actions"] is not None:
        result["actions"] = _check(exp["actions"], fresh["actions"])
    return result


def _brief(value: Any) -> Any:
    """Compact form of a recorded value for the human-readable drift summary."""
    if isinstance(value, dict):
        return {k: v.get("nonzero_count") if isinstance(v, dict) else v for k, v in value.items()}
    return value


_META_KEYS = ("rows", "first_date", "last_date", "missing_values", "columns", "actions")


def compare_dataset(
    manifest: dict[str, Any], fresh_dir: Path, fresh_excluded: list[Any]
) -> dict[str, Any]:
    exp_sha = expected_hashes(manifest)
    exp_canon = (manifest.get("sha256") or {}).get(CANONICAL_KEY)
    if exp_sha:
        names = sorted(exp_sha)
    else:
        names = [f"{s}.csv" for s in manifest["symbols"]]
    fresh_names = sorted(p.name for p in fresh_dir.glob("*.csv"))
    files = [compare_file(n, manifest, exp_sha.get(n), fresh_dir) for n in names]
    unexpected = [n for n in fresh_names if n not in names]

    fresh_hashes = {n: sha256_file(fresh_dir / n) for n in fresh_names}
    single = len(exp_sha) == 1 and exp_canon == next(iter(exp_sha.values()))
    if single and len(fresh_hashes) == 1:
        fresh_canon = next(iter(fresh_hashes.values()))
    else:
        fresh_canon = canonical_dataset_hash(fresh_hashes) if fresh_hashes else None
    canonical = _check(exp_canon, fresh_canon)

    exp_excluded = manifest.get("excluded_at_freeze")
    excluded = None
    if exp_excluded is not None:
        exp_syms = sorted(e.get("symbol") for e in exp_excluded)
        fresh_syms = sorted(e.get("symbol") for e in fresh_excluded)
        excluded = {**_check(exp_syms, fresh_syms), "fresh_reasons": fresh_excluded}

    issues: list[str] = []
    missing_files = [f["file"] for f in files if not f["fresh_file_present"]]
    if missing_files:
        issues.append(f"{len(missing_files)} frozen file(s) not produced: {missing_files[:5]}")
    value_differing = [f for f in files if f["match"] is False and f["fresh_file_present"]]
    if value_differing:
        issues.append(f"{len(value_differing)} of {len(files)} files differ in sha256")
    if unexpected:
        issues.append(f"{len(unexpected)} file(s) produced that the manifest does not list")
    if canonical["match"] is False:
        issues.append("dataset_canonical hash differs")
    for key in _META_KEYS:
        bad = [
            f"{f['file']} {_brief(f[key]['expected'])} -> {_brief(f[key]['fresh'])}"
            for f in files
            if key in f and f[key]["match"] is False
        ]
        if bad:
            issues.append(f"{key} differs in {len(bad)} file(s), e.g. {'; '.join(bad[:3])}")
    if excluded is not None and excluded["match"] is False:
        issues.append(f"excluded symbols differ: {excluded['expected']} -> {excluded['fresh']}")

    has_evidence = bool(exp_sha) or exp_canon is not None
    if not issues and not has_evidence:
        classification = CLASS_UNVERIFIABLE
    elif issues:
        classification = CLASS_DRIFT
        if value_differing and not any(
            f[k]["match"] is False for f in files for k in _META_KEYS if k in f
        ):
            issues.append(
                "row counts, date ranges, missing-value counts, action counts and columns are "
                "unchanged: bytes differ in values only"
            )
        issues.append(DRIFT_EXPLANATION)
    else:
        classification = CLASS_MATCH
    return {
        "classification": classification,
        "files": files,
        "files_matched": sum(1 for f in files if f["match"] is True),
        "files_total": len(files),
        "unexpected_fresh_files": unexpected,
        "fresh_file_hashes": fresh_hashes,
        "dataset_canonical": canonical,
        "excluded_at_freeze": excluded,
        "drift_summary": issues,
    }


# --------------------------------------------------------------------------- report


def package_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for pkg in ("yfinance", "pandas", "numpy"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def run_check(
    configs: tuple[dict[str, Any], ...],
    manifest_dir: Path,
    fetch: Callable[[dict[str, Any], Path], dict[str, Any]],
    work_root: Path,
    only: list[str] | None = None,
) -> dict[str, Any]:
    versions = package_versions()
    results: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}
    for n, config in enumerate(configs):
        if only and config["manifest"] not in only and config["manifest"][:-5] not in only:
            continue
        entry: dict[str, Any] = {"manifest": f"data/manifests/{config['manifest']}"}
        try:
            manifest = json.loads((manifest_dir / config["manifest"]).read_text(encoding="utf-8"))
            entry.update(
                dataset=manifest["dataset_id"],
                source=manifest.get("source"),
                manifest_retrieval_time=manifest.get("retrieval_timestamp_utc"),
                manifest_freeze_time=manifest.get("freeze_timestamp_utc"),
            )
            spec = build_spec(config, manifest)
            entry["acquisition"] = {k: spec[k] for k in ("start", "end", "interval", "symbols")} | {
                "via": config["via"]
            }
            key = spec_cache_key(spec)
            if key not in cache:
                out_dir = work_root / f"fetch{n}"
                out_dir.mkdir(parents=True, exist_ok=True)
                cache[key] = {
                    **fetch(spec, out_dir),
                    "dir": out_dir,
                    "first_dataset": manifest["dataset_id"],
                }
            fetched = cache[key]
            entry["fresh_retrieval_time"] = fetched["retrieval_timestamp_utc"]
            if fetched["first_dataset"] != manifest["dataset_id"]:
                entry["fresh_data_shared_with"] = fetched["first_dataset"]
            entry["status"] = "fetched"
            entry["source_version"] = {
                "manifest": source_version_of(manifest),
                "fresh": versions.get("yfinance") if manifest.get("source") == "yfinance" else None,
            }
            entry["source_version"]["match"] = (
                entry["source_version"]["manifest"] == entry["source_version"]["fresh"]
                if manifest.get("source") == "yfinance"
                else None
            )
            entry.update(compare_dataset(manifest, fetched["dir"], fetched["excluded"]))
            if config.get("caveat"):
                entry["caveat"] = config["caveat"]
        except (FetchError, OSError, KeyError, ValueError) as exc:
            entry["status"] = "fetch_failed"
            entry["classification"] = CLASS_FAILED
            entry["error"] = f"{type(exc).__name__}: {exc}"
        results.append(entry)
    failed = [r for r in results if r["status"] != "fetched"]
    return {
        "generated_at_utc": utc_now(),
        "packages": versions,
        "datasets": results,
        "not_checked": list(NOT_CHECKED),
        "summary": {
            "datasets": len(results),
            "match": sum(1 for r in results if r["classification"] == CLASS_MATCH),
            "drift": sum(1 for r in results if r["classification"] == CLASS_DRIFT),
            "fetch_failed": len(failed),
        },
        "exit_code": 2 if failed or not results else 0,
    }


def render_markdown(report: dict[str, Any]) -> str:
    pk = report["packages"]
    lines = [
        "## Data reproduction check",
        "",
        f"yfinance {pk.get('yfinance')}, pandas {pk.get('pandas')}, Python {pk.get('python')}. "
        "A fresh fetch is compared with each frozen manifest. Yahoo restates adjusted prices, so "
        "DATA DRIFT is expected and is not a failure of the published analysis.",
        "",
        "| Dataset | Manifest fetched | Fresh fetched | Files matching | Result |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in report["datasets"]:
        if r["status"] == "fetched":
            hashed = sum(1 for f in r["files"] if f["match"] is not None)
            files = f"{r['files_matched']}/{hashed}"
            if hashed < r["files_total"]:
                files += f" ({r['files_total'] - hashed} without a recorded hash)"
            if r["dataset_canonical"]["match"] is not None:
                files += ", canonical " + ("same" if r["dataset_canonical"]["match"] else "differs")
        else:
            files = "n/a"
        lines.append(
            f"| {r.get('dataset', r['manifest'])} | {r.get('manifest_retrieval_time', '')} "
            f"| {r.get('fresh_retrieval_time', '')} | {files} | {r['classification']} |"
        )
    for r in report["not_checked"]:
        lines.append(f"| {r['dataset']} | | | n/a | not checked ({r['reason']}) |")
    notes = []
    for r in report["datasets"]:
        if r["status"] != "fetched":
            notes.append(f"- {r.get('dataset', r['manifest'])}: {r.get('error')}")
            continue
        details = [s for s in r["drift_summary"] if s != DRIFT_EXPLANATION]
        if r.get("caveat"):
            details.append(r["caveat"])
        if details:
            notes.append(f"- {r['dataset']}: " + " | ".join(details))
    if notes:
        lines += ["", "### Details", "", *notes]
    s = report["summary"]
    lines += [
        "",
        f"{s['datasets']} dataset(s): {s['match']} match, {s['drift']} drift, "
        f"{s['fetch_failed']} fetch failure(s). Only fetch failures fail the job.",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------------------ CLI


def main(
    argv: list[str] | None = None,
    *,
    fetch: Callable[[dict[str, Any], Path], dict[str, Any]] | None = None,
    repo_root: Path = REPO_ROOT,
    configs: tuple[dict[str, Any], ...] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json-out", type=Path, default=Path("reproduce_data_report.json"))
    parser.add_argument("--md-out", type=Path, default=None, help="also write the Markdown table")
    parser.add_argument("--only", nargs="+", help="manifest file names (or dataset ids) to check")
    parser.add_argument("--work-dir", type=Path, default=None, help="parent for temp downloads")
    args = parser.parse_args(argv)

    manifest_dir = repo_root / "data" / "manifests"
    for out in (args.json_out, args.md_out):
        if out is not None and manifest_dir.resolve() in out.resolve().parents:
            print(f"ERROR: refusing to write {out} inside data/manifests", file=sys.stderr)
            return 2
    if fetch is None:

        def fetch(spec: dict[str, Any], out_dir: Path) -> dict[str, Any]:
            return fetch_dataset(spec, out_dir, repo_root)

    tmp_parent = args.work_dir
    if tmp_parent is not None:
        tmp_parent.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix="reproduce-data-", dir=tmp_parent))
    try:
        work_root.resolve().relative_to(repo_root.resolve())
    except ValueError:
        pass
    else:
        shutil.rmtree(work_root, ignore_errors=True)
        print("ERROR: work directory must be outside the repository", file=sys.stderr)
        return 2
    try:
        report = run_check(configs or DATASETS, manifest_dir, fetch, work_root, args.only)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    if args.md_out is not None:
        args.md_out.write_text(markdown, encoding="utf-8")
    print(markdown)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
