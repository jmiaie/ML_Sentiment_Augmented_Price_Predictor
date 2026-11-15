"""Fail-closed tests for the sentiment-study publication verifier scripts.

The scripts derive their repo root from ``__file__``, so every test runs the
scripts from a throwaway ``git clone`` of the repository under pytest's temp
directory (local clone: no network, committed state only; full history, as the
``hashcheck`` mode needs it). Mutations are applied to that clone and always
restored; the real repository tree is never written to. The pack's figure is a
deterministic stdlib SVG, so no renderer is involved.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
PACK_REL = Path("publication/sentiment-study")
SCRIPTS_REL = PACK_REL / "scripts"

pytestmark = pytest.mark.skipif(
    not (REPO / SCRIPTS_REL / "publication_pack.py").exists(),
    reason="publication pack scripts not present",
)


def _is_shallow(repo: Path) -> bool:
    out = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip() == "true"


# hashcheck re-reads the freeze commit from git history. CI jobs that check out with the
# default depth of 1 do not have it; the jobs with fetch-depth 0 run these tests.
needs_history = pytest.mark.skipif(
    _is_shallow(REPO), reason="needs full git history (shallow checkout)"
)


def _load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO / SCRIPTS_REL / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def clone(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dest: Path = tmp_path_factory.mktemp("sentiment") / "repo"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(REPO), str(dest)],
        check=True,
        capture_output=True,
        text=True,
    )
    return dest


def _run(root: Path, script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / SCRIPTS_REL / script), *args],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _pack(root: Path, mode: str) -> subprocess.CompletedProcess[str]:
    return _run(root, "publication_pack.py", mode)


def _cross(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run(root, "claim_crosscheck.py", *args)


@contextlib.contextmanager
def mutated(path: Path, new: bytes | None) -> Iterator[None]:
    """Replace ``path`` content (or delete it when ``new`` is None); always restore."""
    original = path.read_bytes()
    try:
        if new is None:
            path.unlink()
        else:
            path.write_bytes(new)
        yield
    finally:
        path.write_bytes(original)


def _flip_last_byte(path: Path) -> bytes:
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01
    return bytes(data)


def _repro(root: Path) -> dict:
    data: dict = json.loads((root / PACK_REL / "reproducibility.json").read_text("utf-8"))
    return data


def _append(path: Path, text: str) -> bytes:
    return path.read_bytes() + text.encode("utf-8")


# --------------------------------------------------------------------- pure helpers


def test_sha256_helpers_known_input() -> None:
    mod = _load("pp_under_test", "publication_pack.py")
    abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert mod.sha256_bytes(b"abc") == abc == hashlib.sha256(b"abc").hexdigest()


def test_cell_and_csv_bytes_are_deterministic() -> None:
    mod = _load("pp_under_test2", "publication_pack.py")
    assert mod.cell(None) == "null"
    assert mod.cell(0.1) == "0.1"
    assert mod.cell(3) == "3"
    assert mod.csv_bytes(["a", "b"], [[1, "x,y"]]) == b"a,b\n1,x;y\n"


def test_overclaim_guard_helper_flags_and_passes() -> None:
    mod = _load("cc_under_test", "claim_crosscheck.py")
    assert mod.violations("This is an untouched holdout.")
    assert mod.violations("This is a pristine holdout.")
    assert not mod.violations("This is not an untouched holdout.")
    assert not mod.violations('The phrase "untouched holdout" is prohibited here.')
    # a stray negation elsewhere must not whitelist the overclaim
    assert mod.violations("This is not unusual; the study uses an untouched holdout.")


def test_overclaim_guard_helper_catches_each_banned_token() -> None:
    mod = _load("cc_under_test2", "claim_crosscheck.py")
    for token in mod.BANNED:
        assert mod.violations(f"The result {token} the point."), token


# ------------------------------------------------------------------- committed passes


def test_committed_pack_check_passes(clone: Path) -> None:
    r = _pack(clone, "check")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "check OK" in r.stdout


@needs_history
def test_committed_pack_hashcheck_passes(clone: Path) -> None:
    r = _pack(clone, "hashcheck")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "hashcheck OK" in r.stdout


def test_committed_claim_crosscheck_and_selftest_pass(clone: Path) -> None:
    r = _cross(clone)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "claim cross-check OK" in r.stdout
    s = _cross(clone, "--selftest")
    assert s.returncode == 0, s.stdout + s.stderr
    assert "selftest OK" in s.stdout


def test_invalid_mode_rejected(clone: Path) -> None:
    assert _pack(clone, "bogus").returncode != 0


# ------------------------------------------------------------- publication_pack check


def test_check_fails_on_one_byte_tamper_of_every_source_artifact(clone: Path) -> None:
    for src in _repro(clone)["source_artifacts"]:
        path = clone / src["path"]
        with mutated(path, _flip_last_byte(path)):
            r = _pack(clone, "check")
        # Clean FAIL report, or an uncaught parse error from build(): both fail closed.
        assert r.returncode != 0, f"tamper of {src['path']} not detected"


def test_check_reports_hash_mismatch_for_text_artifact(clone: Path) -> None:
    # Pick a non-JSON artifact if one exists so the clean report path is exercised.
    arts = [s["path"] for s in _repro(clone)["source_artifacts"] if s["path"].endswith(".csv")]
    if not arts:
        pytest.skip("no non-JSON source artifact to exercise the clean report path")
    path = clone / arts[0]
    with mutated(path, _flip_last_byte(path)):
        r = _pack(clone, "check")
    assert r.returncode == 1
    assert "hash mismatch" in r.stdout


def test_check_fails_on_missing_source_artifact(clone: Path) -> None:
    rel = _repro(clone)["source_artifacts"][0]["path"]
    with mutated(clone / rel, None):
        r = _pack(clone, "check")
    assert r.returncode != 0
    assert f"missing source artifact: {rel}" in r.stdout or "FileNotFoundError" in r.stderr


def test_check_fails_on_empty_source_artifact_list(clone: Path) -> None:
    repro_path = clone / PACK_REL / "reproducibility.json"
    doc = json.loads(repro_path.read_text("utf-8"))
    doc["source_artifacts"] = []
    with mutated(repro_path, json.dumps(doc).encode()):
        r = _pack(clone, "check")
    assert r.returncode == 1
    assert "vacuously" in r.stdout


def test_check_fails_on_generated_table_drift(clone: Path) -> None:
    table = clone / PACK_REL / "tables" / "period-metrics.csv"
    with mutated(table, _flip_last_byte(table)):
        r = _pack(clone, "check")
    assert r.returncode == 1
    assert "not reproducible: tables/period-metrics.csv" in r.stdout


def test_check_fails_on_missing_generated_figure(clone: Path) -> None:
    fig = clone / PACK_REL / "figures" / "log-loss-comparison.svg"
    with mutated(fig, None):
        r = _pack(clone, "check")
    assert r.returncode == 1
    assert "missing generated file: figures/log-loss-comparison.svg" in r.stdout


def test_check_fails_on_wrong_citation_hash_in_source_map(clone: Path) -> None:
    smap = clone / PACK_REL / "RESULT-SOURCE-MAP.md"
    text = smap.read_text("utf-8")
    m = re.search(r"^\| C-01 \|.*\| ([0-9a-f]{64}) \|$", text, re.MULTILINE)
    assert m
    with mutated(smap, text.replace(m.group(1), "0" * 64, 1).encode()):
        r = _pack(clone, "check")
    assert r.returncode == 1
    assert "citation C-01" in r.stdout and "hash mismatch" in r.stdout


def test_check_fails_on_citation_row_with_missing_path(clone: Path) -> None:
    smap = clone / PACK_REL / "RESULT-SOURCE-MAP.md"
    text = smap.read_text("utf-8")
    m = re.search(r"^\| C-01 \|[^|]*\| (\S+) \|", text, re.MULTILINE)
    assert m
    with mutated(smap, text.replace(m.group(1), "results/does_not_exist.json", 1).encode()):
        r = _pack(clone, "check")
    assert r.returncode == 1
    assert "missing path" in r.stdout


def test_check_fails_on_uncovered_label_class_row(clone: Path) -> None:
    smap = clone / PACK_REL / "RESULT-SOURCE-MAP.md"
    with mutated(smap, _append(smap, "\n| ZZ-99 | unchecked claim | x | y |\n")):
        r = _pack(clone, "check")
    assert r.returncode == 1
    assert "label class not covered" in r.stdout


def test_check_fails_when_source_map_has_no_citations(clone: Path) -> None:
    smap = clone / PACK_REL / "RESULT-SOURCE-MAP.md"
    with mutated(smap, b"# empty\n"):
        r = _pack(clone, "check")
    assert r.returncode == 1
    assert "vacuously" in r.stdout


# ---------------------------------------------------------- publication_pack hashcheck


def test_hashcheck_fails_on_unknown_sha256_in_pack_doc(clone: Path) -> None:
    doc = clone / PACK_REL / "CASE-STUDY.md"
    with mutated(doc, _append(doc, "\n" + "ab" * 32 + "\n")):
        r = _pack(clone, "hashcheck")
    assert r.returncode == 1
    assert "unknown sha256" in r.stdout


def test_hashcheck_fails_on_undeclared_git_sha(clone: Path) -> None:
    doc = clone / PACK_REL / "CASE-STUDY.md"
    with mutated(doc, _append(doc, "\n" + "1234567890" * 4 + "\n")):
        r = _pack(clone, "hashcheck")
    assert r.returncode == 1
    assert "undeclared git sha" in r.stdout


def test_hashcheck_fails_on_unresolved_abbreviation(clone: Path) -> None:
    doc = clone / PACK_REL / "CASE-STUDY.md"
    with mutated(doc, _append(doc, "\nsee deadbee…\n")):
        r = _pack(clone, "hashcheck")
    assert r.returncode == 1
    assert "unresolved abbreviation" in r.stdout


def test_hashcheck_fails_on_unanchored_canonical_dataset_hash(clone: Path) -> None:
    repro_path = clone / PACK_REL / "reproducibility.json"
    doc = json.loads(repro_path.read_text("utf-8"))
    doc["dataset"]["canonical_dataset_hashes"][0]["must_appear_in"].append("no/such/anchor.json")
    with mutated(repro_path, json.dumps(doc, indent=2).encode()):
        r = _pack(clone, "hashcheck")
    assert r.returncode == 1
    assert "not anchored" in r.stdout


def test_hashcheck_fails_on_unresolvable_declared_git_sha(clone: Path) -> None:
    repro_path = clone / PACK_REL / "reproducibility.json"
    text = repro_path.read_text("utf-8")
    sha = _repro(clone)["accepted_evidence"]["git_shas"][0]
    with mutated(repro_path, text.replace(sha, "1" * 40).encode()):
        r = _pack(clone, "hashcheck")
    assert r.returncode == 1
    assert "git object not resolvable" in r.stdout


# ------------------------------------------------------------------ claim_crosscheck


def test_crosscheck_fails_on_tampered_accepted_artifact(clone: Path) -> None:
    path = clone / "results/historical_text_v2/sentiment_hist_text_v2_primary_validation.json"
    with mutated(path, _flip_last_byte(path)):
        r = _cross(clone)
    assert r.returncode != 0


def test_crosscheck_fails_on_altered_paper_value(clone: Path) -> None:
    # The check is membership-based: every printed occurrence must be altered for it to fail.
    art = (
        clone
        / "results/historical_text_v2/sentiment_hist_text_v2_primary_historical_evaluation.json"
    )
    k = json.loads(art.read_text("utf-8"))["result"]["key_metrics"]
    printed = repr(round(k["headline_delta_log_loss_model3_minus_model1"], 8))
    paper = clone / PACK_REL / "TECHNICAL-PAPER.md"
    text = paper.read_text("utf-8")
    assert printed in text
    with mutated(paper, text.replace(printed, "0.12345678").encode()):
        r = _cross(clone)
    assert r.returncode != 0
    assert "ASSERT FAIL" in r.stdout


def test_crosscheck_fails_on_unnegated_overclaim_in_prose(clone: Path) -> None:
    doc = clone / PACK_REL / "PACK-STATUS.md"
    with mutated(doc, _append(doc, "\nThis is an untouched holdout.\n")):
        r = _cross(clone)
    assert r.returncode != 0
    assert "overclaim token" in r.stdout


def test_crosscheck_fails_on_altered_ledger_hash(clone: Path) -> None:
    ledger = clone / "research/experiment-ledger.csv"
    art = clone / "results/historical_text_v2/sentiment_hist_text_v2_primary_validation.json"
    digest = hashlib.sha256(art.read_bytes()).hexdigest()
    text = ledger.read_text("utf-8")
    assert digest in text
    with mutated(ledger, text.replace(digest, "0" * 64, 1).encode()):
        r = _cross(clone)
    assert r.returncode != 0
    assert "ledger artifact_sha256" in r.stdout


def test_crosscheck_fails_on_missing_deliverable(clone: Path) -> None:
    doc = clone / PACK_REL / "CLAIM-REGISTER.md"
    with mutated(doc, None):
        r = _cross(clone)
    assert r.returncode != 0
