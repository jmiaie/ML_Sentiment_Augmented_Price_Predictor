#!/usr/bin/env python3
"""Supplementary prose-level claim checker for the sentiment publication pack.

The frozen ``claim_crosscheck.py`` asserts artifact values but never reads the
pack's prose for arbitrary numbers. This script closes that gap: it extracts
every statistic-like number from the prose documents (file and line) and checks
each *occurrence* independently against the set of numbers the committed result
artifacts, frozen config, manifest, ledger rows and generated tables support.

A prose number is supported when it equals a supported value rounded to the
number of decimals the prose shows (percent tokens also match value x 100;
integers must match a whole-number value exactly). Tokens that legitimately are
not in the artifacts live in ``scripts/prose_claim_allowlist.json`` with file,
token, expected occurrence count and a one-line reason.

Limits. This checks where a number comes from, not what the sentence says about
it. It is strong for precise statistics (several decimals) and weak for short
ones: a token such as ``0.5`` or ``0.031`` passes if any artifact value rounds
to it, and a bare small integer in running text (``7 survivors``) is checked
only when it stands before a recognised count noun or in a table cell. A wrong
small count, or a right number attached to the wrong claim, can still pass.

Standard library only, offline, read-only.

    python3 scripts/prose_claim_check.py [--root DIR] [--allowlist FILE]

Exit 0: all supported. Exit 1: unsupported numbers listed as ``file:line token``.
Exit 2: usage or input error (unknown root, missing file, bad allowlist).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, localcontext
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Per-repo configuration
# --------------------------------------------------------------------------
PACK = "publication/sentiment-study"
PROSE_DOCS = [
    f"{PACK}/{name}"
    for name in (
        "TECHNICAL-PAPER.md",
        "CASE-STUDY.md",
        "PACK-STATUS.md",
        "CLAIM-REGISTER.md",
        "RESULT-SOURCE-MAP.md",
        "QUANT-RED-TEAM.md",
        "CLAIM-RED-TEAM.md",
        "CITATION-RED-TEAM.md",
        "SOURCE-GATE.md",
    )
]
# (root-relative glob, row-filter prefix for ledger-style CSVs or None)
SOURCES: list[tuple[str, str | None]] = [
    ("results/historical_text_v2/sentiment_hist_text_v2_*.json", None),
    ("configs/experiments/sentiment_historical_text_study_v2.yaml", None),
    ("research/experiment-ledger.csv", "sentiment_hist_text_v2"),
    (f"{PACK}/reproducibility.json", None),
    (f"{PACK}/tables/*.csv", None),
]
# Nouns that make a bare integer a checked count ("32 windows").
COUNT_NOUNS = (
    "windows|window|tests|survivors|rows|symbols|pairs|buckets|bucket|folds|resamples|"
    "sessions|events|trades|observations|samples|B|bytes|assets|artifacts|files"
)
DEFAULT_ALLOWLIST = Path(__file__).resolve().parent / "prose_claim_allowlist.json"
# --------------------------------------------------------------------------

NUM_RE = re.compile(
    r"(?<![\w.,])(?P<sign>[+\-−–]?)(?P<int>\d{1,3}(?:,\d{3})+|\d+)"
    r"(?P<frac>\.\d+)?(?P<exp>[eE][+\-−]?\d+)?(?P<pct>%?)(?![\w])"
)
COUNT_RE = re.compile(rf"\s*(?:{COUNT_NOUNS})\b")
NOISE = [
    re.compile(r"https?://\S+"),
    re.compile(r"\]\([^)]*\)"),
    re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?Z?)?"),
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),
    re.compile(r"\b(?:19|20)\d{2}\s*[-–→]+\s*(?:19|20)\d{2}\b"),
    re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{6,64}\b"),
    re.compile(r"§\s*\d+(?:\.\d+)*"),
    re.compile(r"^\s*#+\s*\d+(?:\.\d+)*\.?"),
    re.compile(r"(?<![\w.])[A-Za-z_][\w-]*\d[\w.-]*"),
    re.compile(r"(?i)\b(?:python|version)\s*\d+(?:\.\d+)*\+?"),
    re.compile(r"\b\d+\.\d+\.\d+\b"),
    re.compile(r"\b(?:19|20)\d{2}\b"),
]
CELL_INT = re.compile(r"^\d{1,3}(?:,\d{3})+$|^\d+$")


@dataclass(frozen=True)
class Token:
    file: str
    line: int
    text: str
    value: Decimal
    decimals: int
    sig: int
    percent: bool
    signed: bool
    sci: bool

    @property
    def key(self) -> tuple[str, str]:
        return (self.file, self.text)


class InputError(Exception):
    """A missing or malformed input; reported as exit status 2."""


def _blank(text: str, pattern: re.Pattern[str]) -> str:
    return pattern.sub(lambda m: " " * len(m.group(0)), text)


def clean_line(line: str) -> str:
    """Blank out identifiers, dates, hashes, section and version numbers."""
    for pattern in NOISE:
        line = _blank(line, pattern)
    return line


def _make_token(file: str, lineno: int, m: re.Match[str], text: str) -> Token:
    sign = m.group("sign")
    neg = sign in ("-", "−", "–")
    digits = m.group("int").replace(",", "")
    frac = m.group("frac") or ""
    exp = m.group("exp") or ""
    mant = f"{digits}{frac}"
    value = Decimal(("-" if neg else "") + mant + exp.replace("−", "-"))
    sig = len((digits + frac[1:]).lstrip("0")) or 1
    return Token(
        file=file,
        line=lineno,
        text=text,
        value=value,
        decimals=len(frac) - 1 if frac else 0,
        sig=sig,
        percent=bool(m.group("pct")),
        signed=bool(sign),
        sci=bool(exp),
    )


def extract_tokens(text: str, file: str) -> list[Token]:
    """Statistic-like numbers in ``text``: decimals, percents, grouped integers,
    integers before a count noun, and bare integer table cells."""
    out: list[Token] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        cleaned = clean_line(raw)
        for m in NUM_RE.finditer(cleaned):
            has_dec = bool(m.group("frac") or m.group("exp"))
            grouped = "," in m.group("int")
            counted = COUNT_RE.match(cleaned, m.end()) is not None
            if not (has_dec or grouped or m.group("pct") or counted):
                continue
            out.append(_make_token(file, lineno, m, m.group(0)))
        if raw.lstrip().startswith("|"):
            for cell in raw.strip().strip("|").split("|"):
                c = cell.strip().strip("*`").strip()
                if CELL_INT.match(c) and not (1990 <= int(c.replace(",", "")) <= 2035):
                    cm = NUM_RE.fullmatch(c)
                    if cm and not has_token(out, lineno, c):
                        out.append(_make_token(file, lineno, cm, c))
    return out


def has_token(tokens: list[Token], lineno: int, text: str) -> bool:
    return any(t.line == lineno and t.text == text for t in tokens)


def all_numbers(text: str) -> list[float]:
    """Every number in free text (configs), without the prose-token filters."""
    vals: list[float] = []
    for raw in text.splitlines():
        cleaned = clean_line(raw)
        for m in NUM_RE.finditer(cleaned):
            vals.append(float(_make_token("", 0, m, m.group(0)).value))
    return vals


# --------------------------------------------------------------------------
# Supported-number collection
# --------------------------------------------------------------------------
def _leaves(obj: Any, out: list[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        if obj == obj and abs(obj) != float("inf"):
            out.append(float(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            # confidence level encoded in a key name: "..._ci95" supports "95%"
            out.extend(float(n) for n in re.findall(r"_ci(\d{2})(?![0-9])", k))
            _leaves(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _leaves(v, out)


def _aggregates(obj: Any, out: list[float]) -> None:
    """Counts and sums a reader could derive: len(list), sums of numeric dict
    values, per-key sums over lists of dicts."""
    if isinstance(obj, list):
        out.append(float(len(obj)))
        dicts = [d for d in obj if isinstance(d, dict)]
        keys = {k for d in dicts for k in d}
        for k in keys:
            vals = [
                d[k] for d in dicts if isinstance(d.get(k), (int, float))
                and not isinstance(d.get(k), bool)
            ]
            if len(vals) >= 2:
                out.append(float(sum(vals)))
        for v in obj:
            _aggregates(v, out)
    elif isinstance(obj, dict):
        nums = [
            v for v in obj.values() if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if len(nums) >= 2:
            out.append(float(sum(nums)))
        out.append(float(len(obj)))
        for v in obj.values():
            _aggregates(v, out)


def _float_or_none(cell: str) -> float | None:
    try:
        v = float(cell)
    except ValueError:
        return None
    return v if v == v and abs(v) != float("inf") else None


def _csv_support(path: Path, prefix: str | None) -> tuple[list[float], list[dict[str, Any]]]:
    vals: list[float] = []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if prefix is not None:
        rows = [r for r in rows if (r.get("experiment_id") or "").startswith(prefix)]
    cols: dict[str, list[float]] = {}
    for row in rows:
        for key, cell in row.items():
            cell = cell or ""
            if cell[:1] in "{[":
                try:
                    _leaves(json.loads(cell), vals)
                except json.JSONDecodeError:
                    pass
                continue
            num = _float_or_none(cell.strip())
            if num is not None:
                vals.append(num)
                cols.setdefault(key, []).append(num)
    for col in cols.values():
        if len(col) >= 2:
            vals.append(float(sum(col)))
    vals.append(float(len(rows)))
    return vals, [dict(r) for r in rows]


def source_paths(root: Path) -> list[Path]:
    """Every existing source file (used by tests to build temp copies)."""
    found: list[Path] = []
    for pattern, _ in SOURCES:
        found.extend(sorted(root.glob(pattern)))
    return found


def derived_support(vals: list[float]) -> list[float]:
    """Pack-specific derived values (none needed beyond the generic aggregates)."""
    return []


def collect_support(root: Path) -> list[float]:
    vals: list[float] = []
    for pattern, prefix in SOURCES:
        files = sorted(root.glob(pattern))
        if not files:
            raise InputError(f"no source artifact matches {pattern}")
        for path in files:
            try:
                vals.append(float(path.stat().st_size))  # sizes quoted as "N B"
                if path.suffix == ".json":
                    obj = json.loads(path.read_text(encoding="utf-8"))
                    _leaves(obj, vals)
                    _aggregates(obj, vals)
                elif path.suffix == ".csv":
                    cvals, _ = _csv_support(path, prefix)
                    vals.extend(cvals)
                else:
                    vals.extend(all_numbers(path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                raise InputError(f"cannot read {path}: {exc}") from exc
    vals.extend(derived_support(vals))
    return vals


class Support:
    """Rounded-value lookup over the supported numbers."""

    def __init__(self, values: Iterable[float]) -> None:
        self.values = [v for v in values if abs(v) < 1e15]
        self._cache: dict[tuple[int, bool], set[Decimal]] = {}
        self._whole = {Decimal(int(v)) for v in self.values if v == int(v)}
        self._sci = [v for v in self.values if v != 0]

    def _quantized(self, decimals: int, scale100: bool) -> set[Decimal]:
        key = (decimals, scale100)
        if key not in self._cache:
            q = Decimal(1).scaleb(-decimals)
            found: set[Decimal] = set()
            with localcontext() as ctx:
                ctx.prec = 80
                for v in self.values:
                    d = Decimal(repr(v)) * (100 if scale100 else 1)
                    for mode in (ROUND_HALF_UP, ROUND_HALF_EVEN):
                        found.add(d.quantize(q, rounding=mode))
            self._cache[key] = found
        return self._cache[key]

    def matches(self, tok: Token) -> bool:
        cands = {tok.value} if tok.signed else {tok.value, -tok.value}
        if tok.sci:
            fmt = f".{max(tok.sig - 1, 0)}e"
            want = {format(float(c), fmt) for c in cands}
            return any(format(v, fmt) in want for v in self._sci)
        if tok.decimals == 0 and not tok.percent:
            return any(c in self._whole for c in cands)
        with localcontext() as ctx:
            ctx.prec = 80
            if any(c in self._quantized(tok.decimals, False) for c in cands):
                return True
            return tok.percent and any(c in self._quantized(tok.decimals, True) for c in cands)


# --------------------------------------------------------------------------
# Allowlist, checking, CLI
# --------------------------------------------------------------------------
def load_allowlist(path: Path) -> dict[tuple[str, str], int]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data["entries"]
        allow: dict[tuple[str, str], int] = {}
        for e in entries:
            if not str(e.get("reason", "")).strip():
                raise ValueError(f"allowlist entry without reason: {e}")
            allow[(e["file"], e["token"])] = int(e.get("count", 1))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise InputError(f"cannot load allowlist {path}: {exc}") from exc
    return allow


@dataclass
class Report:
    extracted: int
    supported: int
    allowlisted: int
    known_discrepancies: int
    unsupported: list[Token]
    stale: list[tuple[str, str]]
    files: int


def check(root: Path, allowlist: Path) -> Report:
    if not root.is_dir():
        raise InputError(f"root is not a directory: {root}")
    texts: dict[str, str] = {}
    for rel in PROSE_DOCS:
        path = root / rel
        if not path.is_file():
            raise InputError(f"prose document missing: {rel}")
        texts[rel] = path.read_text(encoding="utf-8")
    support = Support(collect_support(root))
    allow = load_allowlist(allowlist)
    reasons = _reasons(allowlist)
    used: Counter[tuple[str, str]] = Counter()
    tokens: list[Token] = []
    unsupported: list[Token] = []
    supported = allowlisted = known = 0
    for rel, text in texts.items():
        for tok in extract_tokens(text, rel):
            tokens.append(tok)
            if support.matches(tok):
                supported += 1
            elif used[tok.key] < allow.get(tok.key, 0):
                used[tok.key] += 1
                allowlisted += 1
                known += reasons.get(tok.key, "").startswith("KNOWN DISCREPANCY")
            else:
                unsupported.append(tok)
    stale = [k for k, n in allow.items() if used[k] < n]
    return Report(len(tokens), supported, allowlisted, known, unsupported, stale, len(texts))


def _reasons(path: Path) -> dict[tuple[str, str], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {(e["file"], e["token"]): e["reason"] for e in data["entries"]}


def main(argv: list[str] | None = None) -> int:
    default_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--root", type=Path, default=default_root, help="repository root")
    ap.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    args = ap.parse_args(argv)
    try:
        rep = check(args.root.resolve(), args.allowlist)
    except InputError as exc:
        print(f"prose claim check ERROR: {exc}", file=sys.stderr)
        return 2
    for rel, tok in rep.stale:
        print(f"warning: stale allowlist entry {rel} {tok}", file=sys.stderr)
    if rep.unsupported:
        print(f"prose claim check FAILED: {len(rep.unsupported)} unsupported number(s)")
        for t in rep.unsupported:
            print(f"{t.file}:{t.line} {t.text}")
        return 1
    print(
        f"prose claim check OK: {rep.extracted} numbers in {rep.files} documents; "
        f"{rep.supported} supported by artifacts, {rep.allowlisted} allowlisted "
        f"({rep.known_discrepancies} known discrepancies)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
