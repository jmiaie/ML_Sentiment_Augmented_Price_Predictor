"""SEC EDGAR filing-listing helpers: submission fetch, date/form filtering,
and primary-document archive URLs.

Shared by both acquisition scripts (v1 exploratory and v2 authoritative) so
neither depends on the other for this plumbing -- see ``hashing.py``'s
docstring for why that coupling was a real defect, not just style.
"""

from __future__ import annotations

from typing import Any

from .sec_http import sec_get_json as _sec_get_json


def fetch_company_filings(cik: str) -> list[dict[str, Any]]:
    """Return filing metadata rows from submissions recent + historical shards."""
    padded = cik.zfill(10)
    submissions = _sec_get_json(f"https://data.sec.gov/submissions/CIK{padded}.json")
    rows = _filings_from_block(submissions.get("filings", {}).get("recent", {}))
    for shard in submissions.get("filings", {}).get("files", []) or []:
        name = shard.get("name")
        if not name:
            continue
        shard_url = f"https://data.sec.gov/submissions/{name}"
        block = _sec_get_json(shard_url)
        rows.extend(_filings_from_block(block))
    return rows


def _filings_from_block(block: dict[str, Any]) -> list[dict[str, Any]]:
    if not block:
        return []
    forms = block.get("form") or []
    n = len(forms)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append(
            {
                "form": forms[i],
                "filingDate": (block.get("filingDate") or [None] * n)[i],
                "acceptanceDateTime": (block.get("acceptanceDateTime") or [None] * n)[i],
                "accessionNumber": (block.get("accessionNumber") or [None] * n)[i],
                "primaryDocument": (block.get("primaryDocument") or [None] * n)[i],
                "reportDate": (block.get("reportDate") or [None] * n)[i],
            }
        )
    return rows


def filter_filings(
    rows: list[dict[str, Any]],
    *,
    start: str,
    end_exclusive: str,
    forms: tuple[str, ...],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        form = row.get("form")
        filing_date = row.get("filingDate") or ""
        if form not in forms:
            continue
        if not filing_date:
            continue
        if not (filing_date >= start and end_exclusive > filing_date):
            continue
        if not row.get("accessionNumber") or not row.get("primaryDocument"):
            continue
        if not row.get("acceptanceDateTime"):
            continue
        out.append(row)
    # Stable order
    out.sort(key=lambda r: (r["filingDate"], r["accessionNumber"]))
    return out


def filing_archive_url(cik: str, accession: str, primary_document: str) -> str:
    cik_int = int(cik)
    acc_nodash = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary_document}"
