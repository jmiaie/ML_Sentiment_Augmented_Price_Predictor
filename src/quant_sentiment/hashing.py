"""Content-addressed hashing for frozen dataset files.

Shared by both acquisition scripts (v1 exploratory and v2 authoritative) so
neither depends on the other for this plumbing -- previously defined only
in ``scripts/acquire_edgar_8k_yf_megacap_daily.py`` (v1) and imported from
there by the v2 script, a real coupling defect: v1 is classified
EXPLORATORY / NON-CONFORMING LEGACY EVIDENCE, so retiring or refactoring it
would silently break v2's freeze hashing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_dataset_hash(file_hashes: dict[str, str]) -> str:
    payload = "\n".join(f"{k}:{v}" for k, v in sorted(file_hashes.items())) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
