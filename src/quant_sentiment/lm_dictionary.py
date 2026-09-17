"""Real Loughran-McDonald Master Dictionary features for Directive #9 D9-D
AUTHORITATIVE (v2).

v1 (``lexicon.py``) used a small in-repo fabricated word list, explicitly
labeled non-LM-complete. D9-D's spec requires the actual academic LM Master
Dictionary. Directive #9's spec text: "Do not redistribute the full
dictionary unless permitted" -- so this module never vendors the word list
into this repository. It loads the dictionary at runtime from the
third-party PyPI package ``pysentiment2`` (MIT-licensed wrapper per its own
PyPI metadata; the bundled ``LM.csv`` word list itself is the standard
academic Loughran-McDonald Master Dictionary, https://sraf.nd.edu -- this
package is a redistribution of it, not an Anthropic/this-program
redistribution). ``dictionary_provenance()`` reports the exact installed
package version and the word-list file's own sha256 for the dataset
manifest, so the dependency is fully disclosed and reproducible.

Word-category membership matches ``pysentiment2.lm.LM``'s own approach
(same Porter-stemmed tokenizer for both the dictionary's own words and the
text being scored, so stemmed-token lookups stay internally consistent) --
but extends it to the Uncertainty/Litigious/Constraining categories that
``pysentiment2.lm.LM.get_score`` itself does not expose (only
Positive/Negative/Polarity/Subjectivity), by reading those columns
directly from the same underlying ``LM.csv``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from pysentiment2.base import STATIC_PATH
from pysentiment2.lm import LM

LM_DATASET_ID = "lm_master_dictionary_2026_v1"
_CSV_PATH = f"{STATIC_PATH}/LM.csv"
_CATEGORY_COLUMNS = ("Negative", "Positive", "Uncertainty", "Litigious", "Constraining")


@dataclass(frozen=True)
class LMDictionaryProvenance:
    dataset_id: str
    source: str
    package: str
    package_version: str
    word_list_path: str
    word_list_sha256: str
    n_words_total: int
    n_words_by_category: dict[str, int]
    redistribution_note: str


@dataclass(frozen=True)
class LMFeatures:
    token_count: int
    negative_count: int
    positive_count: int
    uncertainty_count: int
    litigious_count: int
    constraining_count: int
    negative_fraction: float
    positive_fraction: float
    uncertainty_fraction: float
    litigious_fraction: float
    constraining_fraction: float
    net_tone: float
    log_word_count: float


@lru_cache(maxsize=1)
def _tokenizer() -> LM:
    """``LM()`` construction re-reads and re-stems the whole ~86k-row LM.csv
    (see ``pysentiment2.base.BaseDict.__init__`` -> ``init_dict``) -- cache
    a single shared instance rather than paying that cost on every call
    (``score_text`` runs once per filing event, so this matters at
    12-issuer/multi-year scale)."""
    return LM()


@lru_cache(maxsize=1)
def _load_category_sets() -> dict[str, frozenset[str]]:
    """Build stemmed word sets per LM category, mirroring
    ``pysentiment2.lm.LM.init_dict``'s own Positive/Negative construction
    (``tokenize_first`` stems each dictionary word so lookups against a
    stemmed document tokenizer stay consistent) -- extended here to the
    three categories that package's own ``get_score`` does not compute."""
    lm = _tokenizer()
    data = pd.read_csv(_CSV_PATH)
    sets: dict[str, frozenset[str]] = {}
    for column in _CATEGORY_COLUMNS:
        words = data.query(f"{column} > 0")["Word"].apply(lm.tokenize_first).dropna()
        sets[column] = frozenset(words)
    return sets


@lru_cache(maxsize=1)
def dictionary_provenance() -> LMDictionaryProvenance:
    import importlib.metadata

    data = pd.read_csv(_CSV_PATH)
    sets = _load_category_sets()
    digest = hashlib.sha256(open(_CSV_PATH, "rb").read()).hexdigest()
    return LMDictionaryProvenance(
        dataset_id=LM_DATASET_ID,
        source="Loughran-McDonald Master Dictionary (https://sraf.nd.edu), "
        "redistributed by the pysentiment2 PyPI package",
        package="pysentiment2",
        package_version=importlib.metadata.version("pysentiment2"),
        word_list_path=_CSV_PATH,
        word_list_sha256=digest,
        n_words_total=int(data.shape[0]),
        n_words_by_category={col: len(sets[col]) for col in _CATEGORY_COLUMNS},
        redistribution_note=(
            "The word list is not vendored/committed into this repository -- loaded at "
            "runtime from the installed pysentiment2 dependency, per Directive #9's own "
            "instruction not to redistribute the full dictionary unless permitted."
        ),
    )


def score_text(text: str) -> LMFeatures:
    """Pre-specified normalized LM features for one filing's primary
    document text: negative/positive/uncertainty/litigious/constraining
    fraction (category token count / total token count), net tone
    ((positive - negative) / total token count), and log word count
    (``log1p(token_count)``, so an empty document scores 0.0 rather than
    raising on log(0))."""
    lm = _tokenizer()
    sets = _load_category_sets()
    tokens = lm.tokenize(text)
    token_count = len(tokens)

    if token_count == 0:
        return LMFeatures(
            token_count=0,
            negative_count=0,
            positive_count=0,
            uncertainty_count=0,
            litigious_count=0,
            constraining_count=0,
            negative_fraction=0.0,
            positive_fraction=0.0,
            uncertainty_fraction=0.0,
            litigious_fraction=0.0,
            constraining_fraction=0.0,
            net_tone=0.0,
            log_word_count=0.0,
        )

    counts = {col: sum(1 for t in tokens if t in sets[col]) for col in _CATEGORY_COLUMNS}
    negative_count = counts["Negative"]
    positive_count = counts["Positive"]
    uncertainty_count = counts["Uncertainty"]
    litigious_count = counts["Litigious"]
    constraining_count = counts["Constraining"]

    return LMFeatures(
        token_count=token_count,
        negative_count=negative_count,
        positive_count=positive_count,
        uncertainty_count=uncertainty_count,
        litigious_count=litigious_count,
        constraining_count=constraining_count,
        negative_fraction=negative_count / token_count,
        positive_fraction=positive_count / token_count,
        uncertainty_fraction=uncertainty_count / token_count,
        litigious_fraction=litigious_count / token_count,
        constraining_fraction=constraining_count / token_count,
        net_tone=(positive_count - negative_count) / token_count,
        log_word_count=math.log1p(token_count),
    )
