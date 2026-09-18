"""Reproducible simplified financial polarity lexicon for Directive #9.

This is intentionally a small, fully auditable word list embedded in-repo so
CI and offline tests need no network dictionary download. It is **not** the
full Loughran–McDonald Master Dictionary.

Provenance / limitations:
- Words selected for transparency and common financial narrative polarity.
- Scores are deterministic bag-of-words polarity in [-1, 1].
- Not a claim that this lexicon equals academic LM coverage or that scores
  are economically meaningful alpha signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LEXICON_ID = "sentiment_d9_financial_polarity_v1"
LEXICON_VERSION = "1"

POSITIVE_WORDS = frozenset(
    {
        "advantage",
        "beat",
        "beneficial",
        "benefit",
        "boost",
        "confidence",
        "efficient",
        "exceed",
        "exceeded",
        "exceeds",
        "favor",
        "favorable",
        "gain",
        "gains",
        "growth",
        "improve",
        "improved",
        "improvement",
        "improvements",
        "increase",
        "increased",
        "innovation",
        "opportunity",
        "optimistic",
        "outperform",
        "outperformed",
        "positive",
        "profit",
        "profitable",
        "progress",
        "record",
        "strengthen",
        "strengthened",
        "strong",
        "success",
        "successful",
        "upside",
    }
)

NEGATIVE_WORDS = frozenset(
    {
        "adverse",
        "challenge",
        "challenges",
        "decline",
        "declined",
        "decrease",
        "decreased",
        "delay",
        "delayed",
        "disappoint",
        "disappointed",
        "downturn",
        "impairment",
        "lawsuit",
        "litigation",
        "loss",
        "losses",
        "negative",
        "penalty",
        "risk",
        "risks",
        "shortfall",
        "slowdown",
        "uncertainty",
        "unfavorable",
        "weak",
        "weakness",
        "writedown",
        "writeoff",
    }
)

_TOKEN_RE = re.compile(r"[a-z]+", re.IGNORECASE)


@dataclass(frozen=True)
class LexiconScore:
    sentiment_score: float
    confidence: float
    positive_hits: int
    negative_hits: int
    token_count: int
    lexicon_id: str = LEXICON_ID
    lexicon_version: str = LEXICON_VERSION


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def score_text(text: str) -> LexiconScore:
    tokens = tokenize(text)
    if not tokens:
        return LexiconScore(
            sentiment_score=0.0,
            confidence=0.0,
            positive_hits=0,
            negative_hits=0,
            token_count=0,
        )

    positive_hits = sum(1 for token in tokens if token in POSITIVE_WORDS)
    negative_hits = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    denom = positive_hits + negative_hits
    if denom == 0:
        return LexiconScore(
            sentiment_score=0.0,
            confidence=0.0,
            positive_hits=0,
            negative_hits=0,
            token_count=len(tokens),
        )

    polarity = (positive_hits - negative_hits) / denom
    # Confidence rises with polar-hit density but stays in [0, 1].
    confidence = min(1.0, denom / max(20.0, 0.002 * len(tokens)))
    return LexiconScore(
        sentiment_score=float(max(-1.0, min(1.0, polarity))),
        confidence=float(confidence),
        positive_hits=int(positive_hits),
        negative_hits=int(negative_hits),
        token_count=len(tokens),
    )
