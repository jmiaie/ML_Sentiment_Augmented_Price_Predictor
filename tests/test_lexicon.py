"""Offline tests for Directive #9 polarity lexicon."""

from __future__ import annotations

from quant_sentiment.lexicon import NEGATIVE_WORDS, POSITIVE_WORDS, score_text


def test_score_positive_text() -> None:
    scored = score_text("Strong growth and profitable gains improved results.")
    assert scored.sentiment_score > 0
    assert scored.positive_hits >= 1
    assert scored.token_count > 0


def test_score_negative_text() -> None:
    scored = score_text("Material losses and adverse litigation increased risk.")
    assert scored.sentiment_score < 0
    assert scored.negative_hits >= 1


def test_score_empty() -> None:
    scored = score_text("")
    assert scored.sentiment_score == 0.0
    assert scored.confidence == 0.0


def test_lexicon_disjoint() -> None:
    assert POSITIVE_WORDS.isdisjoint(NEGATIVE_WORDS)
