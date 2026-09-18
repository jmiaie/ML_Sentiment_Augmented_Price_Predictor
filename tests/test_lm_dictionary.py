import math

from quant_sentiment.lm_dictionary import dictionary_provenance, score_text


def test_score_is_deterministic() -> None:
    text = "The company reported strong growth but faces significant litigation risk."
    a = score_text(text)
    b = score_text(text)
    assert a == b


def test_empty_text_scores_all_zero() -> None:
    f = score_text("")
    assert f.token_count == 0
    assert f.negative_fraction == 0.0
    assert f.positive_fraction == 0.0
    assert f.net_tone == 0.0
    assert f.log_word_count == 0.0


def test_purely_positive_text_has_positive_net_tone() -> None:
    text = "growth improvement strong record profitable increase " * 5
    f = score_text(text)
    assert f.positive_fraction > 0
    assert f.net_tone > 0


def test_purely_negative_text_has_negative_net_tone() -> None:
    text = "decline loss weak litigation risk uncertainty writedown " * 5
    f = score_text(text)
    assert f.negative_fraction > 0
    assert f.net_tone < 0


def test_fractions_are_token_count_normalized() -> None:
    # "excellent"/"profitable" are confirmed members of the real LM Positive
    # set (verified directly against the dictionary, not assumed); "filler"
    # and "widget" are confirmed NOT filtered by the tokenizer's LM stoplist.
    text = "excellent profitable filler filler filler widget widget widget"
    f = score_text(text)
    assert f.token_count == 8
    assert f.positive_count == 2
    assert f.positive_fraction == 2 / 8


def test_log_word_count_matches_log1p_of_token_count() -> None:
    # Number words ("one", "two", ...) are in the LM tokenizer's own
    # DatesandNumbers stoplist and are dropped -- use plain nouns instead.
    text = "filler widget quality banana filler"
    f = score_text(text)
    assert f.token_count == 5
    assert math.isclose(f.log_word_count, math.log1p(5))


def test_dictionary_provenance_reports_real_categories_and_is_not_vendored() -> None:
    prov = dictionary_provenance()
    assert prov.package == "pysentiment2"
    assert len(prov.word_list_sha256) == 64
    assert prov.n_words_total > 50_000  # the real LM Master Dictionary has ~86k rows
    for category in ("Negative", "Positive", "Uncertainty", "Litigious", "Constraining"):
        assert prov.n_words_by_category[category] > 0
    assert "not vendored" in prov.redistribution_note
