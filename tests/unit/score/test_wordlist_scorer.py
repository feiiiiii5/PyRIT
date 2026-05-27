# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from pyrit.score import (
    OfcomCategory,
    TrueFalseCompositeScorer,
    TrueFalseScoreAggregator,
    WordListMatchMode,
    WordListScorer,
)

_LDNOOBW_SAMPLE = "bitch\nasshole\nf**k\nass\n"

_SLURS_SAMPLE = "Banderite\nBarbarian\nBoche \n"

_OFCOM_SAMPLE = (
    "general\tCrap\t1\n"
    "general\tShit\t2\n"
    "general\tBullshit\t2\n"
    "general\tFuck\t4\n"
    "sexual\tShag\t2\n"
    "raceethnic\tSomeSlur\t3\n"
    "raceethnic\tMildOnly\t1\n"
)


async def test_init_rejects_empty_terms(patch_central_database):
    with pytest.raises(ValueError, match="non-empty"):
        WordListScorer(terms=[], category="x")


async def test_init_rejects_only_whitespace(patch_central_database):
    with pytest.raises(ValueError, match="non-empty"):
        WordListScorer(terms=["   ", "\t", ""], category="x")


async def test_word_mode_does_not_match_substring(patch_central_database):
    """`ass` must not fire inside `class` when match_mode=WORD."""
    scorer = WordListScorer(terms=["ass"], category="profanity")
    score = (await scorer.score_text_async(text="I have a class today"))[0]
    assert score.get_value() is False


async def test_word_mode_matches_whole_word(patch_central_database):
    scorer = WordListScorer(terms=["ass"], category="profanity")
    score = (await scorer.score_text_async(text="don't be an ass"))[0]
    assert score.get_value() is True
    assert score.score_category == ["profanity"]


async def test_substring_mode_matches_inside_word(patch_central_database):
    scorer = WordListScorer(
        terms=["ass"],
        category="profanity",
        match_mode=WordListMatchMode.SUBSTRING,
    )
    score = (await scorer.score_text_async(text="I have a class today"))[0]
    assert score.get_value() is True


async def test_case_insensitive_default(patch_central_database):
    scorer = WordListScorer(terms=["Bitch"], category="profanity")
    score_upper = (await scorer.score_text_async(text="STOP THAT BITCH"))[0]
    score_lower = (await scorer.score_text_async(text="stop that bitch"))[0]
    assert score_upper.get_value() is True
    assert score_lower.get_value() is True


async def test_case_sensitive_opt_in(patch_central_database):
    scorer = WordListScorer(terms=["Bitch"], category="profanity", case_sensitive=True)
    score_upper = (await scorer.score_text_async(text="STOP THAT BITCH"))[0]
    score_exact = (await scorer.score_text_async(text="stop that Bitch"))[0]
    assert score_upper.get_value() is False
    assert score_exact.get_value() is True


async def test_regex_metacharacters_in_terms_are_escaped(patch_central_database):
    """A term like `f**k` must be treated literally, not as `f\\w\\wk`."""
    scorer = WordListScorer(terms=["f**k"], category="profanity")
    matches = (await scorer.score_text_async(text="this is f**k literally"))[0]
    no_match = (await scorer.score_text_async(text="this is fxxk"))[0]
    assert matches.get_value() is True
    assert no_match.get_value() is False


async def test_terms_are_deduplicated_and_stripped(patch_central_database):
    scorer = WordListScorer(terms=["  bad  ", "bad", "BAD", "worse"], category="x")
    score = (await scorer.score_text_async(text="that was bad"))[0]
    assert score.get_value() is True
    # Internal list dedupes case-sensitively but matching is case-insensitive,
    # so "bad" and "BAD" survive as distinct entries; the count is 3.
    assert len(scorer._terms) == 3


async def test_terms_sorted_longest_first(patch_central_database):
    scorer = WordListScorer(
        terms=["son of a bitch", "bitch", "ass"],
        category="x",
        match_mode=WordListMatchMode.SUBSTRING,
    )
    assert scorer._terms[0] == "son of a bitch"


async def test_category_propagates_into_score(patch_central_database):
    scorer = WordListScorer(terms=["foo"], category="ofcom-raceethnic")
    score = (await scorer.score_text_async(text="this has foo in it"))[0]
    assert score.score_category == ["ofcom-raceethnic"]


async def test_rationale_names_the_category(patch_central_database):
    scorer = WordListScorer(terms=["foo"], category="profanity")
    score = (await scorer.score_text_async(text="contains foo"))[0]
    assert "profanity" in score.score_rationale


async def test_identifier_includes_metadata(patch_central_database):
    scorer = WordListScorer(
        terms=["alpha", "beta"],
        category="profanity",
        case_sensitive=True,
        match_mode=WordListMatchMode.SUBSTRING,
    )
    identifier = scorer.get_identifier()
    assert identifier.params["category"] == "profanity"
    assert identifier.params["term_count"] == 2
    assert identifier.params["match_mode"] == "substring"
    assert identifier.params["case_sensitive"] is True


async def test_composite_or_across_two_wordlists(patch_central_database):
    from pyrit.memory.central_memory import CentralMemory
    from pyrit.models import MessagePiece

    profanity = WordListScorer(terms=["bitch"], category="profanity")
    slurs = WordListScorer(terms=["banderite"], category="slurs")
    composite = TrueFalseCompositeScorer(
        scorers=[profanity, slurs],
        aggregator=TrueFalseScoreAggregator.OR,
    )

    memory = CentralMemory.get_memory_instance()

    slur_piece = MessagePiece(
        role="assistant", original_value="he called me a Banderite", conversation_id="c1", sequence=1
    )
    memory.add_message_pieces_to_memory(message_pieces=[slur_piece])
    only_slur = (await composite.score_async(slur_piece.to_message()))[0]
    assert only_slur.get_value() is True

    benign_piece = MessagePiece(role="assistant", original_value="ordinary sentence", conversation_id="c2", sequence=1)
    memory.add_message_pieces_to_memory(message_pieces=[benign_piece])
    neither = (await composite.score_async(benign_piece.to_message()))[0]
    assert neither.get_value() is False


# ---------- Fetch + cache + classmethod loader tests ----------


@pytest.fixture
def lexicon_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect lexicon downloads into a per-test temp directory."""
    import pyrit.score.true_false.wordlist_scorer as module

    monkeypatch.setattr(module, "DB_DATA_PATH", tmp_path)
    return tmp_path / "lexicons"


def _mock_response(text: str) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.text = text
    response.raise_for_status = MagicMock()
    return response


async def test_from_ldnoobw_en_fetches_and_scores(patch_central_database, lexicon_cache_dir):
    with patch("requests.get", return_value=_mock_response(_LDNOOBW_SAMPLE)) as mock_get:
        scorer = WordListScorer.from_ldnoobw_en()
        positive = (await scorer.score_text_async(text="you bitch"))[0]
        negative = (await scorer.score_text_async(text="hello there"))[0]

    assert positive.get_value() is True
    assert negative.get_value() is False
    assert positive.score_category == ["ldnoobw"]
    mock_get.assert_called_once()
    assert mock_get.call_args.args[0] == WordListScorer._LDNOOBW_EN_URL


async def test_from_slurs_en_fetches_and_scores(patch_central_database, lexicon_cache_dir):
    with patch("requests.get", return_value=_mock_response(_SLURS_SAMPLE)) as mock_get:
        scorer = WordListScorer.from_slurs_en()
        positive = (await scorer.score_text_async(text="they're a Banderite"))[0]
        negative = (await scorer.score_text_async(text="ordinary sentence"))[0]

    assert positive.get_value() is True
    assert negative.get_value() is False
    assert positive.score_category == ["slurs"]
    mock_get.assert_called_once()


async def test_from_ofcom_category_filters_strength(patch_central_database, lexicon_cache_dir):
    with patch("requests.get", return_value=_mock_response(_OFCOM_SAMPLE)):
        scorer = WordListScorer.from_ofcom_category(category=OfcomCategory.GENERAL, min_strength=2)
        crap = (await scorer.score_text_async(text="that was crap"))[0]
        shit = (await scorer.score_text_async(text="that was shit"))[0]

    # "Crap" has strength 1, below the min_strength=2 cutoff -> should NOT fire.
    assert crap.get_value() is False
    # "Shit" has strength 2 -> should fire.
    assert shit.get_value() is True
    assert shit.score_category == ["ofcom-general"]


async def test_from_ofcom_category_other_category_excluded(patch_central_database, lexicon_cache_dir):
    """raceethnic terms must not appear in the general scorer."""
    with patch("requests.get", return_value=_mock_response(_OFCOM_SAMPLE)):
        scorer = WordListScorer.from_ofcom_category(category=OfcomCategory.GENERAL)
        race_term = (await scorer.score_text_async(text="someslur"))[0]

    assert race_term.get_value() is False


async def test_from_ofcom_raises_when_no_terms_pass_filter(patch_central_database, lexicon_cache_dir):
    with patch("requests.get", return_value=_mock_response(_OFCOM_SAMPLE)):
        with pytest.raises(ValueError, match="No Ofcom terms"):
            WordListScorer.from_ofcom_category(
                category=OfcomCategory.RACEETHNIC,
                min_strength=4,
            )


async def test_cached_lexicon_skips_second_fetch(patch_central_database, lexicon_cache_dir):
    """Second call to from_ldnoobw_en() must hit the cache, not the network."""
    with patch("requests.get", return_value=_mock_response(_LDNOOBW_SAMPLE)) as mock_get:
        WordListScorer.from_ldnoobw_en()
        WordListScorer.from_ldnoobw_en()

    assert mock_get.call_count == 1


async def test_http_error_propagates(patch_central_database, lexicon_cache_dir):
    failing_response = MagicMock(spec=requests.Response)
    failing_response.raise_for_status.side_effect = requests.HTTPError("boom")
    with patch("requests.get", return_value=failing_response):
        with pytest.raises(requests.HTTPError, match="boom"):
            WordListScorer.from_ldnoobw_en()


# ---------- Pure-function tests for helpers (no network, no memory) ----------


def test_build_pattern_word_mode_uses_word_boundaries():
    pattern = WordListScorer._build_pattern(
        terms=["foo", "bar"],
        match_mode=WordListMatchMode.WORD,
        case_sensitive=False,
    )
    assert pattern.startswith("(?i)")
    assert r"\b(?:" in pattern
    assert r")\b" in pattern
    compiled = re.compile(pattern)
    assert compiled.search("a foo b") is not None
    assert compiled.search("food") is None  # word boundary saves us


def test_build_pattern_substring_mode_omits_word_boundaries():
    pattern = WordListScorer._build_pattern(
        terms=["foo"],
        match_mode=WordListMatchMode.SUBSTRING,
        case_sensitive=True,
    )
    assert not pattern.startswith("(?i)")
    assert r"\b" not in pattern
    compiled = re.compile(pattern)
    assert compiled.search("food") is not None


def test_parse_ofcom_tsv_filters_by_category_and_strength():
    terms = WordListScorer._parse_ofcom_tsv(
        tsv_text=_OFCOM_SAMPLE,
        category=OfcomCategory.GENERAL,
        min_strength=2,
    )
    assert "Shit" in terms
    assert "Bullshit" in terms
    assert "Fuck" in terms
    assert "Crap" not in terms  # strength 1
    assert "Shag" not in terms  # different category
