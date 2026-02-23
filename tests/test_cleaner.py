"""Unit tests for TranscriptCleaner."""

import pytest

from src.config import Config, ProcessingConfig
from src.processing.cleaner import TranscriptCleaner


def _make_cleaner(filler_words=None):
    """Create a cleaner with the given filler words."""
    config = Config()
    config.processing = ProcessingConfig(
        filler_words=filler_words or ["um", "uh", "you know", "basically", "like"]
    )
    return TranscriptCleaner(config)


class TestFillerRemoval:
    def test_removes_single_filler(self):
        cleaner = _make_cleaner()
        result = cleaner.clean("This is um a test sentence.")
        assert "um" not in result.lower().split()

    def test_removes_multi_word_filler(self):
        cleaner = _make_cleaner()
        result = cleaner.clean("So you know the thing is we need to build a model.")
        assert "you know" not in result

    def test_removes_multiple_fillers(self):
        cleaner = _make_cleaner()
        result = cleaner.clean("Um basically you know we are uh building a network.")
        assert "um" not in result.lower().split()
        assert "basically" not in result.lower().split()
        assert "uh" not in result.lower().split()

    def test_preserves_technical_content(self):
        cleaner = _make_cleaner()
        result = cleaner.clean(
            "The transformer architecture uses self-attention mechanisms."
        )
        assert "transformer" in result
        assert "self-attention" in result


class TestWhitespaceNormalization:
    def test_collapses_spaces(self):
        cleaner = _make_cleaner([])
        result = cleaner.clean("Hello    world   this  is  a  test.")
        assert "    " not in result
        assert "   " not in result

    def test_removes_space_before_punctuation(self):
        cleaner = _make_cleaner([])
        result = cleaner.clean("Hello , world . This is a test .")
        assert " ," not in result
        assert " ." not in result


class TestSentenceCasing:
    def test_capitalizes_after_period(self):
        cleaner = _make_cleaner([])
        result = cleaner.clean("First sentence. second sentence.")
        assert "Second" in result

    def test_capitalizes_after_question_mark(self):
        cleaner = _make_cleaner([])
        result = cleaner.clean("Is this a question? yes it is.")
        assert "Yes" in result


class TestParagraphization:
    def test_creates_paragraphs(self):
        cleaner = _make_cleaner([])
        # 12 sentences should create ~2 paragraphs
        sentences = ". ".join([f"Sentence {i}" for i in range(12)]) + "."
        result = cleaner.clean(sentences)
        assert "\n\n" in result
