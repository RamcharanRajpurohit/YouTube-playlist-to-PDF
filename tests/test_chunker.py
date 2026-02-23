"""Unit tests for TextChunker."""

import pytest

from src.config import ChunkingConfig, Config
from src.processing.chunker import TextChunker


def _make_chunker(target_tokens=100, overlap_sentences=2):
    """Create a chunker with the given settings."""
    config = Config()
    config.chunking = ChunkingConfig(
        target_chunk_tokens=target_tokens,
        overlap_sentences=overlap_sentences,
    )
    return TextChunker(config)


class TestTokenCounting:
    def test_empty_string(self):
        chunker = _make_chunker()
        assert chunker.count_tokens("") == 0

    def test_nonempty_string(self):
        chunker = _make_chunker()
        tokens = chunker.count_tokens("Hello world, this is a test.")
        assert tokens > 0


class TestChunking:
    def test_short_text_single_chunk(self):
        chunker = _make_chunker(target_tokens=1000)
        chunks = chunker.chunk("This is a short text.")
        assert len(chunks) == 1
        assert chunks[0].index == 0

    def test_long_text_multiple_chunks(self):
        chunker = _make_chunker(target_tokens=50, overlap_sentences=1)
        # Generate enough text to need multiple chunks
        sentences = ". ".join([f"This is sentence number {i}" for i in range(50)])
        sentences += "."
        chunks = chunker.chunk(sentences)
        assert len(chunks) > 1

    def test_chunks_respect_token_budget(self):
        chunker = _make_chunker(target_tokens=100, overlap_sentences=1)
        sentences = ". ".join([f"This is a test sentence number {i}" for i in range(100)])
        sentences += "."
        chunks = chunker.chunk(sentences)
        for chunk in chunks:
            # Allow some tolerance for overlap
            assert chunk.token_count <= 150, (
                f"Chunk {chunk.index} has {chunk.token_count} tokens, exceeds budget"
            )

    def test_chunk_indices_are_sequential(self):
        chunker = _make_chunker(target_tokens=50, overlap_sentences=1)
        sentences = ". ".join([f"Sentence {i}" for i in range(30)])
        sentences += "."
        chunks = chunker.chunk(sentences)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_empty_text(self):
        chunker = _make_chunker()
        chunks = chunker.chunk("")
        assert len(chunks) == 0

    def test_overlap_context(self):
        chunker = _make_chunker(target_tokens=50, overlap_sentences=2)
        sentences = ". ".join([f"Unique sentence number {i}" for i in range(40)])
        sentences += "."
        chunks = chunker.chunk(sentences)
        if len(chunks) >= 2:
            # The second chunk should contain some text from the end of the first
            # (overlap). This is a heuristic check.
            first_text = chunks[0].text
            second_text = chunks[1].text
            # At least some overlap should exist
            first_words = set(first_text.split()[-10:])
            second_words = set(second_text.split()[:10])
            assert len(first_words & second_words) > 0, "Expected overlap between chunks"
