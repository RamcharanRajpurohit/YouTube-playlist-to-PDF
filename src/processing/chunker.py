"""Text chunking for large transcripts — token-aware splitting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import tiktoken

from src.config import Config


@dataclass
class TextChunk:
    """A single chunk of text sized to fit within an LLM context window."""

    index: int
    text: str
    token_count: int


class TextChunker:
    """Split text into chunks that respect token budgets and sentence boundaries.

    Each chunk includes a small overlap (configurable number of trailing
    sentences from the previous chunk) so the LLM retains context.
    """

    def __init__(self, config: Config) -> None:
        self._target_tokens = config.chunking.target_chunk_tokens
        self._overlap_sentences = config.chunking.overlap_sentences
        # Use cl100k_base encoder (GPT-4 / general purpose) for counting
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoder = tiktoken.get_encoding("gpt2")

    def count_tokens(self, text: str) -> int:
        """Return the token count for *text*."""
        return len(self._encoder.encode(text, disallowed_special=()))

    def chunk(self, text: str) -> List[TextChunk]:
        """Split *text* into token-bounded chunks on sentence boundaries."""
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[TextChunk] = []
        current_sentences: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            stok = self.count_tokens(sentence)

            # If a single sentence exceeds budget, force it into its own chunk
            if stok > self._target_tokens:
                if current_sentences:
                    chunks.append(self._make_chunk(len(chunks), current_sentences))
                    current_sentences = []
                    current_tokens = 0
                chunks.append(self._make_chunk(len(chunks), [sentence]))
                continue

            if current_tokens + stok > self._target_tokens and current_sentences:
                chunks.append(self._make_chunk(len(chunks), current_sentences))
                # Overlap: carry the last N sentences into the next chunk
                overlap = current_sentences[-self._overlap_sentences :]
                current_sentences = list(overlap)
                current_tokens = sum(self.count_tokens(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_tokens += stok

        if current_sentences:
            chunks.append(self._make_chunk(len(chunks), current_sentences))

        return chunks

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Naive sentence splitter that handles common abbreviations."""
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _make_chunk(self, index: int, sentences: List[str]) -> TextChunk:
        joined = " ".join(sentences)
        return TextChunk(
            index=index,
            text=joined,
            token_count=self.count_tokens(joined),
        )
