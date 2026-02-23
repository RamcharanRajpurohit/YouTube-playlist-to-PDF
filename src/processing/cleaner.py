"""Transcript cleaning — filler removal, normalization, paragraph joining."""

from __future__ import annotations

import re
from typing import List

from src.config import Config
from src.transcript.fetcher import TranscriptSegment


class TranscriptCleaner:
    """Clean raw transcript text into readable prose-ready paragraphs."""

    def __init__(self, config: Config) -> None:
        self._filler_words = config.processing.filler_words
        # Build a regex that matches filler phrases as whole words/phrases
        if self._filler_words:
            escaped = [re.escape(w) for w in sorted(self._filler_words, key=len, reverse=True)]
            self._filler_re = re.compile(
                r"\b(?:" + "|".join(escaped) + r")\b",
                re.IGNORECASE,
            )
        else:
            self._filler_re = None

    def clean(self, full_text: str) -> str:
        """Apply all cleaning steps and return the cleaned text."""
        text = full_text

        # 1. Remove filler words / phrases
        text = self._remove_fillers(text)

        # 2. Normalize whitespace
        text = self._normalize_whitespace(text)

        # 3. Fix casing after sentence boundaries
        text = self._fix_sentence_casing(text)

        # 4. Remove excessive repeated punctuation
        text = re.sub(r"([.!?])\1+", r"\1", text)

        # 5. Group into paragraphs (split on long pauses indicated by
        #    sentence endings followed by topic-shift heuristics)
        text = self._paragraphize(text)

        return text.strip()

    def clean_segments(self, segments: List[TranscriptSegment]) -> str:
        """Join segments into a single string, then clean."""
        full = " ".join(seg.text for seg in segments)
        return self.clean(full)

    # ── internal helpers ─────────────────────────────────────────

    def _remove_fillers(self, text: str) -> str:
        if self._filler_re is None:
            return text
        return self._filler_re.sub("", text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        # Collapse multiple spaces / tabs into one
        text = re.sub(r"[ \t]+", " ", text)
        # Remove space before punctuation
        text = re.sub(r"\s+([.!?,;:])", r"\1", text)
        # Ensure space after punctuation (if followed by a letter)
        text = re.sub(r"([.!?,;:])([A-Za-z])", r"\1 \2", text)
        return text

    @staticmethod
    def _fix_sentence_casing(text: str) -> str:
        """Capitalize the first letter after sentence-ending punctuation."""

        def _capitalize_match(m: re.Match) -> str:
            return m.group(1) + m.group(2).upper()

        return re.sub(r"([.!?]\s+)([a-z])", _capitalize_match, text)

    @staticmethod
    def _paragraphize(text: str) -> str:
        """Insert paragraph breaks roughly every 5–7 sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        paragraphs: List[str] = []
        buffer: List[str] = []
        for sent in sentences:
            buffer.append(sent)
            if len(buffer) >= 6:
                paragraphs.append(" ".join(buffer))
                buffer = []
        if buffer:
            paragraphs.append(" ".join(buffer))
        return "\n\n".join(paragraphs)
