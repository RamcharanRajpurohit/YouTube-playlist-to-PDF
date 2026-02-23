"""Optional content verification using a secondary LLM."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verifying a chapter's content."""

    chapter_title: str
    is_valid: bool
    warnings: List[str]
    suggestions: List[str]


class ContentVerifier:
    """Cross-check chapter content against the original transcript using an LLM.

    This uses a *different* LLM from the one that wrote the chapter, which
    reduces the chance that systematic hallucinations pass undetected.
    """

    _SYSTEM_PROMPT = (
        "You are a meticulous technical reviewer. Your job is to compare a "
        "chapter draft against the original transcript source material. "
        "Identify any statements in the chapter that:\n"
        "1. Are NOT supported by the transcript.\n"
        "2. Could be technically inaccurate.\n"
        "3. Appear to be hallucinated or invented.\n\n"
        "Respond in this exact format:\n"
        "VALID: yes/no\n"
        "WARNINGS:\n- <warning 1>\n- <warning 2>\n"
        "SUGGESTIONS:\n- <suggestion 1>\n- <suggestion 2>\n"
        "If there are no issues, respond with VALID: yes and empty lists."
    )

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def verify(
        self, chapter_title: str, chapter_text: str, source_transcript: str
    ) -> VerificationResult:
        """Verify *chapter_text* against *source_transcript*."""
        prompt = (
            f"## Chapter Title\n{chapter_title}\n\n"
            f"## Chapter Draft\n{chapter_text}\n\n"
            f"## Original Transcript Source\n{source_transcript}\n\n"
            "Please compare the chapter draft against the original transcript "
            "source and identify any issues."
        )

        try:
            raw = self._llm.generate(prompt, system_prompt=self._SYSTEM_PROMPT)
            return self._parse_response(chapter_title, raw)
        except Exception as exc:
            logger.error("Verification failed for '%s': %s", chapter_title, exc)
            return VerificationResult(
                chapter_title=chapter_title,
                is_valid=False,
                warnings=[f"Verification call failed: {exc}"],
                suggestions=[],
            )

    @staticmethod
    def _parse_response(chapter_title: str, raw: str) -> VerificationResult:
        lines = raw.strip().splitlines()
        is_valid = True
        warnings: List[str] = []
        suggestions: List[str] = []
        section = None

        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith("VALID:"):
                val = stripped.split(":", 1)[1].strip().lower()
                is_valid = val in ("yes", "true")
            elif stripped.upper().startswith("WARNINGS:"):
                section = "warnings"
            elif stripped.upper().startswith("SUGGESTIONS:"):
                section = "suggestions"
            elif stripped.startswith("- "):
                item = stripped[2:].strip()
                if section == "warnings" and item:
                    warnings.append(item)
                elif section == "suggestions" and item:
                    suggestions.append(item)

        return VerificationResult(
            chapter_title=chapter_title,
            is_valid=is_valid,
            warnings=warnings,
            suggestions=suggestions,
        )
