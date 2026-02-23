"""Abstract base class for LLM providers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

# Default retry parameters
_MAX_RETRIES = 3


class LLMProvider(ABC):
    """Interface that every LLM backend must implement."""

    @abstractmethod
    def _call(self, prompt: str, system_prompt: str = "") -> str:
        """Raw API call — subclasses implement this."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return approximate token count for *text*."""

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        """Maximum context window size in tokens."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""

    # ── public interface (with retry) ────────────────────────────

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_retries: int = _MAX_RETRIES,
    ) -> str:
        """Call the LLM with exponential-backoff retry on transient errors."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                result = self._call(prompt, system_prompt)
                return result
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[%s] Attempt %d/%d failed: %s — retrying immediately",
                    self.name,
                    attempt,
                    max_retries,
                    exc,
                )
        raise RuntimeError(
            f"[{self.name}] All {max_retries} attempts failed."
        ) from last_exc
