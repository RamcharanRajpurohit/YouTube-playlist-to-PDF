"""Factory for creating LLM provider instances."""

from __future__ import annotations

from src.config import Config
from src.llm.base import LLMProvider
from src.llm.gemini_provider import GeminiProvider


class LLMFactory:
    """Create LLMProvider instances from config."""

    @staticmethod
    def create(provider_name: str, config: Config) -> LLMProvider:
        """Return an LLMProvider for *provider_name*."""
        provider_name = provider_name.lower()

        if provider_name == "gemini":
            model_config = config.llm_configs.get("gemini")
            if model_config is None:
                raise ValueError("No Gemini model config found in config.")
            return GeminiProvider(
                api_key=config.gemini_api_key,
                model_config=model_config,
            )

        raise ValueError(f"Unknown LLM provider: {provider_name!r}")
