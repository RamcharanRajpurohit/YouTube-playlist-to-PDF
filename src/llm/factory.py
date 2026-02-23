"""Factory for creating LLM provider instances."""

from __future__ import annotations

from src.config import Config
from src.llm.base import LLMProvider
from src.llm.gemini_provider import GeminiProvider
from src.llm.groq_provider import GroqProvider
from src.llm.ollama_provider import OllamaProvider


class LLMFactory:
    """Create LLMProvider instances from config."""

    @staticmethod
    def create(provider_name: str, config: Config) -> LLMProvider:
        """Return an LLMProvider for *provider_name* ('gemini', 'groq', or 'ollama')."""
        provider_name = provider_name.lower()

        if provider_name == "gemini":
            model_config = config.llm_configs.get("gemini")
            if model_config is None:
                raise ValueError("No Gemini model config found in config.")
            return GeminiProvider(
                api_key=config.gemini_api_key,
                model_config=model_config,
            )

        if provider_name == "groq":
            model_config = config.llm_configs.get("groq")
            if model_config is None:
                raise ValueError("No Groq model config found in config.")
            return GroqProvider(
                api_key=config.groq_api_key,
                model_config=model_config,
            )

        if provider_name == "ollama":
            model_config = config.llm_configs.get("ollama")
            if model_config is None:
                raise ValueError("No Ollama model config found in config.")
            return OllamaProvider(model_config=model_config)

        raise ValueError(f"Unknown LLM provider: {provider_name!r}")
