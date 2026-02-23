"""Google Gemini LLM provider."""

from __future__ import annotations

import logging

import tiktoken
from google import genai
from google.genai import types

from src.config import LLMModelConfig
from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Gemini provider using the google-genai SDK."""

    def __init__(self, api_key: str, model_config: LLMModelConfig) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self._client = genai.Client(api_key=api_key)
        self._model = model_config.model
        self._max_output_tokens = model_config.max_output_tokens
        self._temperature = model_config.temperature
        self._max_context = model_config.max_context_tokens
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoder = tiktoken.get_encoding("gpt2")

    @property
    def name(self) -> str:
        return f"Gemini ({self._model})"

    @property
    def max_context_tokens(self) -> int:
        return self._max_context

    def count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text, disallowed_special=()))

    def _call(self, prompt: str, system_prompt: str = "") -> str:
        config = types.GenerateContentConfig(
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        return response.text or ""
