"""Groq LLM provider via OpenAI-compatible endpoint."""

from __future__ import annotations

import logging

import tiktoken
from openai import OpenAI

from src.config import LLMModelConfig
from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """Groq provider using the OpenAI SDK pointed at Groq's API."""

    _BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str, model_config: LLMModelConfig) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")
        self._client = OpenAI(api_key=api_key, base_url=self._BASE_URL)
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
        return f"Groq ({self._model})"

    @property
    def max_context_tokens(self) -> int:
        return self._max_context

    def count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _call(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_output_tokens,
            temperature=self._temperature,
        )
        choice = response.choices[0]
        return choice.message.content or ""
