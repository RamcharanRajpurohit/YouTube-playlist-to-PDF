"""Ollama local LLM provider — runs open-source models on CPU, no API key needed.

Requires Ollama to be installed (the setup.sh script handles this automatically).
If Ollama is not running, the provider will attempt to start it automatically.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time

import tiktoken
from openai import OpenAI

from src.config import LLMModelConfig
from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Local Ollama provider using its OpenAI-compatible API.

    Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1
    so we can reuse the OpenAI SDK with no API key required.

    Auto-starts Ollama and pulls the model if needed.
    """

    _BASE_URL = "http://localhost:11434/v1"

    def __init__(self, model_config: LLMModelConfig, base_url: str | None = None) -> None:
        self._base_url = base_url or self._BASE_URL
        self._model = model_config.model
        self._max_output_tokens = model_config.max_output_tokens
        self._temperature = model_config.temperature
        self._max_context = model_config.max_context_tokens
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoder = tiktoken.get_encoding("gpt2")

        # Ensure Ollama is ready before first call
        self._ensure_ollama_ready()
        self._client = OpenAI(api_key="ollama", base_url=self._base_url)

    @property
    def name(self) -> str:
        return f"Ollama ({self._model})"

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

    # ── Auto-setup helpers ───────────────────────────────────────

    def _ensure_ollama_ready(self) -> None:
        """Check that Ollama is installed, running, and the model is pulled."""
        # 1. Check if Ollama is installed
        if not shutil.which("ollama"):
            logger.error(
                "Ollama is not installed. Run ./setup.sh or install from https://ollama.com/download"
            )
            raise RuntimeError(
                "Ollama is not installed. Run ./setup.sh to set up everything automatically."
            )

        # 2. Start Ollama if not running
        if not self._is_server_running():
            logger.info("Ollama server not running — starting it now...")
            self._start_server()

        # 3. Pull model if not available
        if not self._is_model_available():
            logger.info("Model '%s' not found — pulling it now (first time only)...", self._model)
            self._pull_model()

    def _is_server_running(self) -> bool:
        """Check if Ollama server is responding."""
        try:
            import urllib.request
            req = urllib.request.Request("http://localhost:11434/api/tags")
            urllib.request.urlopen(req, timeout=3)
            return True
        except Exception:
            return False

    def _start_server(self) -> None:
        """Start Ollama server in the background."""
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Wait for server to be ready
            for _ in range(15):
                time.sleep(1)
                if self._is_server_running():
                    logger.info("Ollama server started successfully.")
                    return
            logger.warning("Ollama server may not have started. Continuing anyway...")
        except Exception as exc:
            logger.error("Failed to start Ollama server: %s", exc)
            raise RuntimeError("Could not start Ollama. Run 'ollama serve' manually.") from exc

    def _is_model_available(self) -> bool:
        """Check if the target model is already pulled."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return self._model in result.stdout
        except Exception:
            return False

    def _pull_model(self) -> None:
        """Pull the model from Ollama registry."""
        try:
            logger.info("Downloading model '%s'... This may take a few minutes.", self._model)
            subprocess.run(
                ["ollama", "pull", self._model],
                check=True,
                # No timeout — large models (e.g. Mistral 4.4 GB) can take
                # 15–30+ min on slow connections. User can Ctrl+C to abort.
            )
            logger.info("Model '%s' pulled successfully.", self._model)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to pull model '{self._model}': {exc}") from exc
