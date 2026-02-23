"""Centralized configuration loader."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv


@dataclass
class LLMModelConfig:
    """Settings for a single LLM provider."""

    model: str
    max_output_tokens: int = 8192
    temperature: float = 0.3
    max_context_tokens: int = 131072


@dataclass
class ChunkingConfig:
    target_chunk_tokens: int = 12000
    overlap_sentences: int = 3


@dataclass
class TranscriptConfig:
    languages: List[str] = field(default_factory=lambda: ["en"])
    cache_dir: str = "output/transcripts"
    cookies_file: str = ""
    delay_seconds: float = 2.0


@dataclass
class ProcessingConfig:
    filler_words: List[str] = field(default_factory=list)
    # Number of chapters to generate in parallel (tune to your API tier's rate limits)
    # Free tier: keep at 3-4. Paid tier: can go up to 10.
    parallel_chapters: int = 5


@dataclass
class OutputConfig:
    manuscript_md: str = "output/manuscript.md"
    manuscript_pdf: str = "output/manuscript.pdf"
    chapters_dir: str = "output/chapters"


@dataclass
class VerificationConfig:
    enabled: bool = False
    provider: str = "gemini"


@dataclass
class Config:
    """Top-level application configuration."""

    primary_provider: str = "gemini"
    llm_configs: Dict[str, LLMModelConfig] = field(default_factory=dict)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    transcript: TranscriptConfig = field(default_factory=TranscriptConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)

    # API keys (loaded from env)
    gemini_api_key: str = ""

    @classmethod
    def load(cls, config_path: str = "config/default.yaml") -> "Config":
        """Load configuration from YAML file and environment variables."""
        load_dotenv()

        raw: Dict[str, Any] = {}
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as fh:
                raw = yaml.safe_load(fh) or {}

        llm_section = raw.get("llm", {})
        llm_configs: Dict[str, LLMModelConfig] = {}
        provider_name = "gemini"
        provider_raw = llm_section.get(provider_name, {})
        if provider_raw:
            llm_configs[provider_name] = LLMModelConfig(**provider_raw)

        chunking_raw = raw.get("chunking", {})
        transcript_raw = raw.get("transcript", {})
        processing_raw = raw.get("processing", {})
        output_raw = raw.get("output", {})
        verification_raw = raw.get("verification", {})

        return cls(
            primary_provider=llm_section.get("primary_provider", "gemini"),
            llm_configs=llm_configs,
            chunking=ChunkingConfig(**chunking_raw) if chunking_raw else ChunkingConfig(),
            transcript=TranscriptConfig(**transcript_raw) if transcript_raw else TranscriptConfig(),
            processing=ProcessingConfig(**processing_raw) if processing_raw else ProcessingConfig(),
            output=OutputConfig(**output_raw) if output_raw else OutputConfig(),
            verification=VerificationConfig(**verification_raw) if verification_raw else VerificationConfig(),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        )
