#!/usr/bin/env python3
"""
Playlist-to-Book Pipeline
=========================
End-to-end CLI that converts pre-fetched YouTube playlist transcripts into a
professionally formatted book manuscript (Markdown + PDF).

Transcript data and playlist metadata are read from the ``data/`` directory.

Usage:
    python main.py [--config config/default.yaml] [--verify] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from typing import List

from tqdm import tqdm

from src.book.exporter import MarkdownExporter, PDFExporter
from src.book.structurer import BookStructurer, Chapter, ChapterOutline
from src.config import Config
from src.llm.factory import LLMFactory
from src.processing.chunker import TextChunker
from src.processing.cleaner import TranscriptCleaner
from src.transcript.fetcher import (
    VideoInfo,
    load_all_transcripts,
    load_playlist_metadata,
    load_transcript,
)

logger = logging.getLogger("playlist_to_book")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    # Suppress HF Symlinks warning widely
    import os
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert pre-fetched YouTube playlist transcripts into a book manuscript.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config file (default: config/default.yaml).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable content verification using a secondary LLM.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load data and propose TOC — no chapter writing.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


# ── Provider setup wizard ─────────────────────────────────────────────

_PROVIDERS = {
    "1": "gemini",
}

_PROVIDER_LABELS = {
    "gemini": "Gemini (Google AI)",
}

_KEY_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",
}


def _save_to_dotenv(key: str, value: str, dotenv_path: str = ".env") -> None:
    """Persist a key=value pair in the .env file (create if absent)."""
    from pathlib import Path
    path = Path(dotenv_path)
    lines = path.read_text().splitlines() if path.exists() else []
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}=") or line.startswith(f"{key} ="):
            lines[i] = f"{key}={value}"
            updated = True
            break
    if not updated:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n")


def prompt_provider_setup(config: "Config") -> None:
    """Interactive wizard: ask the user which LLM provider to use and
    collect any missing credentials. Updates *config* in-place."""
    import os

    provider = "gemini"

    # For API-key providers, check and prompt if missing
    env_var = _KEY_ENV_VARS.get(provider)
    if env_var:
        existing_key = os.getenv(env_var, "").strip()
        if existing_key:
            print(f"  ✔  {env_var} found in environment — using it.\n")
            # Ensure config carries the key
            if provider == "gemini":
                config.gemini_api_key = existing_key
        else:
            print(f"\n  {env_var} not found.")
            key = input(f"  Enter your {_PROVIDER_LABELS[provider]} API key: ").strip()
            while not key:
                key = input("  API key cannot be empty. Try again: ").strip()

            # Save to .env for future runs
            os.environ[env_var] = key
            try:
                _save_to_dotenv(env_var, key)
                print(f"  ✔  Saved to .env for future runs.\n")
            except PermissionError:
                print(f"  ✔  Key set for this session (no write access to .env).\n")

            if provider == "gemini":
                config.gemini_api_key = key

    config.primary_provider = provider
    print("═" * 50 + "\n")


# ── Pipeline steps ───────────────────────────────────────────────────


def step_generate_toc(
    videos: List[VideoInfo],
    config: Config,
    book_title: str,
) -> List[ChapterOutline]:
    """Step 2: Generate table of contents from video metadata via LLM."""
    logger.info("━━━ Step 2/4: Generating book structure ━━━")
    llm = LLMFactory.create(config.primary_provider, config)
    chunker = TextChunker(config)
    structurer = BookStructurer(llm, chunker, config)
    outlines = structurer.generate_toc(videos, book_title)
    logger.info("Proposed %d chapters:", len(outlines))
    for o in outlines:
        logger.info("  Ch. %d: %s (videos: %s)", o.number, o.title, o.video_indices)
    return outlines


def step_write_chapters(
    outlines: List[ChapterOutline],
    video_map: dict,
    cleaner: TranscriptCleaner,
    config: Config,
) -> List[Chapter]:
    """Step 3: Load transcripts, clean, and write chapters in parallel batches."""
    logger.info("━━━ Step 3/4: Writing chapters (parallel) ━━━")

    batch_size = getattr(config.processing, "parallel_chapters", 5)
    logger.info("  Batch size: %d chapters at a time", batch_size)

    async def _write_one(
        outline: ChapterOutline,
        structurer: BookStructurer,
        loop: asyncio.AbstractEventLoop,
    ) -> Chapter | None:
        """Load transcripts and write a single chapter (runs in thread pool)."""
        videos_needed = [video_map[i] for i in outline.video_indices if i in video_map]
        if not videos_needed:
            logger.warning("No videos found for chapter %d, skipping.", outline.number)
            return None

        logger.info(
            "  [Ch %d] Loading %d transcript(s)",
            outline.number,
            len(videos_needed),
        )
        transcripts = []
        for v in videos_needed:
            try:
                transcripts.append(load_transcript(v))
            except FileNotFoundError as exc:
                logger.error("  [Ch %d] Missing transcript for '%s': %s", outline.number, v.title, exc)

        if not transcripts:
            logger.warning("No transcripts found for chapter %d, skipping.", outline.number)
            return None

        cleaned: dict = {t.video.index: cleaner.clean(t.full_text) for t in transcripts}
        logger.info(
            "  [Ch %d] -> Passing %d cleaned segments to the LLM...",
            outline.number, len(cleaned),
        )

        # LLM call is blocking — run in a thread so other coroutines can proceed
        chapter = await loop.run_in_executor(
            None,
            lambda: structurer.write_single_chapter(outline, transcripts, cleaned),
        )
        if chapter:
            logger.info("  [Ch %d] ✓ Done: %s", outline.number, outline.title)
        return chapter

    async def _run_all() -> List[Chapter]:
        llm = LLMFactory.create(config.primary_provider, config)
        chunker = TextChunker(config)
        structurer = BookStructurer(llm, chunker, config)
        loop = asyncio.get_event_loop()

        all_chapters: List[Chapter] = []
        batches = [outlines[i:i + batch_size] for i in range(0, len(outlines), batch_size)]

        for batch_idx, batch in enumerate(batches, 1):
            logger.info(
                "━━━ Batch %d/%d: Chapters %s ━━━",
                batch_idx, len(batches),
                [o.number for o in batch],
            )
            tasks = [_write_one(o, structurer, loop) for o in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for outline, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error("  [Ch %d] Failed: %s", outline.number, result)
                elif result is not None:
                    all_chapters.append(result)

        # Return chapters sorted in correct order
        all_chapters.sort(key=lambda c: c.number)
        return all_chapters

    return asyncio.run(_run_all())


def step_export(chapters: List[Chapter], config: Config, book_title: str):
    """Step 4: Export to Markdown and PDF."""
    logger.info("━━━ Step 4/4: Exporting manuscript ━━━")
    md_exporter = MarkdownExporter(config)
    manuscript = md_exporter.export(chapters, book_title)

    pdf_exporter = PDFExporter(config)
    pdf_exporter.export(manuscript, book_title)

    logger.info("✅ Done! Outputs:")
    logger.info("   Markdown: %s", config.output.manuscript_md)
    logger.info("   PDF:      %s", config.output.manuscript_pdf)


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║   Playlist → Book Pipeline               ║")
    logger.info("╚══════════════════════════════════════════╝")

    config = Config.load(args.config)

    # Interactive provider selection (before any pipeline work)
    prompt_provider_setup(config)

    # Override verification setting from CLI
    if args.verify:
        config.verification.enabled = True

    # Step 1: Load video metadata from data/
    logger.info("━━━ Step 1/4: Loading playlist metadata ━━━")
    book_title, videos = load_playlist_metadata()
    logger.info("Loaded %d videos. Book title: '%s'", len(videos), book_title)

    if not videos:
        logger.error("No videos found in metadata. Exiting.")
        sys.exit(1)

    # Build index → VideoInfo map
    video_map = {v.index: v for v in videos}

    # Step 2: Generate TOC from metadata (no transcripts needed)
    outlines = step_generate_toc(videos, config, book_title)

    if args.dry_run:
        logger.info("━━━ Dry-run complete — exiting before chapter generation ━━━")
        return

    # Step 3: Write chapters (load transcripts from data/, clean, write)
    cleaner = TranscriptCleaner(config)
    chapters = step_write_chapters(outlines, video_map, cleaner, config)

    # Step 4: Export
    step_export(chapters, config, book_title)


if __name__ == "__main__":
    main()
