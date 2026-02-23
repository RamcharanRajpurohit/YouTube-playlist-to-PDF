#!/usr/bin/env python3
"""
Playlist-to-Book Pipeline
=========================
End-to-end CLI that converts a YouTube playlist (or single video) into a
professionally formatted book manuscript (Markdown + PDF).

Usage:
    python main.py --url <playlist_or_video_url> [--config config/default.yaml]
                   [--verify] [--dry-run] [--title "Book Title"]
"""

from __future__ import annotations

import argparse
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
from src.transcript.fetcher import PlaylistExtractor, TranscriptFetcher, VideoInfo

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
        description="Convert a YouTube playlist into a book manuscript.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url",
        required=True,
        help="YouTube playlist or video URL.",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config file (default: config/default.yaml).",
    )
    parser.add_argument(
        "--title",
        default="Building LLMs from Scratch",
        help="Book title (default: 'Building LLMs from Scratch').",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable content verification using a secondary LLM.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only fetch transcripts and propose TOC — no chapter writing.",
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
    "2": "groq",
    "3": "ollama",
}

_PROVIDER_LABELS = {
    "gemini": "Gemini (Google AI)",
    "groq":   "Grok / Groq (fast cloud inference)",
    "ollama": "Local model (Ollama, no API key needed)",
}

_KEY_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",
    "groq":   "GROQ_API_KEY",
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

    print("\n" + "═" * 50)
    print("  🤖  Choose your LLM provider")
    print("═" * 50)
    for num, name in _PROVIDERS.items():
        print(f"  [{num}] {_PROVIDER_LABELS[name]}")
    print()

    while True:
        choice = input("Enter choice [1/2/3]: ").strip()
        if choice in _PROVIDERS:
            break
        print("  ⚠  Please enter 1, 2, or 3.")

    provider = _PROVIDERS[choice]
    print(f"\n  ✔  Selected: {_PROVIDER_LABELS[provider]}")

    # For API-key providers, check and prompt if missing
    env_var = _KEY_ENV_VARS.get(provider)
    if env_var:
        existing_key = os.getenv(env_var, "").strip()
        if existing_key:
            print(f"  ✔  {env_var} found in environment — using it.\n")
            # Ensure config carries the key
            if provider == "gemini":
                config.gemini_api_key = existing_key
            elif provider == "groq":
                config.groq_api_key = existing_key
        else:
            print(f"\n  {env_var} not found.")
            key = input(f"  Enter your {_PROVIDER_LABELS[provider]} API key: ").strip()
            while not key:
                key = input("  API key cannot be empty. Try again: ").strip()

            # Save to .env for future runs
            _save_to_dotenv(env_var, key)
            os.environ[env_var] = key
            print(f"  ✔  Saved to .env for future runs.\n")

            if provider == "gemini":
                config.gemini_api_key = key
            elif provider == "groq":
                config.groq_api_key = key
    else:
        # Ollama — no key needed
        print("  ℹ  No API key required. Ollama will start automatically if needed.\n")

    config.primary_provider = provider
    print("═" * 50 + "\n")


# ── Pipeline steps ───────────────────────────────────────────────────


def step_generate_toc(
    videos: List[VideoInfo],
    config: Config,
    book_title: str,
) -> List[ChapterOutline]:
    """Step 2: Generate table of contents from video metadata via LLM."""
    logger.info("━━━ Step 2/5: Generating book structure ━━━")
    llm = LLMFactory.create(config.primary_provider, config)
    chunker = TextChunker(config)
    structurer = BookStructurer(llm, chunker, config)
    outlines = structurer.generate_toc(videos, book_title)
    logger.info("Proposed %d chapters:", len(outlines))
    for o in outlines:
        logger.info("  Ch. %d: %s (videos: %s)", o.number, o.title, o.video_indices)
    return outlines


def step_write_chapters_lazy(
    outlines: List[ChapterOutline],
    video_map: dict,
    fetcher: TranscriptFetcher,
    cleaner: TranscriptCleaner,
    config: Config,
) -> List[Chapter]:
    """Step 3: Lazily fetch, clean, and write one chapter at a time.

    Transcripts are fetched only for the videos belonging to the current
    chapter, immediately before writing it. This spaces out YouTube API
    calls and avoids rate limits.
    """
    logger.info("━━━ Step 3/5: Writing chapters (lazy fetch) ━━━")
    llm = LLMFactory.create(config.primary_provider, config)
    chunker = TextChunker(config)
    structurer = BookStructurer(llm, chunker, config)
    chapters: List[Chapter] = []

    for outline in tqdm(outlines, desc="Chapters"):
        logger.info(
            "━━━ Chapter %d/%d: '%s' ━━━",
            outline.number, len(outlines), outline.title,
        )

        # Fetch only the transcripts this chapter needs
        videos_needed = [video_map[i] for i in outline.video_indices if i in video_map]
        if not videos_needed:
            logger.warning("No videos found for chapter %d, skipping.", outline.number)
            continue

        logger.info(
            "  Fetching %d transcript(s): %s",
            len(videos_needed),
            [v.title for v in videos_needed],
        )
        transcripts = []
        for v in videos_needed:
            try:
                transcripts.append(fetcher.fetch(v))
                if getattr(config.transcript, "delay_seconds", 0) > 0:
                    time.sleep(config.transcript.delay_seconds)
            except Exception as exc:
                logger.error("  Could not fetch transcript for '%s': %s", v.title, exc)

        if not transcripts:
            logger.warning("No transcripts fetched for chapter %d, skipping.", outline.number)
            continue

        # Clean the fetched transcripts
        cleaned: dict = {t.video.index: cleaner.clean(t.full_text) for t in transcripts}

        # Write the chapter
        logger.info("  -> Passing %d cleaned segments to the LLM (generation may take a moment)...", len(cleaned))
        chapter = structurer.write_single_chapter(outline, transcripts, cleaned)
        if chapter is not None:
            chapters.append(chapter)

    return chapters

def save_playlist_metadata(
    videos: List[VideoInfo],
    outlines: List[ChapterOutline],
    book_title: str,
    config: Config,
) -> None:
    """Persist playlist metadata and TOC to output/playlist_metadata/ as JSON."""
    import json
    from dataclasses import asdict
    from pathlib import Path

    meta_dir = Path(config.output.chapters_dir).parent / "playlist_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # --- videos_metadata.json ---
    videos_data = [
        {
            "index": v.index,
            "video_id": v.video_id,
            "title": v.title,
            "duration_seconds": v.duration,
            "duration_minutes": round(v.duration / 60, 1),
            "chapters": [
                {"title": ch.title, "start_time": ch.start_time}
                for ch in v.chapters
            ],
        }
        for v in videos
    ]
    videos_path = meta_dir / "videos_metadata.json"
    with open(videos_path, "w") as fh:
        json.dump({"book_title": book_title, "total_videos": len(videos), "videos": videos_data}, fh, indent=2)
    logger.info("Saved videos metadata → %s", videos_path)

    # --- toc.json ---
    if outlines:
        toc_data = [
            {
                "number": o.number,
                "title": o.title,
                "video_indices": o.video_indices,
                "description": o.description,
            }
            for o in outlines
        ]
        toc_path = meta_dir / "toc.json"
        with open(toc_path, "w") as fh:
            json.dump({"book_title": book_title, "chapters": toc_data}, fh, indent=2)
        logger.info("Saved TOC → %s", toc_path)


def step_export(chapters: List[Chapter], config: Config, book_title: str):
    """Step 5: Export to Markdown and PDF."""
    logger.info("━━━ Step 5/5: Exporting manuscript ━━━")
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

    # Step 1: Extract videos (metadata + chapters)
    logger.info("━━━ Step 1/5: Extracting video list ━━━")
    extractor = PlaylistExtractor(config)
    videos = extractor.extract(args.url)
    logger.info("Found %d videos.", len(videos))
    for v in videos:
        logger.info("  [%d] %s (%.0f min, %d chapters)",
                    v.index, v.title, v.duration / 60, len(v.chapters))

    if not videos:
        logger.error("No videos found. Exiting.")
        sys.exit(1)

    # Save video list + chapters to playlist_metadata/
    save_playlist_metadata(videos, [], args.title, config)

    # Build index → VideoInfo map for the lazy fetch loop
    video_map = {v.index: v for v in videos}

    # Step 2: Generate TOC from metadata (no transcripts needed)
    outlines = step_generate_toc(videos, config, args.title)

    # Persist TOC alongside video metadata
    save_playlist_metadata(videos, outlines, args.title, config)

    if args.dry_run:
        logger.info("━━━ Dry-run complete — exiting before chapter generation ━━━")
        return

    # Step 3: Write chapters (lazy fetch — one chapter at a time)
    fetcher = TranscriptFetcher(config)
    cleaner = TranscriptCleaner(config)
    chapters = step_write_chapters_lazy(outlines, video_map, fetcher, cleaner, config)

    # Step 4: Export (Skipping Refinement to save tokens)
    step_export(chapters, config, args.title)


if __name__ == "__main__":
    main()
