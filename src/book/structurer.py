"""Book structuring — table of contents generation, chapter writing, refinement."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from src.config import Config
from src.llm.base import LLMProvider
from src.processing.chunker import TextChunk, TextChunker
from src.transcript.fetcher import VideoInfo, VideoTranscript

logger = logging.getLogger(__name__)


# ── Data models ──────────────────────────────────────────────────────

@dataclass
class ChapterOutline:
    """One entry in the proposed table of contents."""

    number: int
    title: str
    video_indices: List[int]  # which videos contribute to this chapter
    description: str = ""


@dataclass
class Chapter:
    """A fully written chapter."""

    number: int
    title: str
    content: str  # full Markdown text of the chapter
    source_video_indices: List[int] = field(default_factory=list)


# ── Prompts ──────────────────────────────────────────────────────────

_TOC_SYSTEM = (
    "You are a senior technical book editor. Given the list of video titles, "
    "durations, and short summaries from a YouTube playlist, produce a logical "
    "book outline. Merge related CONSECUTIVE videos into coherent chapters. "
    "CRITICAL: video_indices MUST be contiguous integers (e.g., [0, 1, 2], not [0, 2, 5]). "
    "Do not skip any videos and do not jump out of order. "
    "Output ONLY valid JSON — an array of objects with keys: number, title, video_indices "
    "(0-based list), description."
)

_TOC_USER = (
    "Here are the videos in the playlist:\n\n{video_list}\n\n"
    "Create a table of contents for a professional technical book titled "
    '"{book_title}". Merge closely related CONSECUTIVE videos into single chapters. '
    "Each chapter should cover a coherent topic. The videos MUST remain in their original "
    "chronological order. Output ONLY the JSON array."
)

_CHAPTER_SYSTEM = (
    "You are a professional technical author. Your task is to REWRITE the "
    "provided transcript into formal, book-quality prose. CRITICAL RULES:\n\n"
    "CONTENT PRESERVATION:\n"
    "- Include ALL information from the transcript — do not summarize the content "
    "itself. Preserve technical information, formulas, and code.\n\n"
    "TONE CONVERSION:\n"
    "- Remove ALL conversational filler words.\n"
    "- Convert spoken lecture style into formal or instructional tone.\n\n"
    "FORMATTING & CONTINUITY:\n"
    "- Use clear Markdown headings (##, ###) accurately.\n"
    "- Format code blocks with language tags.\n"
    "- Ensure smooth, logical flow between paragraphs. Since this is chunked, "
    "connect your writing gracefully to the summary of the previous chunk if provided.\n\n"
    "SUMMARY EXTRACTION (CRITICAL):\n"
    "At the VERY END of your markdown output, you MUST provide a brief summary "
    "of what you just wrote wrapped in <summary> tags. This will be fed into "
    "the generation of the next chunk to maintain narrative context.\n"
    "Example:\n"
    "Your chapter prose here...\n"
    "<summary>In this chunk, we covered the basics of tokenization and Byte-Pair Encoding.</summary>"
)

_CHAPTER_USER = (
    "## Chapter {chapter_number}: {chapter_title}\n\n"
    "### Chapter Description\n{chapter_description}\n\n"
    "### Transcript Content (chunk {chunk_index}/{total_chunks})\n\n"
    "{transcript_chunk}\n\n"
    "---\n"
    "REWRITE the entire transcript content above into formal book prose. "
    "Include ALL information — do not summarize or condense. "
    "This is chunk {chunk_index} of {total_chunks} for this chapter."
    "{continuation_note}"
)

_MERGE_SYSTEM = (
    "You are a professional book editor merging already-written parts into a "
    "single, cohesive markdown chapter.\n\n"
    "RULES:\n"
    "- Do NOT alter the technical content. Keep ALL information.\n"
    "- Ensure transitions between parts are seamless.\n"
    "- Remove duplicate overlaps at boundaries if any exist.\n"
    "- The chapter should start with a single # heading for the chapter title.\n"
    "- Provide ONLY the final, merged markdown chapter. No trailing tags."
)

_MERGE_USER = (
    "## Chapter {chapter_number}: {chapter_title}\n\n"
    "Below are {num_parts} parts that must be merged into one cohesive chapter.\n\n{parts}"
)


# ── BookStructurer ───────────────────────────────────────────────────

class BookStructurer:
    """Orchestrate the multi-phase book generation process.

    Phase 1: Generate Table of Contents from video metadata.
    Phase 2: Write chunks, chaining summaries forward for logical continuity.
    Phase 3: Merge chunks seamlessly into one chapter (no separate refine pass).
    """

    def __init__(
        self,
        llm: LLMProvider,
        chunker: TextChunker,
        config: Config,
    ) -> None:
        self._llm = llm
        self._chunker = chunker
        self._config = config
        self._chapters_dir = Path(config.output.chapters_dir)
        self._chapters_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Table of Contents ───────────────────────────────

    def generate_toc(
        self,
        videos: List[VideoInfo],
        book_title: str = "Building LLMs from Scratch",
    ) -> List[ChapterOutline]:
        """Ask the LLM to create a logical chapter structure from video metadata.

        Uses YouTube chapter markers (keyframes) when available. No transcript
        text is required — this runs purely from VideoInfo metadata.
        """
        video_list_parts = []
        for v in videos:
            if v.chapters:
                chapter_lines = []
                for ch in v.chapters:
                    mins = int(ch.start_time // 60)
                    secs = int(ch.start_time % 60)
                    chapter_lines.append(f"     [{mins}:{secs:02d}] {ch.title}")
                keyframes = "\n".join(chapter_lines)
                video_list_parts.append(
                    f"{v.index}. \"{v.title}\" "
                    f"(duration: {v.duration / 60:.0f} min)\n"
                    f"   Chapters:\n{keyframes}"
                )
            else:
                video_list_parts.append(
                    f"{v.index}. \"{v.title}\" "
                    f"(duration: {v.duration / 60:.0f} min)\n"
                    f"   (no chapter markers available)"
                )

        prompt = _TOC_USER.format(
            video_list="\n".join(video_list_parts),
            book_title=book_title,
        )

        raw = self._llm.generate(prompt, system_prompt=_TOC_SYSTEM)
        return self._parse_toc(raw)

    def _parse_toc(self, raw: str) -> List[ChapterOutline]:
        """Parse JSON TOC from LLM response."""
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse TOC JSON: %s\nRaw: %s", exc, raw[:500])
            raise RuntimeError("LLM did not produce valid JSON for TOC.") from exc

        outlines: List[ChapterOutline] = []
        for item in data:
            outlines.append(
                ChapterOutline(
                    number=item["number"],
                    title=item["title"],
                    video_indices=item.get("video_indices", []),
                    description=item.get("description", ""),
                )
            )
        return outlines

    # ── Phase 2: Chapter Writing ─────────────────────────────────

    def write_single_chapter(
        self,
        outline: ChapterOutline,
        transcripts: List[VideoTranscript],
        cleaned_texts: Dict[int, str],
    ) -> Optional[Chapter]:
        """Write one chapter from its pre-fetched, pre-cleaned transcripts.

        Returns None if no transcript text is available for this chapter.
        Uses the chapter cache to skip already-written chapters.
        """
        cached = self._load_cached_chapter(outline.number)
        if cached is not None:
            logger.info("Using cached chapter %d: %s", outline.number, outline.title)
            return cached

        logger.info(
            "Writing chapter %d: %s (from videos %s)",
            outline.number,
            outline.title,
            outline.video_indices,
        )

        # Combine cleaned transcripts, preserving all content with boundary markers
        combined_text = ""
        for vid_position, vi in enumerate(outline.video_indices):
            if vi in cleaned_texts:
                if vid_position > 0:
                    combined_text += (
                        "\n\n[CONTINUATION: The following content continues "
                        "from the next video in the series. Ensure a smooth "
                        "narrative transition.]\n\n"
                    )
                combined_text += cleaned_texts[vi] + "\n\n"

        if not combined_text.strip():
            logger.warning("No transcript text for chapter %d, skipping.", outline.number)
            return None

        chunks = self._chunker.chunk(combined_text)

        if len(chunks) == 1:
            part_content, _ = self._write_single_chunk(outline, chunks[0], 1, 1, "")
            chapter_content = part_content
        else:
            parts = []
            rolling_summary = ""
            for i, chunk in enumerate(chunks):
                part_content, rolling_summary = self._write_single_chunk(
                    outline, chunk, i + 1, len(chunks), rolling_summary
                )
                parts.append(part_content)
            
            # Fast merge without expensive re-writing
            chapter_content = self._merge_parts(outline, parts)

        chapter = Chapter(
            number=outline.number,
            title=outline.title,
            content=chapter_content,
            source_video_indices=outline.video_indices,
        )
        self._cache_chapter(chapter)
        return chapter

    def write_chapters(
        self,
        outlines: List[ChapterOutline],
        transcripts: List[VideoTranscript],
        cleaned_texts: Dict[int, str],
    ) -> List[Chapter]:
        """Write all chapters (bulk mode). Kept for backwards-compatibility."""
        chapters: List[Chapter] = []
        for outline in tqdm(outlines, desc="Writing chapters"):
            chapter = self.write_single_chapter(outline, transcripts, cleaned_texts)
            if chapter is not None:
                chapters.append(chapter)
        return chapters

    def _write_single_chunk(
        self,
        outline: ChapterOutline,
        chunk: TextChunk,
        chunk_index: int,
        total_chunks: int,
        previous_summary: str,
    ) -> tuple[str, str]:
        continuation = ""
        if chunk_index > 1:
            continuation = (
                f"\n\nPRIOR CONTEXT:\nHere is a summary of what you wrote in the "
                f"previous chunk: {previous_summary}\n"
                f"Write the next chunk to follow naturally from that point."
            )

        prompt = _CHAPTER_USER.format(
            chapter_number=outline.number,
            chapter_title=outline.title,
            chapter_description=outline.description,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            transcript_chunk=chunk.text,
            continuation_note=continuation,
        )

        raw_output = self._llm.generate(prompt, system_prompt=_CHAPTER_SYSTEM)
        
        # Extract <summary> tags
        new_summary = ""
        summary_match = re.search(r"<summary>(.*?)</summary>", raw_output, re.DOTALL | re.IGNORECASE)
        if summary_match:
            new_summary = summary_match.group(1).strip()
            # Strip the <summary> block out of the generated prose
            chapter_prose = re.sub(r"<summary>.*?</summary>", "", raw_output, flags=re.DOTALL | re.IGNORECASE).strip()
        else:
            chapter_prose = raw_output.strip()
            
        return chapter_prose, new_summary

    def _merge_parts(self, outline: ChapterOutline, parts: List[str]) -> str:
        """Merge multiple chunk outputs into one coherent chapter."""
        parts_text = ""
        for i, part in enumerate(parts, 1):
            parts_text += f"\n### Part {i}\n{part}\n"

        prompt = _MERGE_USER.format(
            chapter_number=outline.number,
            chapter_title=outline.title,
            num_parts=len(parts),
            parts=parts_text,
        )

        return self._llm.generate(prompt, system_prompt=_MERGE_SYSTEM)

    # ── Chapter caching ──────────────────────────────────────────

    def _cache_chapter(self, chapter: Chapter) -> None:
        path = self._chapters_dir / f"chapter_{chapter.number:02d}.md"
        with open(path, "w") as fh:
            fh.write(chapter.content)
        logger.debug("Cached chapter %d to %s", chapter.number, path)

    def _load_cached_chapter(self, number: int) -> Optional[Chapter]:
        path = self._chapters_dir / f"chapter_{number:02d}.md"
        if not path.exists():
            return None
        with open(path, "r") as fh:
            content = fh.read()
        # We don't have the full metadata, so return a minimal Chapter
        return Chapter(number=number, title="", content=content)
