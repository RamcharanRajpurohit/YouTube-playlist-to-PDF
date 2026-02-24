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
from src.transcript.fetcher import TranscriptSegment, VideoChapter, VideoInfo, VideoTranscript

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
    "durations, and chapter markers from a YouTube playlist, produce a logical "
    "book outline.\n\n"
    "GROUPING PHILOSOPHY:\n"
    "- Each chapter should represent a MEANINGFUL unit of a book — a reader should "
    "finish a chapter feeling they learned one coherent topic.\n"
    "- Group CONSECUTIVE videos that cover the same broad topic into ONE chapter. "
    "Videos that are continuations or parts of the same subject belong together.\n"
    "- Chapters should be BALANCED in size"
    "- Think about what a READER expects from a chapter title — 'Attention Mechanisms' "
    "is a good chapter that covers simplified, self, causal, and multi-head attention. "
    "Having separate chapters for each type of attention is too granular for a book.\n\n"
    "CONSTRAINTS:\n"
    "- video_indices MUST be contiguous integers (e.g., [0, 1, 2], not [0, 2, 5]).\n"
    "- Do not skip any videos and do not jump out of order.\n"
    "- Every video must appear in exactly one chapter.\n\n"
    "Output ONLY valid JSON — an array of objects with keys: number, title, video_indices "
    "(0-based list), description."
    "keep it between 1/3 to 1/2 the total number of videos (e.g., for 20 videos, aim for 6-10 chapters)."
    "never combine more than 3 videos into a single chapter"
)

_TOC_USER = (
    "Here are the {total_videos} videos in the playlist (total runtime: {total_runtime}):\n\n"
    "{video_list}\n\n"
    "Create a table of contents for a professional technical book titled "
    '"{book_title}". Group consecutive videos into chapters by broad topic. '
    "Short related videos should be merged together; a chapter break should happen "
    "when the subject clearly changes. "
    "The videos MUST remain in their original chronological order. "
    "Output ONLY the JSON array."
)

_CHAPTER_SYSTEM = (
    "You are a professional technical book author. Your task is to transform the "
    "provided transcript into formal, book-quality written prose.\n\n"

    "PRIMARY OBJECTIVE:\n"
    "- Use the transcript as the SOLE source of truth.\n"
    "- Preserve ALL technical meaning, explanations, examples, reasoning steps, and code.\n"
    "- Maintain the SAME logical order of ideas as the transcript.\n"
    "- Do NOT summarize, compress, or omit technical explanations.\n\n"

    "CRITICAL PROHIBITIONS (VERY IMPORTANT):\n"
    "- NEVER write lecture-style meta phrases such as:\n"
    "  'In this lecture...', 'In this video...', 'In this chapter we will see...',\n"
    "  'Previously we discussed...', 'Now let's talk about...', 'Welcome...', etc.\n"
    "- NEVER refer to the transcript, speaker, lecture, video, or audience.\n"
    "- NEVER repeat structural filler phrases.\n"
    "- NEVER add introductions or conclusions that were not explicitly explained in the transcript.\n"
    "- NEVER invent new information, examples, or explanations.\n\n"

    "ALLOWED TRANSFORMATIONS:\n"
    "- Remove filler words: um, uh, you know, like, so, basically, okay, right, etc.\n"
    "- Remove conversational padding and repetition.\n"
    "- Rewrite sentences into formal technical prose.\n"
    "- Combine fragmented spoken sentences into clear written sentences.\n"
    "- Improve clarity while preserving original meaning.\n\n"

    "BOOK-STYLE WRITING REQUIREMENTS:\n"
    "- Write as if this is part of a professional technical textbook.\n"
    "- Present information directly and confidently.\n"
    "- Focus ONLY on explaining the subject matter.\n"
    "- Do NOT mention what will be covered — simply explain it.\n"
    "- Do NOT use teaching narration — use declarative explanatory prose.\n"
    "- Avoid conversational tone completely.\n\n"

    "CONTENT COMPLETENESS (MANDATORY):\n"
    "- Preserve EVERY concept, example, explanation, and step.\n"
    "- Preserve ALL code, formulas, and technical details.\n"
    "- If an example is explained step-by-step, keep ALL steps.\n"
    "- DO NOT shorten explanations.\n"
    "- Output length should be approximately equal to input length.\n\n"

    "STRUCTURE AND FORMAT:\n"
    "- Use proper Markdown headings where appropriate.\n"
    "- Use proper paragraphs with logical flow.\n"
    "- Use code blocks with correct formatting if present.\n"
    "- For lists, ALWAYS place each item on its own line using `- ` (dash-space) "
    "syntax. NEVER put multiple list items on a single line separated by `*`. "
    "Always leave a blank line before the first list item.\n"
    "- Ensure smooth, continuous textbook-style flow.\n\n"

    "STRICT BOOK-STYLE RULE:\n"
    "- The output must read like a textbook chapter, NOT like a lecture transcript.\n"
    "- It must NOT sound like someone speaking. It must sound like formal written knowledge.\n"
    "- Write ONLY in English. Do NOT include any non-English text, foreign scripts, "
    "or transliterations. If the transcript contains non-English words, translate them "
    "to English or omit them.\n\n"

    "SUMMARY REQUIREMENT:\n"
    "At the VERY END, include a concise factual summary wrapped in <summary> tags.\n"
    "Example:\n"
    "<summary>This section explained gradient descent, its update rule, and a worked example.</summary>"
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



# ── BookStructurer ───────────────────────────────────────────────────

class BookStructurer:
    """Orchestrate the multi-phase book generation process.

    Phase 1: Generate Table of Contents from video metadata.
    Phase 2: Write chunks, chaining summaries forward for logical continuity.
    Phase 3: Concatenate chunk outputs directly (no LLM merge pass needed).
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

        total_seconds = sum(v.duration for v in videos)
        total_hours = int(total_seconds // 3600)
        total_mins = int((total_seconds % 3600) // 60)
        total_runtime = f"{total_hours}h {total_mins}m"

        prompt = _TOC_USER.format(
            total_videos=len(videos),
            total_runtime=total_runtime,
            video_list="\n".join(video_list_parts),
            book_title=book_title,
        )

        try:
            raw, toc_in_tokens, toc_out_tokens = self._llm.generate(prompt, system_prompt=_TOC_SYSTEM)
            logger.info("[TOC Generation] Input tokens: %d, Output tokens: %d", toc_in_tokens, toc_out_tokens)
            return self._parse_toc(raw)
        except Exception as exc:
            logger.error(
                "TOC generation failed: %s. Falling back to video-per-chapter TOC.",
                exc,
            )
            return self._generate_fallback_toc(videos)

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

    def _generate_fallback_toc(
        self, videos: List[VideoInfo]
    ) -> List[ChapterOutline]:
        """Fallback TOC: each video becomes its own chapter using video titles."""
        logger.warning(
            "Using fallback TOC: each video becomes its own chapter (%d chapters).",
            len(videos),
        )
        outlines: List[ChapterOutline] = []
        for v in videos:
            if v.chapters:
                description = "Sections: " + ", ".join(ch.title for ch in v.chapters)
            else:
                description = ""
            outlines.append(
                ChapterOutline(
                    number=v.index + 1,
                    title=v.title,
                    video_indices=[v.index],
                    description=description,
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

        try:
            chunks = self._chunker.chunk(combined_text)

            if len(chunks) == 1:
                part_content, _, in_tok, out_tok = self._write_single_chunk(outline, chunks[0], 1, 1, "")
                chapter_content = part_content
                total_in_tokens = in_tok
                total_out_tokens = out_tok
            else:
                parts = []
                rolling_summary = ""
                total_in_tokens = 0
                total_out_tokens = 0
                for i, chunk in enumerate(chunks):
                    part_content, rolling_summary, in_tok, out_tok = self._write_single_chunk(
                        outline, chunk, i + 1, len(chunks), rolling_summary
                    )
                    parts.append(part_content)
                    total_in_tokens += in_tok
                    total_out_tokens += out_tok

                # Direct concatenation — rolling summaries already ensure continuity
                chapter_content = "\n\n".join(parts)

            logger.info(
                "Chapter %d token usage - Input: %d, Output: %d",
                outline.number, total_in_tokens, total_out_tokens
            )

            chapter = Chapter(
                number=outline.number,
                title=outline.title,
                content=chapter_content,
                source_video_indices=outline.video_indices,
            )
            self._cache_chapter(chapter)
            return chapter

        except Exception as exc:
            logger.error(
                "LLM chapter writing failed for chapter %d (%s): %s. "
                "Falling back to raw transcript with keyframes.",
                outline.number, outline.title, exc,
            )
            return self._generate_fallback_chapter(outline, transcripts, cleaned_texts)

    def _generate_fallback_chapter(
        self,
        outline: ChapterOutline,
        transcripts: List[VideoTranscript],
        cleaned_texts: Dict[int, str],
    ) -> Optional[Chapter]:
        """Fallback chapter from raw transcript text organised by keyframe markers."""
        logger.warning(
            "Generating fallback chapter %d: %s (raw transcript + keyframes)",
            outline.number, outline.title,
        )

        parts: List[str] = []
        multi_video = len(outline.video_indices) > 1

        for transcript in transcripts:
            video = transcript.video
            if video.index not in cleaned_texts:
                continue
            cleaned = cleaned_texts[video.index]
            if not cleaned.strip():
                continue

            if multi_video:
                parts.append(f"\n## {video.title}\n")

            if video.chapters:
                sections = self._split_by_keyframes(transcript.segments, video.chapters)
                for section_title, section_text in sections:
                    heading = "##" if not multi_video else "###"
                    parts.append(f"\n{heading} {section_title}\n")
                    parts.append(section_text.strip() + "\n")
            else:
                parts.append(cleaned.strip() + "\n")

        content = "\n".join(parts)
        if not content.strip():
            return None

        chapter = Chapter(
            number=outline.number,
            title=outline.title,
            content=content,
            source_video_indices=outline.video_indices,
        )
        self._cache_chapter(chapter)
        return chapter

    @staticmethod
    def _split_by_keyframes(
        segments: List[TranscriptSegment],
        chapters: List[VideoChapter],
    ) -> List[tuple]:
        """Split transcript segments into sections based on keyframe timestamps.

        Returns list of (section_title, section_text) tuples.
        """
        if not chapters or not segments:
            return [("Content", " ".join(s.text for s in segments))]

        sorted_chapters = sorted(chapters, key=lambda c: c.start_time)
        result: List[tuple] = []

        for i, ch in enumerate(sorted_chapters):
            start = ch.start_time
            end = sorted_chapters[i + 1].start_time if i + 1 < len(sorted_chapters) else float("inf")
            seg_texts = [s.text for s in segments if start <= s.start < end]
            joined = " ".join(seg_texts).strip()
            if joined:
                result.append((ch.title, joined))

        return result if result else [("Content", " ".join(s.text for s in segments))]

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
    ) -> tuple[str, str, int, int]:
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

        raw_output, in_tokens, out_tokens = self._llm.generate(prompt, system_prompt=_CHAPTER_SYSTEM)

        # Extract <summary> tags
        new_summary = ""
        summary_match = re.search(r"<summary>(.*?)</summary>", raw_output, re.DOTALL | re.IGNORECASE)
        if summary_match:
            new_summary = summary_match.group(1).strip()
            # Strip the <summary> block out of the generated prose
            chapter_prose = re.sub(r"<summary>.*?</summary>", "", raw_output, flags=re.DOTALL | re.IGNORECASE).strip()
        else:
            chapter_prose = raw_output.strip()

        chapter_prose = self._truncate_repetition(chapter_prose)

        return chapter_prose, new_summary, in_tokens, out_tokens

    @staticmethod
    def _truncate_repetition(text: str, window: int = 60, threshold: int = 4) -> str:
        """Detect and truncate degenerate repetition loops in LLM output.

        Scans for any sequence of `window` characters that repeats `threshold`
        or more times consecutively, and truncates at the first repetition.
        """
        if len(text) < window * threshold:
            return text

        # Check for repeated phrases by looking at sliding windows
        for w in (window, 30, 15):
            i = 0
            while i < len(text) - w * threshold:
                pattern = text[i:i + w]
                # Count consecutive repetitions of this pattern
                count = 1
                pos = i + w
                while pos + w <= len(text) and text[pos:pos + w] == pattern:
                    count += 1
                    pos += w
                if count >= threshold:
                    logger.warning(
                        "Detected degenerate repetition (%d repeats of %d-char pattern) — truncating",
                        count, w,
                    )
                    return text[:i].rstrip()
                i += 1
        return text

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
