"""Unit tests for BookStructurer fallback logic (TOC + chapter writing)."""

import pytest
from unittest.mock import MagicMock, patch

from src.book.structurer import BookStructurer, Chapter, ChapterOutline
from src.config import ChunkingConfig, Config, OutputConfig
from src.llm.base import LLMProvider
from src.processing.chunker import TextChunker
from src.transcript.fetcher import TranscriptSegment, VideoChapter, VideoInfo, VideoTranscript


# ── Helpers ──────────────────────────────────────────────────────────


def _make_video(index: int, title: str, chapters=None, duration: float = 600.0) -> VideoInfo:
    return VideoInfo(
        video_id=f"vid_{index}",
        title=title,
        duration=duration,
        index=index,
        chapters=chapters or [],
    )


def _make_transcript(video: VideoInfo, segments=None, full_text: str = "") -> VideoTranscript:
    return VideoTranscript(
        video=video,
        segments=segments or [],
        full_text=full_text,
    )


def _make_segments(timestamps_and_texts: list) -> list:
    """Build TranscriptSegment list from [(start, text), ...]."""
    return [
        TranscriptSegment(text=text, start=start, duration=5.0)
        for start, text in timestamps_and_texts
    ]


class FakeLLM(LLMProvider):
    """LLM provider that either returns a fixed response or raises."""

    def __init__(self, response: str = "", should_fail: bool = False):
        self._response = response
        self._should_fail = should_fail

    def _call(self, prompt: str, system_prompt: str = "") -> str:
        if self._should_fail:
            raise RuntimeError("Simulated LLM failure")
        return self._response

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    @property
    def max_context_tokens(self) -> int:
        return 100000

    @property
    def name(self) -> str:
        return "FakeLLM"


def _make_structurer(llm: LLMProvider, tmp_path) -> BookStructurer:
    config = Config()
    config.chunking = ChunkingConfig(target_chunk_tokens=12000, overlap_sentences=2)
    config.output = OutputConfig(chapters_dir=str(tmp_path / "chapters"))
    chunker = TextChunker(config)
    return BookStructurer(llm, chunker, config)


# ── TOC Fallback Tests ───────────────────────────────────────────────


class TestTocFallback:
    def test_fallback_toc_on_llm_failure(self, tmp_path):
        """When LLM fails, each video becomes its own chapter."""
        llm = FakeLLM(should_fail=True)
        structurer = _make_structurer(llm, tmp_path)

        videos = [
            _make_video(0, "Intro to LLMs"),
            _make_video(1, "Transformer Basics"),
            _make_video(2, "Tokenization"),
        ]

        outlines = structurer.generate_toc(videos)

        assert len(outlines) == 3
        assert outlines[0].number == 1
        assert outlines[0].title == "Intro to LLMs"
        assert outlines[0].video_indices == [0]
        assert outlines[1].number == 2
        assert outlines[1].title == "Transformer Basics"
        assert outlines[2].number == 3
        assert outlines[2].title == "Tokenization"

    def test_fallback_toc_includes_keyframe_descriptions(self, tmp_path):
        """Fallback TOC description includes keyframe titles."""
        llm = FakeLLM(should_fail=True)
        structurer = _make_structurer(llm, tmp_path)

        chapters = [
            VideoChapter(title="Introduction", start_time=0.0),
            VideoChapter(title="Architecture", start_time=120.0),
        ]
        videos = [_make_video(0, "Transformers", chapters=chapters)]

        outlines = structurer.generate_toc(videos)

        assert len(outlines) == 1
        assert "Introduction" in outlines[0].description
        assert "Architecture" in outlines[0].description

    def test_fallback_toc_no_keyframes(self, tmp_path):
        """Videos without keyframes get empty description."""
        llm = FakeLLM(should_fail=True)
        structurer = _make_structurer(llm, tmp_path)

        videos = [_make_video(0, "No Chapters Video")]
        outlines = structurer.generate_toc(videos)

        assert outlines[0].description == ""

    def test_normal_toc_on_success(self, tmp_path):
        """When LLM succeeds with valid JSON, normal TOC is returned."""
        toc_json = (
            '[{"number": 1, "title": "Getting Started", '
            '"video_indices": [0, 1], "description": "Intro stuff"}]'
        )
        llm = FakeLLM(response=toc_json)
        structurer = _make_structurer(llm, tmp_path)

        videos = [
            _make_video(0, "Video A"),
            _make_video(1, "Video B"),
        ]
        outlines = structurer.generate_toc(videos)

        assert len(outlines) == 1
        assert outlines[0].title == "Getting Started"
        assert outlines[0].video_indices == [0, 1]

    def test_fallback_toc_on_invalid_json(self, tmp_path):
        """When LLM returns garbage (not JSON), fallback TOC is used."""
        llm = FakeLLM(response="This is not valid JSON at all!!!")
        structurer = _make_structurer(llm, tmp_path)

        videos = [_make_video(0, "Only Video")]
        outlines = structurer.generate_toc(videos)

        assert len(outlines) == 1
        assert outlines[0].title == "Only Video"


# ── Chapter Writing Fallback Tests ───────────────────────────────────


class TestChapterWritingFallback:
    def test_fallback_chapter_on_llm_failure(self, tmp_path):
        """When LLM fails, chapter is built from raw transcript + keyframes."""
        llm = FakeLLM(should_fail=True)
        structurer = _make_structurer(llm, tmp_path)

        video = _make_video(
            0, "Attention Mechanisms",
            chapters=[
                VideoChapter(title="Self Attention", start_time=0.0),
                VideoChapter(title="Multi-Head", start_time=60.0),
            ],
        )
        segments = _make_segments([
            (0.0, "Self attention allows a model to look at other positions."),
            (10.0, "Each position gets a query key and value."),
            (60.0, "Multi-head attention runs multiple attention heads in parallel."),
            (70.0, "The outputs are concatenated and projected."),
        ])
        transcript = _make_transcript(video, segments=segments, full_text="full text here")

        outline = ChapterOutline(
            number=1, title="Attention Mechanisms",
            video_indices=[0], description="About attention",
        )
        cleaned = {0: "Self attention allows a model to look at other positions."}

        chapter = structurer.write_single_chapter(outline, [transcript], cleaned)

        assert chapter is not None
        assert chapter.number == 1
        assert chapter.title == "Attention Mechanisms"
        assert "Self Attention" in chapter.content
        assert "Multi-Head" in chapter.content

    def test_fallback_chapter_no_keyframes(self, tmp_path):
        """Fallback without keyframes uses cleaned text directly."""
        llm = FakeLLM(should_fail=True)
        structurer = _make_structurer(llm, tmp_path)

        video = _make_video(0, "Simple Video")
        transcript = _make_transcript(video, full_text="some text")

        outline = ChapterOutline(
            number=1, title="Simple Video",
            video_indices=[0], description="",
        )
        cleaned = {0: "This is the cleaned transcript content for the chapter."}

        chapter = structurer.write_single_chapter(outline, [transcript], cleaned)

        assert chapter is not None
        assert "cleaned transcript content" in chapter.content

    def test_fallback_chapter_multi_video(self, tmp_path):
        """Fallback with multiple videos adds sub-headings per video."""
        llm = FakeLLM(should_fail=True)
        structurer = _make_structurer(llm, tmp_path)

        vid0 = _make_video(0, "Part 1: Basics")
        vid1 = _make_video(1, "Part 2: Advanced")
        t0 = _make_transcript(vid0, full_text="basics text")
        t1 = _make_transcript(vid1, full_text="advanced text")

        outline = ChapterOutline(
            number=1, title="Full Topic",
            video_indices=[0, 1], description="",
        )
        cleaned = {0: "Basics content here.", 1: "Advanced content here."}

        chapter = structurer.write_single_chapter(outline, [t0, t1], cleaned)

        assert chapter is not None
        assert "## Part 1: Basics" in chapter.content
        assert "## Part 2: Advanced" in chapter.content

    def test_fallback_chapter_is_cached(self, tmp_path):
        """Fallback chapter gets written to the cache directory."""
        llm = FakeLLM(should_fail=True)
        structurer = _make_structurer(llm, tmp_path)

        video = _make_video(0, "Cached Chapter")
        transcript = _make_transcript(video, full_text="some text")

        outline = ChapterOutline(
            number=3, title="Cached Chapter",
            video_indices=[0], description="",
        )
        cleaned = {0: "Content to cache."}

        chapter = structurer.write_single_chapter(outline, [transcript], cleaned)

        cache_path = tmp_path / "chapters" / "chapter_03.md"
        assert cache_path.exists()
        assert "Content to cache" in cache_path.read_text()

    def test_normal_chapter_on_success(self, tmp_path):
        """When LLM succeeds, normal chapter writing path is used."""
        llm = FakeLLM(
            response="This is polished prose from the LLM.\n\n"
            "<summary>Summary of the chapter.</summary>"
        )
        structurer = _make_structurer(llm, tmp_path)

        video = _make_video(0, "Good Chapter")
        transcript = _make_transcript(video, full_text="transcript text")

        outline = ChapterOutline(
            number=1, title="Good Chapter",
            video_indices=[0], description="desc",
        )
        cleaned = {0: "Transcript text that will be sent to the LLM."}

        chapter = structurer.write_single_chapter(outline, [transcript], cleaned)

        assert chapter is not None
        assert "polished prose" in chapter.content
        # Summary tags should be stripped
        assert "<summary>" not in chapter.content

    def test_fallback_returns_none_for_empty_transcripts(self, tmp_path):
        """Fallback returns None when no transcript text is available."""
        llm = FakeLLM(should_fail=True)
        structurer = _make_structurer(llm, tmp_path)

        video = _make_video(0, "Empty")
        transcript = _make_transcript(video, full_text="")

        outline = ChapterOutline(
            number=1, title="Empty Chapter",
            video_indices=[0], description="",
        )
        cleaned = {0: "   "}  # whitespace only

        chapter = structurer.write_single_chapter(outline, [transcript], cleaned)

        # Should return None because combined_text is empty (before even trying LLM)
        assert chapter is None


# ── _split_by_keyframes Tests ────────────────────────────────────────


class TestSplitByKeyframes:
    def test_splits_correctly(self):
        segments = _make_segments([
            (0.0, "Hello world."),
            (30.0, "More intro stuff."),
            (120.0, "Now the architecture."),
            (150.0, "Layers and layers."),
        ])
        chapters = [
            VideoChapter(title="Introduction", start_time=0.0),
            VideoChapter(title="Architecture", start_time=120.0),
        ]

        result = BookStructurer._split_by_keyframes(segments, chapters)

        assert len(result) == 2
        assert result[0][0] == "Introduction"
        assert "Hello world" in result[0][1]
        assert "More intro" in result[0][1]
        assert result[1][0] == "Architecture"
        assert "architecture" in result[1][1]
        assert "Layers" in result[1][1]

    def test_empty_segments(self):
        chapters = [VideoChapter(title="Intro", start_time=0.0)]
        result = BookStructurer._split_by_keyframes([], chapters)

        assert len(result) == 1
        assert result[0][0] == "Content"

    def test_empty_chapters(self):
        segments = _make_segments([(0.0, "Some text.")])
        result = BookStructurer._split_by_keyframes(segments, [])

        assert len(result) == 1
        assert result[0][0] == "Content"
        assert "Some text" in result[0][1]

    def test_single_keyframe(self):
        segments = _make_segments([
            (0.0, "First sentence."),
            (10.0, "Second sentence."),
        ])
        chapters = [VideoChapter(title="Everything", start_time=0.0)]

        result = BookStructurer._split_by_keyframes(segments, chapters)

        assert len(result) == 1
        assert result[0][0] == "Everything"
        assert "First" in result[0][1]
        assert "Second" in result[0][1]

    def test_keyframes_out_of_order_are_sorted(self):
        segments = _make_segments([
            (0.0, "Alpha."),
            (100.0, "Beta."),
        ])
        chapters = [
            VideoChapter(title="Second", start_time=100.0),
            VideoChapter(title="First", start_time=0.0),
        ]

        result = BookStructurer._split_by_keyframes(segments, chapters)

        assert result[0][0] == "First"
        assert result[1][0] == "Second"
