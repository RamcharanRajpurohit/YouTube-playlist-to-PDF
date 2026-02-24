"""Load pre-fetched transcript and playlist data from the local data/ directory.

All YouTube fetching logic has been removed. This module reads the permanent
data stored in ``data/transcripts/`` and ``data/playlist_metadata/``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Data directory (repo root / data) ────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class VideoChapter:
    """A single YouTube chapter marker (keyframe)."""

    title: str
    start_time: float  # seconds


@dataclass
class VideoInfo:
    """Metadata for a single video in the playlist."""

    video_id: str
    title: str
    duration: float  # seconds
    index: int  # 0-based position in playlist
    chapters: list = field(default_factory=list)  # List[VideoChapter]


@dataclass
class TranscriptSegment:
    """A single timed segment from a transcript."""

    text: str
    start: float
    duration: float


@dataclass
class VideoTranscript:
    """Full transcript for one video."""

    video: VideoInfo
    segments: List[TranscriptSegment]
    full_text: str


# ── Playlist metadata loader ────────────────────────────────────────

def load_playlist_metadata() -> tuple[str, List[VideoInfo]]:
    """Load all video metadata from ``data/playlist_metadata/videos_metadata.json``.

    Returns:
        (book_title, list_of_VideoInfo)
    """
    meta_path = _DATA_DIR / "playlist_metadata" / "videos_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Playlist metadata not found at {meta_path}. "
            "Ensure data/playlist_metadata/videos_metadata.json exists."
        )

    with open(meta_path, "r") as fh:
        data = json.load(fh)

    book_title: str = data.get("book_title", "Untitled")
    videos: List[VideoInfo] = []
    for v in data.get("videos", []):
        chapters = [
            VideoChapter(title=ch["title"], start_time=ch["start_time"])
            for ch in v.get("chapters", [])
        ]
        videos.append(
            VideoInfo(
                video_id=v["video_id"],
                title=v["title"],
                duration=v.get("duration_seconds", 0),
                index=v["index"],
                chapters=chapters,
            )
        )

    logger.info("Loaded metadata for %d videos from %s", len(videos), meta_path)
    return book_title, videos


# ── Transcript loader ────────────────────────────────────────────────

def load_transcript(video: VideoInfo) -> VideoTranscript:
    """Load a single transcript from ``data/transcripts/{video_id}.json``.

    Raises FileNotFoundError if the transcript file is missing.
    """
    path = _DATA_DIR / "transcripts" / f"{video.video_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Transcript not found for video '{video.title}' ({video.video_id}) "
            f"at {path}"
        )

    with open(path, "r") as fh:
        data = json.load(fh)

    segments = [TranscriptSegment(**s) for s in data.get("segments", [])]
    full_text = data.get("full_text", "")

    logger.debug("Loaded transcript for '%s' (%d segments)", video.title, len(segments))
    return VideoTranscript(video=video, segments=segments, full_text=full_text)


def load_all_transcripts(videos: List[VideoInfo]) -> List[VideoTranscript]:
    """Load transcripts for all videos, skipping any that are missing."""
    transcripts: List[VideoTranscript] = []
    for video in videos:
        try:
            transcripts.append(load_transcript(video))
        except FileNotFoundError as exc:
            logger.error("Skipping video '%s': %s", video.title, exc)
    return transcripts
