"""Transcript fetching from YouTube playlists and individual videos."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from youtube_transcript_api import YouTubeTranscriptApi

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

from src.config import Config

logger = logging.getLogger(__name__)


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


# ── Playlist extraction ─────────────────────────────────────────────

class PlaylistExtractor:
    """Resolve a YouTube URL into a list of VideoInfo objects.

    Handles both playlist URLs and single-video URLs.
    """
    
    def __init__(self, config: Config | None = None) -> None:
        self.config = config

    @staticmethod
    def _is_playlist_url(url: str) -> bool:
        """Check if URL contains a playlist parameter."""
        return "list=" in url

    @staticmethod
    def _fetch_chapters(video_id: str) -> list:
        """Fetch YouTube chapter markers for a video via yt-dlp.

        Returns a list of VideoChapter objects, or an empty list if the
        video has no chapters or the fetch fails.
        """
        url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }
        if self.config and self.config.transcript.cookies_file:
            ydl_opts["cookiefile"] = self.config.transcript.cookies_file
            
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            raw_chapters = (info or {}).get("chapters") or []
            return [
                VideoChapter(
                    title=c.get("title", ""),
                    start_time=float(c.get("start_time", 0)),
                )
                for c in raw_chapters
            ]
        except Exception as exc:
            logger.debug("Could not fetch chapters for %s: %s", video_id, exc)
            return []

    def extract(self, url: str) -> List[VideoInfo]:
        """Return ordered list of VideoInfo for every video in the URL."""
        if self._is_playlist_url(url):
            return self._extract_playlist(url)
        return self._extract_single(url)

    def _extract_playlist(self, url: str) -> List[VideoInfo]:
        """Extract all videos from a playlist URL."""
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": "in_playlist",
            "skip_download": True,
        }
        if self.config and self.config.transcript.cookies_file:
            ydl_opts["cookiefile"] = self.config.transcript.cookies_file

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        if info is None:
            raise RuntimeError(f"Could not extract info from URL: {url}")

        entries = info.get("entries", [])
        if not entries:
            logger.warning("No entries found in playlist, trying as single video.")
            return self._extract_single(url)

        videos: List[VideoInfo] = []
        for idx, entry in enumerate(entries):
            if entry is None:
                continue
            # In flat extraction, the 'id' or 'url' field holds the video ID
            video_id = entry.get("id", "")
            if not video_id:
                video_id = entry.get("url", "")
            # Skip if we got a playlist ID instead of a video ID
            if video_id.startswith("PL") or len(video_id) > 15:
                logger.warning("Skipping non-video entry: %s", video_id)
                continue
            chapters = self._fetch_chapters(video_id)
            videos.append(
                VideoInfo(
                    video_id=video_id,
                    title=entry.get("title", f"Video {idx + 1}"),
                    duration=float(entry.get("duration") or 0),
                    index=idx,
                    chapters=chapters,
                )
            )
            logger.debug(
                "Video %d '%s': %d chapter(s) found.",
                idx,
                entry.get("title", ""),
                len(chapters),
            )

        logger.info("Extracted %d videos from playlist.", len(videos))
        return videos

    def _extract_single(self, url: str) -> List[VideoInfo]:
        """Extract info for a single video URL."""
        # Try to extract video ID from URL directly
        video_id = self._extract_video_id(url)
        if not video_id:
            # Fall back to yt-dlp full extraction
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "noplaylist": True,
            }
            if self.config and self.config.transcript.cookies_file:
                ydl_opts["cookiefile"] = self.config.transcript.cookies_file
                
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            if info is None:
                raise RuntimeError(f"Could not extract info from URL: {url}")
            video_id = info.get("id", "")
            raw_chapters = info.get("chapters") or []
            chapters = [
                VideoChapter(
                    title=c.get("title", ""),
                    start_time=float(c.get("start_time", 0)),
                )
                for c in raw_chapters
            ]
            return [
                VideoInfo(
                    video_id=video_id,
                    title=info.get("title", "Untitled"),
                    duration=float(info.get("duration") or 0),
                    index=0,
                    chapters=chapters,
                )
            ]

        chapters = self._fetch_chapters(video_id)
        return [
            VideoInfo(
                video_id=video_id,
                title="Untitled",
                duration=0,
                index=0,
                chapters=chapters,
            )
        ]

    @staticmethod
    def _extract_video_id(url: str) -> Optional[str]:
        """Extract YouTube video ID from URL string."""
        patterns = [
            r"(?:v=|/v/)([a-zA-Z0-9_-]{11})",
            r"youtu\.be/([a-zA-Z0-9_-]{11})",
        ]
        for pattern in patterns:
            m = re.search(pattern, url)
            if m:
                return m.group(1)
        return None


# ── Transcript fetching ─────────────────────────────────────────────

class TranscriptFetcher:
    """Fetch and cache transcripts for individual videos.

    Uses youtube-transcript-api v1.2+ instance-based API.
    """

    def __init__(self, config: Config) -> None:
        self._languages = config.transcript.languages
        self._cache_dir = Path(config.transcript.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load custom cookies if provided
        cookies_file = config.transcript.cookies_file
        if cookies_file:
            logger.info("Using yt cookies from: %s", cookies_file)
            self._api = YouTubeTranscriptApi(cookies=cookies_file)
        else:
            self._api = YouTubeTranscriptApi()

    # ── public API ───────────────────────────────────────────────

    def fetch(self, video: VideoInfo) -> VideoTranscript:
        """Return the transcript for *video*, using cache when available."""
        cached = self._load_cache(video.video_id)
        if cached is not None:
            logger.info("Using cached transcript for '%s'.", video.title)
            return cached

        logger.info("Fetching transcript for '%s' (%s)…", video.title, video.video_id)
        try:
            # v1.2+ API: instance method .fetch(video_id)
            transcript = self._api.fetch(video.video_id, languages=self._languages)
            segments = [
                TranscriptSegment(
                    text=snippet.text,
                    start=snippet.start,
                    duration=snippet.duration,
                )
                for snippet in transcript
            ]
        except Exception:
            # Fall back: try listing available transcripts and pick any
            logger.warning(
                "Preferred languages unavailable for '%s'; trying any available transcript.",
                video.title,
            )
            try:
                transcript_list = self._api.list(video.video_id)
                # Try to find a generated transcript in preferred languages
                found = None
                for t in transcript_list:
                    if t.language_code in self._languages:
                        found = t
                        break
                if found is None:
                    # Just take the first available transcript
                    found = transcript_list[0]
                fetched = found.fetch()
                segments = [
                    TranscriptSegment(
                        text=snippet.text,
                        start=snippet.start,
                        duration=snippet.duration,
                    )
                    for snippet in fetched
                ]
            except Exception as exc:
                logger.error("No transcript available for '%s': %s", video.title, exc)
                raise

        full_text = " ".join(seg.text for seg in segments)
        result = VideoTranscript(video=video, segments=segments, full_text=full_text)

        self._save_cache(result)
        return result

    def fetch_all(self, videos: List[VideoInfo]) -> List[VideoTranscript]:
        """Fetch transcripts for all videos, skipping failures."""
        transcripts: List[VideoTranscript] = []
        for video in videos:
            try:
                transcripts.append(self.fetch(video))
            except Exception as exc:
                logger.error("Skipping video '%s': %s", video.title, exc)
        return transcripts

    # ── cache helpers ────────────────────────────────────────────

    def _cache_path(self, video_id: str) -> Path:
        return self._cache_dir / f"{video_id}.json"

    def _load_cache(self, video_id: str) -> Optional[VideoTranscript]:
        path = self._cache_path(video_id)
        if not path.exists():
            return None
        try:
            with open(path, "r") as fh:
                data = json.load(fh)
            video_data = data["video"]
            # Deserialise chapters list if present (backward-compatible)
            raw_chapters = video_data.pop("chapters", [])
            video = VideoInfo(
                **video_data,
                chapters=[
                    VideoChapter(title=c["title"], start_time=c["start_time"])
                    for c in raw_chapters
                ],
            )
            segments = [TranscriptSegment(**s) for s in data["segments"]]
            return VideoTranscript(
                video=video, segments=segments, full_text=data["full_text"]
            )
        except Exception as exc:
            logger.warning("Corrupt cache for %s, refetching: %s", video_id, exc)
            return None

    def _save_cache(self, transcript: VideoTranscript) -> None:
        path = self._cache_path(transcript.video.video_id)
        data = {
            "video": asdict(transcript.video),
            "segments": [asdict(s) for s in transcript.segments],
            "full_text": transcript.full_text,
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        logger.debug("Cached transcript to %s.", path)
