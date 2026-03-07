from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi  # :contentReference[oaicite:2]{index=2}


class InvalidYouTubeUrl(ValueError):
    pass


@dataclass(frozen=True)
class Segment:
    text: str
    start_seconds: float
    end_seconds: float


_WHITESPACE_RE = re.compile(r"\s+")

def normalize_caption_text(text: str) -> str:
    """
    Normalize caption text so embeddings/chunking aren't polluted by weird whitespace.
    - converts non-breaking spaces to normal spaces
    - removes line breaks
    - collapses repeated whitespace
    """
    text = text.replace("\u00a0", " ")  # NBSP (\xa0)
    text = text.replace("\r", " ").replace("\n", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def extract_video_id(url: str) -> str:
    """
    Very small URL → video_id helper.
    Supports:
      - youtube.com/watch?v=VIDEOID
      - youtu.be/VIDEOID
      - youtube.com/shorts/VIDEOID
      - youtube.com/embed/VIDEOID
    """
    parsed = urlparse(url.strip())
    host = (parsed.netloc or "").lower()
    path = parsed.path.strip("/")

    if "youtu.be" in host:
        if not path:
            raise InvalidYouTubeUrl("Invalid youtu.be URL")
        return path.split("/")[0]

    if "youtube.com" in host:
        if path == "watch":
            vid = (parse_qs(parsed.query).get("v") or [None])[0]
            if not vid:
                raise InvalidYouTubeUrl("Missing v= parameter")
            return vid

        m = re.match(r"^(shorts|embed)/([^/?#]+)", path)
        if m:
            return m.group(2)

    raise InvalidYouTubeUrl("Not a recognized YouTube URL format")


def get_transcript_segments_v1(source_url: str, languages: Optional[List[str]] = None) -> Dict:
    """
    Minimal MVP:
    - takes a YouTube URL
    - fetches transcript via youtube-transcript-api
    - returns normalized segments
    """
    languages = languages or ["en"]
    video_id = extract_video_id(source_url)

    ytt = YouTubeTranscriptApi()
    fetched = ytt.fetch(video_id, languages=languages)  # :contentReference[oaicite:3]{index=3}
    items = fetched.to_raw_data()  # list[{"text","start","duration"}] :contentReference[oaicite:4]{index=4}

    segments: List[Segment] = []
    for it in items:
        raw = it.get("text") or ""
        text = normalize_caption_text(raw)
        start = float(it.get("start") or 0.0)
        dur = float(it.get("duration") or 0.0)
        end = start + dur
        if text:
            segments.append(Segment(text=text, start_seconds=start, end_seconds=end))

    return {
        "source_url": source_url,
        "video_id": video_id,
        "language": languages[0],
        "provider": "youtube_transcript_api",
        "segment_count": len(segments),
        "segments": [
            {"start_seconds": s.start_seconds, "end_seconds": s.end_seconds, "text": s.text}
            for s in segments
        ],
    }