#YouTube transcript fetching + normalization, Then pipelines/ingest.py calls this function.
#Why normalize?
#So chunking/embedding/vector DB code stays identical for every source.

# A chunk has:
    # chunk_text
    # start_seconds (from first segment)
    # end_seconds (from last segment)
    # metadata: video_id, source_url, etc.

# Embedding stage (chunks → vectors)
    # Each chunk becomes a vector embedding.
    # Why embeddings?
        # It lets you do semantic search: “SQL joins types” finds relevant chunks even if exact words differ.

# Vector DB stage (store embeddings + metadata)
# Store:
    # vector
    # chunk text
    # timestamps
    # video_id
# RAG index is ready.

# Query stage (question → answer)
    # When user asks a question:
        # embed the question
        # retrieve top-k chunks from vector DB (optionally filter by video_id)
        # send those chunks to the LLM
        # return answer (+ timestamps for citations)

#flow to understand the transcript part
# Frontend sends URL
# Backend extracts video_id
# Backend tries youtube-transcript-api (fastest, clean structured segments)
# If it fails (disabled/unavailable), backend optionally falls back to yt-dlp (downloads captions only, not video)
# Backend returns a normalized dict like you showed (source_url, duration optional, segments with start/end/text)
# And yes: you typically don’t send the full transcript to frontend in production, but having a /transcript endpoint is very useful:
# debuggin
# showing “preview transcript” in UI
# confirming ingestion works

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse


# -----------------------------
# Exceptions (clean error handling)
# -----------------------------
class InvalidYouTubeUrl(ValueError):
    """Raised when we cannot extract a YouTube video id from the given URL."""


class TranscriptUnavailable(RuntimeError):
    """Raised when transcript/captions cannot be fetched by any provider."""


# -----------------------------
# Normalized internal structure
# -----------------------------
@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start_seconds: float
    end_seconds: float



_WHITESPACE_RE = re.compile(r"\s+")

def normalize_caption_text(text: str) -> str:
    """
    Normalize caption text so chunking/embeddings aren't polluted by weird whitespace.
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
    Supports common formats:
      - https://www.youtube.com/watch?v=VIDEOID
      - https://youtu.be/VIDEOID
      - https://www.youtube.com/shorts/VIDEOID
      - https://www.youtube.com/embed/VIDEOID
    """
    if not url or not isinstance(url, str):
        raise InvalidYouTubeUrl("URL is empty or not a string.")

    parsed = urlparse(url.strip())
    host = (parsed.netloc or "").lower()
    path = parsed.path.strip("/")

    if "youtu.be" in host:
        if path:
            return path.split("/")[0]
        raise InvalidYouTubeUrl("Could not extract video id from youtu.be URL.")

    if "youtube.com" in host:
        if path == "watch":
            vid = (parse_qs(parsed.query).get("v") or [None])[0]
            if vid:
                return vid
            raise InvalidYouTubeUrl("Missing v= parameter in watch URL.")

        m = re.match(r"^(shorts|embed)/([^/?#]+)", path)
        if m:
            return m.group(2)

    raise InvalidYouTubeUrl("Not a recognized YouTube URL format.")


# -----------------------------
# Provider 1: youtube-transcript-api
# -----------------------------
def _fetch_with_youtube_transcript_api(video_id: str, languages: List[str]) -> List[TranscriptSegment]:
    """
    Supports the newer youtube-transcript-api API (YouTubeTranscriptApi().fetch(...)).
    Also keeps compatibility with older versions if get_transcript exists.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        # Exception locations vary slightly across versions, so we try both.
        try:
            from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
        except Exception:
            from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
    except Exception as e:
        raise TranscriptUnavailable("youtube-transcript-api not installed or import failed.") from e

    try:
        # Older versions used classmethod get_transcript; newer use instance.fetch
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            items = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)  # type: ignore[attr-defined]
        else:
            ytt_api = YouTubeTranscriptApi()
            fetched = ytt_api.fetch(video_id, languages=languages)  # :contentReference[oaicite:1]{index=1}
            items = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else list(fetched)
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        raise TranscriptUnavailable(str(e)) from e
    except VideoUnavailable as e:
        raise TranscriptUnavailable("Video unavailable (private/removed/region-blocked).") from e
    except Exception as e:
        raise TranscriptUnavailable(f"Failed to fetch transcript: {e}") from e

    segments: List[TranscriptSegment] = []
    for it in items:
        raw = it.get("text") or ""
        text = normalize_caption_text(raw)
        start = float(it.get("start") or 0.0)
        duration = float(it.get("duration") or 0.0)
        end = start + duration
        if text:
            segments.append(TranscriptSegment(text=text, start_seconds=start, end_seconds=end))

    if not segments:
        raise TranscriptUnavailable("Transcript returned empty.")
    return segments


# -----------------------------
# Provider 2 (fallback): yt-dlp subtitles (.vtt) + tiny parser
# -----------------------------
_VTT_TS = re.compile(
    r"(?P<sh>\d{2}):(?P<sm>\d{2}):(?P<ss>\d{2})\.(?P<sms>\d{3})\s*-->\s*"
    r"(?P<eh>\d{2}):(?P<em>\d{2}):(?P<es>\d{2})\.(?P<ems>\d{3})"
)


def _ts_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _parse_vtt(vtt_text: str) -> List[TranscriptSegment]:
    lines = [ln.rstrip("\n") for ln in vtt_text.splitlines()]
    segments: List[TranscriptSegment] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line.upper().startswith("WEBVTT"):
            i += 1
            continue

        m = _VTT_TS.search(line)
        if not m:
            i += 1
            continue

        start = _ts_to_seconds(m.group("sh"), m.group("sm"), m.group("ss"), m.group("sms"))
        end = _ts_to_seconds(m.group("eh"), m.group("em"), m.group("es"), m.group("ems"))

        i += 1
        text_lines = []
        while i < len(lines) and lines[i].strip() != "":
            t = re.sub(r"<[^>]+>", "", lines[i]).strip()
            if t:
                text_lines.append(t)
            i += 1

        text = normalize_caption_text(" ".join(text_lines))
        if text:
            segments.append(TranscriptSegment(text=text, start_seconds=start, end_seconds=end))

        i += 1

    if not segments:
        raise TranscriptUnavailable("Failed to parse VTT transcript (empty).")
    return segments


def _fetch_with_ytdlp(url: str, languages: List[str]) -> List[TranscriptSegment]:
    with tempfile.TemporaryDirectory() as td:
        outtmpl = str(Path(td) / "%(id)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-format",
            "vtt",
            "--sub-lang",
            ",".join(languages),
            "-o",
            outtmpl,
            url,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise TranscriptUnavailable("yt-dlp not installed (pip install yt-dlp).") from e
        except subprocess.CalledProcessError as e:
            err = (e.stderr or "").strip()
            raise TranscriptUnavailable(f"yt-dlp failed to fetch subtitles. {err[:250]}") from e

        vtt_files = list(Path(td).glob("*.vtt"))
        if not vtt_files:
            raise TranscriptUnavailable("yt-dlp did not produce any .vtt subtitle files.")

        vtt_text = vtt_files[0].read_text(encoding="utf-8", errors="ignore")
        return _parse_vtt(vtt_text)


# -----------------------------
# Optional: metadata (title/duration) via yt-dlp (best-effort)
# -----------------------------
def _fetch_metadata_with_ytdlp(url: str) -> Dict[str, Any]:
    cmd = ["yt-dlp", "--skip-download", "--dump-single-json", url]
    try:
        p = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(p.stdout)
        return {
            "title": data.get("title"),
            "duration_seconds": data.get("duration"),
        }
    except Exception:
        return {"title": None, "duration_seconds": None}


# -----------------------------
# Public function: normalized transcript doc
# -----------------------------
def get_transcript_document(
    source_url: str,
    languages: Optional[List[str]] = None,
    allow_ytdlp_fallback: bool = True,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Returns a normalized transcript document for downstream chunking/embedding.

    Output shape:
      {
        "source_url": str,
        "video_id": str,
        "title": Optional[str],
        "duration_seconds": Optional[int|float],
        "language": str,
        "provider": "youtube_transcript_api" | "yt_dlp",
        "segment_count": int,
        "segments": [
          {"start_seconds": float, "end_seconds": float, "text": str}, ...
        ]
      }
    """
    languages = languages or ["en"]
    video_id = extract_video_id(source_url)

    metadata = {"title": None, "duration_seconds": None}
    if include_metadata:
        metadata = _fetch_metadata_with_ytdlp(source_url)

    # Try primary provider first
    try:
        segs = _fetch_with_youtube_transcript_api(video_id, languages=languages)
        provider = "youtube_transcript_api"
    except TranscriptUnavailable:
        if not allow_ytdlp_fallback:
            raise
        segs = _fetch_with_ytdlp(source_url, languages=languages)
        provider = "yt_dlp"

    return {
        "source_url": source_url,
        "video_id": video_id,
        "title": metadata.get("title"),
        "duration_seconds": metadata.get("duration_seconds"),
        "language": languages[0],
        "provider": provider,
        "segment_count": len(segs),
        "segments": [
            {
                "start_seconds": s.start_seconds,
                "end_seconds": s.end_seconds,
                "text": s.text,
            }
            for s in segs
        ],
    }


