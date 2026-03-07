# Pydantic models for transcript-related requests and responses.
# Defines the expected input (YouTube URL) and output (segments with start/end/text).

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class TranscriptRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")
    languages: List[str] = Field(default_factory=lambda: ["en"], description="Preferred transcript languages")


class TranscriptSegmentOut(BaseModel):
    start_seconds: float
    end_seconds: float
    text: str


class TranscriptResponse(BaseModel):
    source_url: str
    video_id: str
    language: str
    provider: str
    segment_count: int
    segments: List[TranscriptSegmentOut]

    # Optional fields (keep for future v2; v1 can return None)
    title: Optional[str] = None
    duration_seconds: Optional[float] = None