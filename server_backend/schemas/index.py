from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IndexYoutubeRequest(BaseModel):
    url: str
    languages: List[str] = Field(default_factory=lambda: ["en"])
    collection_name: str = "graphlens_chunks"
    force_reindex: bool = False
    chunk_cfg: Optional[Dict[str, Any]] = None
    course_id: Optional[str] = None  # future course scope


class IndexYoutubeResponse(BaseModel):
    # what frontend needs for later queries
    scope_type: str
    scope_id: str

    # UI content
    summary: str
    key_topics: List[str] = Field(default_factory=list)
    estimated_duration_seconds: Optional[float] = None
    estimated_duration: Optional[str] = None

    # useful debugging / UI display
    video_id: str
    source_url: str
    chunks_indexed: int
    collection_name: str