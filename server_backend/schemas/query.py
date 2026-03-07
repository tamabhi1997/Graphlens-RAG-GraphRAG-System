from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    scope_type: Literal["video", "course"] = "video"
    scope_id: Optional[str] = None
    collection_name: str = "graphlens_chunks"


class SourceOut(BaseModel):
    chunk_id: str
    similarity: float
    video_id: Optional[str] = None
    course_id: Optional[str] = None
    source_url: Optional[str] = None
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    text: str


class QueryResponse(BaseModel):
    refused: bool
    reason: Optional[str] = None
    best_similarity: Optional[float] = None
    answer: Optional[str] = None
    sources: List[SourceOut] = Field(default_factory=list)