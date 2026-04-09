from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IndexPdfResponse(BaseModel):
    # what frontend needs for later queries
    scope_type: str          # always "document"
    scope_id: str            # doc_id — send this back with every query

    # UI content
    summary: str
    key_topics: List[str] = Field(default_factory=list)
    page_count: int

    # useful for debugging / UI display
    doc_id: str
    filename: str
    chunks_indexed: int
    collection_name: str