from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IndexPdfResponse(BaseModel):
    scope_type: str
    scope_id: str
    summary: str
    key_topics: List[str] = Field(default_factory=list)
    page_count: int
    doc_id: str
    filename: str
    chunks_indexed: int
    collection_name: str
    pdf_url: Optional[str] = None    