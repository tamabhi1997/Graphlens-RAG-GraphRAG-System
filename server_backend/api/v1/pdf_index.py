from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import json

from server_backend.schemas.pdf_index import IndexPdfResponse
from graphlens.pipelines.ingest_pdf_v1 import ingest_pdf_v1

router = APIRouter()


@router.post("/pdf/index", response_model=IndexPdfResponse)
async def pdf_index(
    file: UploadFile = File(..., description="PDF file to index"),
    collection_name: str = Form(default="graphlens_chunks"),
    force_reindex: bool = Form(default=False),
    course_id: Optional[str] = Form(default=None),
    chunk_cfg: Optional[str] = Form(default=None),  # JSON string from frontend
):
    """
    Ingest a PDF file into GraphLens.

    The frontend sends a multipart/form-data request with:
      - file:            the actual PDF bytes
      - collection_name: (optional) Chroma collection name
      - force_reindex:   (optional) wipe and re-index if already stored
      - course_id:       (optional) group multiple docs under a course scope
      - chunk_cfg:       (optional) JSON string of chunker overrides

    Returns scope_type="document" and scope_id=doc_id for subsequent queries.
    """

    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file."
        )

    # Read bytes
    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 50 MB hard limit
    MAX_BYTES = 50 * 1024 * 1024
    if len(pdf_bytes) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(pdf_bytes) // (1024*1024)} MB). Maximum is 50 MB."
        )

    # Parse optional chunk_cfg JSON string
    parsed_chunk_cfg = None
    if chunk_cfg:
        try:
            parsed_chunk_cfg = json.loads(chunk_cfg)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="chunk_cfg must be valid JSON.")

    # Run ingest pipeline
    try:
        out = ingest_pdf_v1(
            pdf_bytes=pdf_bytes,
            filename=file.filename,
            collection_name=collection_name,
            force_reindex=force_reindex,
            chunk_cfg=parsed_chunk_cfg,
            course_id=course_id,
        )
        return IndexPdfResponse(**out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))