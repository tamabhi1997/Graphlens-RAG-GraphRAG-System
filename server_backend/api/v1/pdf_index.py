from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import json
import os
import shutil


from server_backend.schemas.pdf_index import IndexPdfResponse
from graphlens.pipelines.ingest_pdf_v1 import ingest_pdf_v1

router = APIRouter()


@router.post("/pdf/index", response_model=IndexPdfResponse)
async def pdf_index(
    file: UploadFile = File(..., description="PDF file to index"),
    collection_name: str = Form(default="graphlens_chunks"),
    force_reindex: bool = Form(default=False),
    course_id: Optional[str] = Form(default=None),
    chunk_cfg: Optional[str] = Form(default=None),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    # ← ADD HERE: save to disk for static serving
    safe_filename = file.filename.replace(" ", "_")
    pdf_dir = "data/pdfs"
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, safe_filename)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
        file.seek(0)

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    MAX_BYTES = 50 * 1024 * 1024
    if len(pdf_bytes) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large.")

    parsed_chunk_cfg = None
    if chunk_cfg:
        try:
            parsed_chunk_cfg = json.loads(chunk_cfg)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="chunk_cfg must be valid JSON.")

    try:
        out = ingest_pdf_v1(
            pdf_bytes=pdf_bytes,
            filename=file.filename,
            collection_name=collection_name,
            force_reindex=force_reindex,
            chunk_cfg=parsed_chunk_cfg,
            course_id=course_id,
        )
        # ← ADD HERE: attach pdf_url to response
        out["pdf_url"] = f"http://127.0.0.1:8000/static/pdfs/{safe_filename}"
        return IndexPdfResponse(**out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))