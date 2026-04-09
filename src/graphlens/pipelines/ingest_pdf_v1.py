from __future__ import annotations

import os
from dotenv import load_dotenv

from graphlens.utils.pdf_v1 import extract_pdf_segments
from graphlens.chunking.hybrid_v2 import chunk_transcript_doc_v2
from graphlens.chunking.chunk_cleaner import clean_for_embedding
from graphlens.vectorstores.chroma_v1 import ChromaStore
from graphlens.pipelines.summarize_v1 import (
    make_summary_from_chunks,
    extract_key_topics,
)
from graphlens.embeddings.openai_v1 import embed_texts


def ingest_pdf_v1(
    pdf_bytes: bytes,
    filename: str = "document.pdf",
    collection_name: str = "graphlens_chunks",
    force_reindex: bool = False,
    chunk_cfg: dict = None,
    batch_size: int = 64,
    store_raw_text: bool = True,
    course_id: str = None,
) -> dict:
    """
    Full PDF ingestion pipeline:
      1) Extract text segments from PDF bytes (PyMuPDF)
      2) Chunk segments (hybrid_v2 — token-based + hard boundaries)
      3) Clean chunk text (remove noise)
      4) Embed cleaned chunks (OpenAI)
      5) Store in ChromaDB with scope_type="document"

    Args:
        pdf_bytes:        Raw bytes of the uploaded PDF.
        filename:         Original filename — used for doc_id + metadata.
        collection_name:  Chroma collection (same as YouTube — shared store).
        force_reindex:    If True, wipes existing chunks for this doc_id first.
        chunk_cfg:        Optional overrides for chunker params.
        batch_size:       Embedding batch size (OpenAI limit safe at 64).
        store_raw_text:   If True, stores uncleaned text in metadata["raw_text"].
        course_id:        Optional course scope for cross-doc querying.

    Returns:
        Dict with scope_type, scope_id, summary, key_topics, chunks_indexed, etc.
        Same shape as ingest_youtube_url_v1 return value.
    """
    load_dotenv()

    if chunk_cfg is None:
        chunk_cfg = {}

    # -------------------------
    # 1) Extract PDF segments
    # -------------------------
    doc = extract_pdf_segments(pdf_bytes, filename=filename)
    doc_id = doc["doc_id"]
    page_count = doc["page_count"]
    segment_count = doc["segment_count"]

    print(f"[pdf_ingest] Extracted {segment_count} segments from {page_count} pages")

    # -------------------------
    # 2) Chunking
    # -------------------------
    chunks = chunk_transcript_doc_v2(
        doc,
        max_tokens=chunk_cfg.get("max_tokens", 400),
        min_tokens=chunk_cfg.get("min_tokens", 80),   # lower than video — PDFs have short paragraphs
        overlap_tokens=chunk_cfg.get("overlap_tokens", 40),
        max_seconds=chunk_cfg.get("max_seconds", 9999.0),  # effectively disabled for PDFs
    )

    print(f"[pdf_ingest] Produced {len(chunks)} chunks")

    # -------------------------
    # 3) Cleaning
    # -------------------------
    raw_texts = [c.text for c in chunks]
    clean_texts = [clean_for_embedding(t) for t in raw_texts]

    # -------------------------
    # 4) Embeddings
    # -------------------------
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    vectors = []

    for i in range(0, len(clean_texts), batch_size):
        batch = clean_texts[i: i + batch_size]
        vectors.extend(embed_texts(batch))

    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding mismatch: {len(vectors)} vectors for {len(chunks)} chunks."
        )

    # -------------------------
    # 5) Store in Chroma
    # -------------------------
    store = ChromaStore(
        persist_path=os.getenv("CHROMA_PERSIST_PATH", "data/chroma"),
        collection_name=collection_name,
    )

    if force_reindex:
        try:
            # Reuse delete_by_video_id — doc_id is stored as video_id internally
            store.delete_by_video_id(doc_id)
        except Exception:
            pass

    ids = [c.chunk_id for c in chunks]

    metadatas = []
    for c, raw in zip(chunks, raw_texts):
        md = dict(c.metadata)
        md["doc_id"] = doc_id
        md["video_id"] = doc_id        # keeps Chroma filter compatible
        md["scope_type"] = "document"
        md["filename"] = filename
        md["page_count"] = page_count
        md["source_url"] = filename

        # Replace seconds with page numbers for PDF citation
        # chunk metadata already has start_seconds = page_number (set in extractor)
        md["start_page"] = int(c.start_seconds)
        md["end_page"] = int(c.end_seconds)

        if course_id:
            md["course_id"] = course_id

        if store_raw_text:
            md["raw_text"] = raw

        metadatas.append(md)

    store.upsert(
        ids=ids,
        embeddings=vectors,
        documents=clean_texts,
        metadatas=metadatas,
    )

    # -------------------------
    # Summary + topics
    # -------------------------
    # lead_text = clean_texts[0] if clean_texts else ""
    # summary = make_summary_from_text(lead_text, max_sentences=3, max_chars=600)
    summary = make_summary_from_chunks(clean_texts, max_sentences=3, max_chars=600)

    topic_texts = clean_texts[:8]
    key_topics = extract_key_topics(topic_texts, top_n=8)

    return {
        # scope for frontend queries
        "scope_type": "document",
        "scope_id": doc_id,

        # UI-friendly info
        "summary": summary,
        "key_topics": key_topics,
        "page_count": page_count,

        # debug / display
        "doc_id": doc_id,
        "filename": filename,
        "segments": segment_count,
        "chunks_indexed": len(chunks),
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "chroma_path": os.getenv("CHROMA_PERSIST_PATH", "data/chroma"),
    }