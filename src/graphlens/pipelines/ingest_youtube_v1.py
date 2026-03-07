import os
from dotenv import load_dotenv

from graphlens.utils.youtube_v1 import get_transcript_segments_v1
from graphlens.chunking.hybrid_v1 import chunk_transcript_doc
from graphlens.chunking.chunk_cleaner import clean_for_embedding
from graphlens.vectorstores.chroma_v1 import ChromaStore
from graphlens.pipelines.summarize_v1 import (
    estimate_duration_seconds,
    seconds_to_hhmmss,
    make_summary_from_text,
    extract_key_topics,
)

# Your embeddings file might be "function based" (embed_texts)
# or "class based" (OpenAIEmbedder). We'll support BOTH.
try:
    # If you simplified openai_v1.py to functions
    from graphlens.embeddings.openai_v1 import embed_texts
    _EMBEDDINGS_MODE = "functions"
except Exception:
    # If you still have class-based embedder
    from graphlens.embeddings.openai_v1 import OpenAIEmbedder
    _EMBEDDINGS_MODE = "class"


def ingest_youtube_url_v1(
    url,
    languages=None,
    collection_name="graphlens_chunks",
    force_reindex=False,
    chunk_cfg=None,
    batch_size=64,
    store_raw_text=True,
    course_id=None,   # <--- NEW
):
    """
    Full ingestion pipeline (MVP):
      1) Fetch transcript from YouTube
      2) Chunk transcript (hybrid: time + chars)
      3) Clean chunk text (remove uh/um, [Applause], etc.)
      4) Create embeddings for cleaned chunks
      5) Store embeddings + text + metadata in Chroma

    Returns a summary dict that can be returned by an API endpoint later.
    """

    # Load .env so scripts also work (not just FastAPI).
    # If already loaded, this does nothing.
    load_dotenv()

    if languages is None:
        languages = ["en"]

    if chunk_cfg is None:
        chunk_cfg = {}

    # -------------------------
    # 1) Transcript
    # -------------------------
    doc = get_transcript_segments_v1(url, languages=languages)
    video_id = doc["video_id"]
    segment_count = doc["segment_count"]

    # -------------------------
    # 2) Chunking
    # -------------------------
    chunks = chunk_transcript_doc(
        doc,
        max_chars=chunk_cfg.get("max_chars", 2200),
        min_chars=chunk_cfg.get("min_chars", 1200),
        max_seconds=chunk_cfg.get("max_seconds", 135.0),
        overlap_chars=chunk_cfg.get("overlap_chars", 250),
    )

    # -------------------------
    # 3) Cleaning (for embeddings + retrieval)
    # -------------------------
    raw_texts = [c.text for c in chunks]
    clean_texts = [clean_for_embedding(t) for t in raw_texts]

    # -------------------------
    # 4) Embeddings
    # -------------------------
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    vectors = []

    if _EMBEDDINGS_MODE == "functions":
        # embed_texts(list[str]) -> list[list[float]]
        for i in range(0, len(clean_texts), batch_size):
            batch = clean_texts[i : i + batch_size]
            vectors.extend(embed_texts(batch))
    else:
        # OpenAIEmbedder().embed_texts(list[str])
        embedder = OpenAIEmbedder(model=embedding_model)
        for i in range(0, len(clean_texts), batch_size):
            batch = clean_texts[i : i + batch_size]
            vectors.extend(embedder.embed_texts(batch))

    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding mismatch: got {len(vectors)} vectors for {len(chunks)} chunks."
        )

    # -------------------------
    # 5) Store in Chroma
    # -------------------------
    store = ChromaStore(
        persist_path=os.getenv("CHROMA_PERSIST_PATH", "data/chroma"),
        collection_name=collection_name,
    )

    duration_seconds = estimate_duration_seconds(doc)
    duration_hhmmss = seconds_to_hhmmss(duration_seconds)

    # Summary: use first cleaned chunk as the "lead", because it's usually the intro/overview
    lead_text = clean_texts[0] if clean_texts else ""
    summary = make_summary_from_text(lead_text, max_sentences=3, max_chars=600)

    # Topics: use the first few chunks (or all if you want)
    topic_texts = clean_texts[:8]  # limit for speed + relevance
    key_topics = extract_key_topics(topic_texts, top_n=8)

    # Optional: delete old entries for this video_id (fresh rebuild)
    if force_reindex:
        try:
            store.delete_by_video_id(video_id)
        except Exception:
            # Some setups may not support delete well; upsert-by-id still works
            pass

    ids = [c.chunk_id for c in chunks]

    metadatas = []
    for c, raw in zip(chunks, raw_texts):
        md = dict(c.metadata)  # chunk metadata already has video_id, timestamps, etc.
        md["video_id"] = video_id  # ensure present for filtering
        md["source_url"] = url
        if course_id:
            md["course_id"] = course_id   # <--- NEW

        if store_raw_text:
            md["raw_text"] = raw

        metadatas.append(md)

    # Store cleaned text as the searchable "document"
    store.upsert(
        ids=ids,
        embeddings=vectors,
        documents=clean_texts,
        metadatas=metadatas,
    )

    # -------------------------
    # Summary (return)
    # -------------------------
    return {
    # scope for frontend
    "scope_type": "video",
    "scope_id": video_id,

    # UI-friendly info
    "summary": summary,
    "key_topics": key_topics,
    "estimated_duration_seconds": duration_seconds,
    "estimated_duration": duration_hhmmss,

    # debug/useful info
    "video_id": video_id,
    "source_url": url,
    "segments": segment_count,
    "chunks_indexed": len(chunks),
    "collection_name": collection_name,
    "embedding_model": embedding_model,
    "chroma_path": os.getenv("CHROMA_PERSIST_PATH", "data/chroma"),
}