from dotenv import load_dotenv
load_dotenv()

from graphlens.embeddings.openai_v1 import embed_query
from graphlens.vectorstores.chroma_v1 import ChromaStore
from graphlens.retrievers.reranker_v1 import rerank  # NEW


def query_v1(
    question,
    scope_type="video",     # "video", "course", or "document"
    scope_id=None,
    collection_name="graphlens_chunks",
    chroma_path="data/chroma",
):
    """
    Retrieval + Reranking RAG query:
    - Stage 1: ChromaDB cosine similarity → top 20 candidates (fast)
    - Stage 2: Cross-encoder reranker → top 4 (accurate)
    - Stage 3: Refusal gate on best reranked similarity
    """

    # Backend knobs
    RETRIEVAL_K = 20        # how many candidates to fetch from ChromaDB
    RERANK_TOP_N = 4        # how many to keep after reranking
    MIN_SIM_VIDEO = 0.28
    MIN_SIM_COURSE = 0.25
    MIN_SIM_DOCUMENT = 0.25

    # 1) Embed question
    qvec = embed_query(question)

    # 2) Build scope filter
    where = None
    if scope_type == "video" and scope_id:
        where = {"video_id": scope_id}
        min_sim = MIN_SIM_VIDEO
    elif scope_type == "course" and scope_id:
        where = {"course_id": scope_id}
        min_sim = MIN_SIM_COURSE
    elif scope_type == "document" and scope_id:
        where = {"doc_id": scope_id}
        min_sim = MIN_SIM_DOCUMENT
    else:
        min_sim = MIN_SIM_COURSE

    # 3) Stage 1 — vector retrieval (cast wide net: top 20)
    store = ChromaStore(persist_path=chroma_path, collection_name=collection_name)
    res = store.query(query_embedding=qvec, top_k=RETRIEVAL_K, where=where)

    ids   = res["ids"][0]
    docs  = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    # Build candidate sources list
    candidates = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        sim = 1.0 - float(dist)
        candidates.append({
            "chunk_id":      cid,
            "similarity":    sim,          # cosine similarity (stage 1 score)
            "video_id":      meta.get("video_id"),
            "course_id":     meta.get("course_id"),
            "doc_id":        meta.get("doc_id"),
            "source_url":    meta.get("source_url"),
            "start_seconds": meta.get("start_seconds"),
            "end_seconds":   meta.get("end_seconds"),
            "text":          doc,
        })

    if not candidates:
        return {
            "refused": True,
            "reason": "No content found for the selected scope.",
            "best_similarity": None,
            "sources": [],
        }

    # 4) Stage 2 — cross-encoder reranking (accurate, picks best 4 from 20)
    sources = rerank(question, candidates, top_n=RERANK_TOP_N)

    # 5) Refusal gate — use cosine similarity of best reranked chunk
    #    (rerank_score is a raw logit, not bounded 0-1, so we gate on
    #     the original cosine similarity of the top reranked chunk)
    best_sim = sources[0]["similarity"] if sources else None

    if best_sim is None or best_sim < min_sim:
        return {
            "refused": True,
            "reason": "I don't have enough relevant evidence in the selected content to answer that.",
            "best_similarity": best_sim,
            "rerank_score": sources[0]["rerank_score"] if sources else None,
            "sources": sources[:3],
        }

    return {
        "refused":      False,
        "best_similarity": best_sim,
        "rerank_score": sources[0]["rerank_score"],  # expose for debugging
        "answer":       None,   # Gemini later
        "sources":      sources,
    }