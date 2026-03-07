from dotenv import load_dotenv
load_dotenv()

from graphlens.embeddings.openai_v1 import embed_query
from graphlens.vectorstores.chroma_v1 import ChromaStore


def query_v1(
    question,
    scope_type="video",     # "video" or "course"
    scope_id=None,          # video_id or course_id
    collection_name="graphlens_chunks",
    chroma_path="data/chroma",
):
    """
    Retrieval-only RAG query:
    - Frontend sends: question + scope
    - Backend controls: top_k, thresholds, refusal
    """

    # Backend knobs (keep here for now; later move to config/env)
    TOP_K = 4
    MIN_SIM_VIDEO = 0.28
    MIN_SIM_COURSE = 0.25

    # 1) Embed question
    qvec = embed_query(question)

    # 2) Build filter
    where = None
    if scope_type == "video" and scope_id:
        where = {"video_id": scope_id}
        min_sim = MIN_SIM_VIDEO
    elif scope_type == "course" and scope_id:
        where = {"course_id": scope_id}
        min_sim = MIN_SIM_COURSE
    else:
        # If no scope provided, search everything (you can choose to refuse instead)
        min_sim = MIN_SIM_COURSE

    # 3) Query vector DB
    store = ChromaStore(persist_path=chroma_path, collection_name=collection_name)
    res = store.query(query_embedding=qvec, top_k=TOP_K, where=where)

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    # 4) Build sources + similarity (similarity = 1 - cosine_distance)
    sources = []
    sims = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        sim = 1.0 - float(dist)
        sims.append(sim)

        sources.append(
            {
                "chunk_id": cid,
                "similarity": sim,
                "video_id": meta.get("video_id"),
                "course_id": meta.get("course_id"),
                "source_url": meta.get("source_url"),
                "start_seconds": meta.get("start_seconds"),
                "end_seconds": meta.get("end_seconds"),
                "text": doc,
            }
        )

    best_sim = sims[0] if sims else None

    # 5) Refusal gate
    if best_sim is None or best_sim < min_sim:
        return {
            "refused": True,
            "reason": "I don't have enough relevant evidence in the selected content to answer that.",
            "best_similarity": best_sim,
            "sources": sources[:3],
        }

    return {
        "refused": False,
        "best_similarity": best_sim,
        "answer": None,   # Gemini later
        "sources": sources,
    }