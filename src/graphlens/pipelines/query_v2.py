from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from graphlens.embeddings.openai_v1 import embed_query
from graphlens.vectorstores.chroma_v1 import ChromaStore
from graphlens.retrievers.reranker_v1 import rerank
from graphlens.graphrag.neo4j_client import get_related_chunk_ids
from graphlens.graphrag.graph_builder import build_graph_for_chunks
from graphlens.pipelines.generator_v1 import generate_answer
from graphlens.pipelines.reliability_model import ReliabilityModel


_RELIABILITY_MODEL = ReliabilityModel.load()

# _RELIABILITY_MODEL = None


def _get_reliability_model() -> ReliabilityModel:
    return _RELIABILITY_MODEL


def query_v2(
    question: str,
    scope_type: str = "video",
    scope_id: str = None,
    collection_name: str = "graphlens_chunks",
    chroma_path: str = "data/chroma",
    use_graph: bool = True,
) -> dict:
    """
    Full GraphRAG query pipeline:
    Stage 1  — Vector retrieval (ChromaDB top-20)
    Stage 2  — Cross-encoder reranking (top-4)
    Stage 3  — Graph entity extraction (lazy, cached)
    Stage 4  — 1-hop Neo4j graph expansion
    Stage 5  — Fetch expanded chunks from ChromaDB
    Stage 6  — Final rerank on merged set
    Stage 7  — Refusal gate
    Stage 8  — Gemini grounded answer generation
    Stage 9  — Logistic regression confidence score
    """

    RETRIEVAL_K      = 20
    RERANK_TOP_N     = 4
    GRAPH_EXPAND_LIM = 6
    FINAL_TOP_N      = 4
    MIN_SIM_VIDEO    = 0.28
    MIN_SIM_COURSE   = 0.25
    MIN_SIM_DOCUMENT = 0.25

    # 1) Embed
    qvec = embed_query(question)

    # 2) Scope filter
    where = None
    if scope_type == "video" and scope_id:
        where = {"video_id": scope_id}; min_sim = MIN_SIM_VIDEO
    elif scope_type == "course" and scope_id:
        where = {"course_id": scope_id}; min_sim = MIN_SIM_COURSE
    elif scope_type == "document" and scope_id:
        where = {"doc_id": scope_id}; min_sim = MIN_SIM_DOCUMENT
    else:
        min_sim = MIN_SIM_COURSE

    # 3) Vector retrieval
    store = ChromaStore(persist_path=chroma_path, collection_name=collection_name)
    res   = store.query(query_embedding=qvec, top_k=RETRIEVAL_K, where=where)

    candidates = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        candidates.append({
            "chunk_id": cid, "similarity": 1.0 - float(dist),
            "video_id": meta.get("video_id"), "course_id": meta.get("course_id"),
            "doc_id": meta.get("doc_id"), "source_url": meta.get("source_url"),
            "start_seconds": meta.get("start_seconds"), "end_seconds": meta.get("end_seconds"),
            "text": doc, "expanded": False,
        })

    if not candidates:
        return {"refused": True, "reason": "No content found for the selected scope.",
                "best_similarity": None, "sources": [],
                "graph_expansion": {"expanded_chunks": 0, "method": "none"}, "confidence": 0.0}

    # 4) Rerank top-4
    top_chunks = rerank(question, candidates, top_n=RERANK_TOP_N)
    graph_expansion_info = {"expanded_chunks": 0, "method": "none"}

    if use_graph:
        # 5) Graph extraction (lazy, cached)
        build_graph_for_chunks(top_chunks)

        # 6) 1-hop expansion
        top_ids      = [c["chunk_id"] for c in top_chunks]
        expanded_ids = get_related_chunk_ids(top_ids, limit=GRAPH_EXPAND_LIM)

        if expanded_ids:
            exp_res  = store.query(query_embedding=qvec, top_k=len(expanded_ids) + 5, where=where)
            seen     = {c["chunk_id"] for c in top_chunks}
            for eid, edoc, emeta, edist in zip(
                exp_res["ids"][0], exp_res["documents"][0],
                exp_res["metadatas"][0], exp_res["distances"][0]
            ):
                if eid in expanded_ids and eid not in seen:
                    top_chunks.append({
                        "chunk_id": eid, "similarity": 1.0 - float(edist),
                        "video_id": emeta.get("video_id"), "course_id": emeta.get("course_id"),
                        "doc_id": emeta.get("doc_id"), "source_url": emeta.get("source_url"),
                        "start_seconds": emeta.get("start_seconds"), "end_seconds": emeta.get("end_seconds"),
                        "text": edoc, "expanded": True,
                    })
                    seen.add(eid)
            graph_expansion_info = {"expanded_chunks": len(expanded_ids), "method": "1-hop neo4j"}

        # 7) Final rerank on merged set
        top_chunks = rerank(question, top_chunks, top_n=FINAL_TOP_N)

    # 8) Refusal gate
    best_sim = top_chunks[0]["similarity"] if top_chunks else None
    if best_sim is None or best_sim < min_sim:
        return {
            "refused": True,
            "reason": "I don't have enough relevant evidence in the selected content to answer that.",
            "best_similarity": best_sim, "sources": top_chunks[:3],
            "graph_expansion": graph_expansion_info, "confidence": 0.0,
        }

    # 9) Generate answer
    generated = generate_answer(question, top_chunks)

    response = {
        "refused": generated["refused"], "reason": None,
        "best_similarity": best_sim, "rerank_score": top_chunks[0].get("rerank_score"),
        "answer": generated["answer"], "citations": generated["citations"],
        "model": generated["model"], "sources": top_chunks,
        "graph_expansion": graph_expansion_info,
    }

    # 10) Reliability score
    # 10) Reliability score
    try:
        conf = _get_reliability_model().predict(response)
        print(f"[reliability] confidence={conf}, refused={response.get('refused')}, sources={len(response.get('sources',[]))}")
        response["confidence"] = conf
    except Exception as e:
        print(f"[reliability] FAILED: {e}")
        response["confidence"] = None

    return response