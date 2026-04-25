from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from graphlens.embeddings.openai_v1 import embed_query
from graphlens.vectorstores.chroma_v1 import ChromaStore
from graphlens.retrievers.reranker_v1 import rerank
from graphlens.graphrag.neo4j_client import get_related_chunk_ids
from graphlens.graphrag.graph_builder import build_graph_for_chunks
from graphlens.pipelines.generator_v1 import generate_answer



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

    Stage 1 — Vector retrieval:  ChromaDB top 20 (fast, approximate)
    Stage 2 — Reranking:         Cross-encoder top 4 (accurate)
    Stage 3 — Graph extraction:  Extract entities from top 4 (lazy, cached)
    Stage 4 — Graph expansion:   1-hop Neo4j neighbors
    Stage 5 — Fetch expanded:    Pull expanded chunks from ChromaDB
    Stage 6 — Final rerank:      Rerank merged set
    Stage 7 — Refusal gate:      Refuse if best similarity too low

    Args:
        use_graph: Set False to skip GraphRAG and run plain RAG only.
                   Useful for ablation testing.
    """

    # Backend knobs
    RETRIEVAL_K = 20
    RERANK_TOP_N = 4
    GRAPH_EXPAND_LIMIT = 6       # max extra chunks from graph expansion
    FINAL_TOP_N = 4              # final chunks after merging + reranking
    MIN_SIM_VIDEO = 0.28
    MIN_SIM_COURSE = 0.25
    MIN_SIM_DOCUMENT = 0.25

    # -------------------------
    # 1) Embed question
    # -------------------------
    qvec = embed_query(question)

    # -------------------------
    # 2) Build scope filter
    # -------------------------
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

    # -------------------------
    # 3) Stage 1: vector retrieval (top 20)
    # -------------------------
    store = ChromaStore(persist_path=chroma_path, collection_name=collection_name)
    res = store.query(query_embedding=qvec, top_k=RETRIEVAL_K, where=where)

    ids   = res["ids"][0]
    docs  = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    candidates = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        sim = 1.0 - float(dist)
        candidates.append({
            "chunk_id":      cid,
            "similarity":    sim,
            "video_id":      meta.get("video_id"),
            "course_id":     meta.get("course_id"),
            "doc_id":        meta.get("doc_id"),
            "source_url":    meta.get("source_url"),
            "start_seconds": meta.get("start_seconds"),
            "end_seconds":   meta.get("end_seconds"),
            "text":          doc,
            "expanded":      False,   # marks vector-retrieved chunks
        })

    if not candidates:
        return {
            "refused": True,
            "reason": "No content found for the selected scope.",
            "best_similarity": None,
            "sources": [],
            "graph_expansion": {"expanded_chunks": 0, "method": "none"},
        }

    # -------------------------
    # 4) Stage 2: rerank top 4
    # -------------------------
    top_chunks = rerank(question, candidates, top_n=RERANK_TOP_N)

    graph_expansion_info = {"expanded_chunks": 0, "method": "none"}

    if use_graph:
        # -------------------------
        # 5) Stage 3: graph extraction (lazy, cached)
        # -------------------------
        build_graph_for_chunks(top_chunks)

        # -------------------------
        # 6) Stage 4: 1-hop graph expansion
        # -------------------------
        top_chunk_ids = [c["chunk_id"] for c in top_chunks]
        expanded_ids = get_related_chunk_ids(top_chunk_ids, limit=GRAPH_EXPAND_LIMIT)

        if expanded_ids:
            # -------------------------
            # 7) Stage 5: fetch expanded chunks from ChromaDB
            # -------------------------
            expanded_res = store.query(
                query_embedding=qvec,
                top_k=len(expanded_ids) + 5,
                where=where,
            )

            expanded_docs  = expanded_res["documents"][0]
            expanded_metas = expanded_res["metadatas"][0]
            expanded_dists = expanded_res["distances"][0]
            expanded_ids_returned = expanded_res["ids"][0]

            # Filter to only the graph-expanded chunk ids
            already_seen = {c["chunk_id"] for c in top_chunks}
            for eid, edoc, emeta, edist in zip(
                expanded_ids_returned, expanded_docs,
                expanded_metas, expanded_dists
            ):
                if eid in expanded_ids and eid not in already_seen:
                    top_chunks.append({
                        "chunk_id":      eid,
                        "similarity":    1.0 - float(edist),
                        "video_id":      emeta.get("video_id"),
                        "course_id":     emeta.get("course_id"),
                        "doc_id":        emeta.get("doc_id"),
                        "source_url":    emeta.get("source_url"),
                        "start_seconds": emeta.get("start_seconds"),
                        "end_seconds":   emeta.get("end_seconds"),
                        "text":          edoc,
                        "expanded":      True,  # marks graph-expanded chunks
                    })
                    already_seen.add(eid)

            graph_expansion_info = {
                "expanded_chunks": len(expanded_ids),
                "method": "1-hop neo4j",
            }

        # -------------------------
        # 8) Stage 6: final rerank on merged set
        # -------------------------
        top_chunks = rerank(question, top_chunks, top_n=FINAL_TOP_N)

    # -------------------------
    # 9) Stage 7: refusal gate
    # -------------------------
    best_sim = top_chunks[0]["similarity"] if top_chunks else None

    if best_sim is None or best_sim < min_sim:
        return {
            "refused": True,
            "reason": "I don't have enough relevant evidence in the selected content to answer that.",
            "best_similarity": best_sim,
            "sources": top_chunks[:3],
            "graph_expansion": graph_expansion_info,
        }

    # -------------------------
    # 10) Stage 8: generate answer
    # -------------------------
    generated = generate_answer(question, top_chunks)

    return {
        "refused":          generated["refused"],
        "reason":           None,
        "best_similarity":  best_sim,
        "rerank_score":     top_chunks[0].get("rerank_score"),
        "answer":           generated["answer"],
        "citations":        generated["citations"],
        "model":            generated["model"],
        "sources":          top_chunks,
        "graph_expansion":  graph_expansion_info,
    }