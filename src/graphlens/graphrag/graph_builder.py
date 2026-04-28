from __future__ import annotations

from typing import Any, Dict, List

from graphlens.graphrag.neo4j_client import (
    chunk_exists,
    store_chunk_node,
    store_concept,
    store_mentions,
    store_relationship,
)
from graphlens.graphrag.entity_extractor import extract_entities


def build_graph_for_chunk(
    chunk_id: str,
    chunk_text: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Process a single chunk into the knowledge graph.

    Steps:
      1. Check if chunk already in Neo4j (cache hit → skip extraction)
      2. Store Chunk node
      3. Extract entities (spaCy → LLM fallback)
      4. Store Concept nodes
      5. Store MENTIONS edges (Chunk → Concept)
      6. Store Concept → Concept relationships

    Args:
        chunk_id:   Same chunk_id as in ChromaDB.
        chunk_text: Cleaned chunk text.
        metadata:   Chunk metadata dict from ChromaDB.

    Returns:
        Dict with extraction results + cache hit status.
    """

    # --- Cache check ---
    if chunk_exists(chunk_id):
        return {
            "chunk_id": chunk_id,
            "cache_hit": True,
            "concepts_added": 0,
            "relationships_added": 0,
            "method": "cached",
        }

    # --- Store Chunk node ---
    store_chunk_node(
        chunk_id=chunk_id,
        video_id=metadata.get("video_id", ""),
        source_url=metadata.get("source_url", ""),
        start_seconds=float(metadata.get("start_seconds", 0.0)),
        end_seconds=float(metadata.get("end_seconds", 0.0)),
        scope_type=metadata.get("scope_type", "video"),
        course_id=metadata.get("course_id", ""),
        doc_id=metadata.get("doc_id", ""),
    )

    # --- Extract entities ---
    extracted = extract_entities(chunk_text)
    concepts = extracted.get("concepts", [])
    relationships = extracted.get("relationships", [])
    method = extracted.get("method", "unknown")

    # --- Store Concept nodes + MENTIONS edges ---
    for concept in concepts:
        name = concept.get("name", "").strip()
        description = concept.get("description", "")
        if not name:
            continue
        store_concept(name, description)
        store_mentions(chunk_id, name)

    # --- Store Concept → Concept relationships ---
    for rel in relationships:
        from_c = rel.get("from", "").strip()
        rel_type = rel.get("type", "RELATES_TO")
        to_c = rel.get("to", "").strip()
        if from_c and to_c:
            store_relationship(from_c, rel_type, to_c)

    return {
        "chunk_id": chunk_id,
        "cache_hit": False,
        "concepts_added": len(concepts),
        "relationships_added": len(relationships),
        "method": method,
    }


def build_graph_for_chunks(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Process multiple chunks into the graph.
    Used at query time for the top retrieved chunks.

    Each chunk dict must have: chunk_id, text, metadata.
    """
    results = []
    for chunk in chunks:
        result = build_graph_for_chunk(
            chunk_id=chunk["chunk_id"],
            chunk_text=chunk["text"],
            metadata=chunk,
        )
        results.append(result)
        if not result["cache_hit"]:
            print(
                f"[graph_builder] {chunk['chunk_id']} → "
                f"{result['concepts_added']} concepts, "
                f"{result['relationships_added']} rels "
                f"({result['method']})"
            )
    return results
