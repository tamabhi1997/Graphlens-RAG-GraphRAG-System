from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase


# ---------------------------------------------------------------------------
# Lazy singleton driver — created once, reused across all calls
# ---------------------------------------------------------------------------

_DRIVER = None

_RELATED_CONCEPT_NOISE = {
    "almost",
    "associated",
    "bunch",
    "done",
    "going",
    "said",
    "together",
    "thing",
    "things",
    "way",
    "kind",
    "type",
    "example",
    "part",
    "use",
    "used",
    "using",
}


def _is_display_concept(name: str, seed_terms: List[str]) -> bool:
    value = (name or "").strip().lower()
    if not value or value in seed_terms:
        return False
    if value in _RELATED_CONCEPT_NOISE:
        return False
    if len(value) < 4:
        return False
    if value.isdigit():
        return False
    return True

def _get_driver():
    global _DRIVER
    if _DRIVER is None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        _DRIVER = GraphDatabase.driver(uri, auth=(user, password))
    return _DRIVER


def close_driver():
    global _DRIVER
    if _DRIVER:
        _DRIVER.close()
        _DRIVER = None


# ---------------------------------------------------------------------------
# Schema setup — run once at startup to create indexes
# ---------------------------------------------------------------------------

def create_indexes() -> None:
    """
    Create Neo4j indexes for fast lookups.
    Safe to call multiple times — uses CREATE IF NOT EXISTS.
    """
    driver = _get_driver()
    with driver.session() as session:
        session.run(
            "CREATE INDEX concept_name IF NOT EXISTS "
            "FOR (c:Concept) ON (c.name)"
        )
        session.run(
            "CREATE INDEX chunk_id IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.chunk_id)"
        )
    print("[neo4j] Indexes created/verified")


# ---------------------------------------------------------------------------
# Chunk operations
# ---------------------------------------------------------------------------

def chunk_exists(chunk_id: str) -> bool:
    """Check if a chunk has already been processed into the graph."""
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(
            "MATCH (c:Chunk {chunk_id: $chunk_id}) RETURN c LIMIT 1",
            chunk_id=chunk_id,
        )
        return result.single() is not None


def store_chunk_node(
    chunk_id: str,
    video_id: str,
    source_url: str,
    start_seconds: float,
    end_seconds: float,
    scope_type: str = "video",
    course_id: str = "",
    doc_id: str = "",
) -> None:
    """Create a Chunk node in Neo4j (mirrors ChromaDB metadata)."""
    driver = _get_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (c:Chunk {chunk_id: $chunk_id})
            SET c.video_id = $video_id,
                c.course_id = $course_id,
                c.doc_id = $doc_id,
                c.source_url = $source_url,
                c.start_seconds = $start_seconds,
                c.end_seconds = $end_seconds,
                c.scope_type = $scope_type
            """,
            chunk_id=chunk_id,
            video_id=video_id,
            course_id=course_id,
            doc_id=doc_id,
            source_url=source_url,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            scope_type=scope_type,
        )


# These are the THREE updated functions to replace in neo4j_client.py
# Find each function by name and replace the full function body.
# Add this import at the top of neo4j_client.py:
#   from graphlens.graphrag.entity_normaliser import normalise_concept
 
 
def store_concept(name: str, description: str = "") -> None:
    """Create or update a Concept node — normalises name before storing."""
    from graphlens.graphrag.entity_normaliser import normalise_concept
 
    canonical = normalise_concept(name)
    if not canonical:
        return  # skip noise concepts
 
    driver = _get_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (c:Concept {name: $name})
            ON CREATE SET c.description = $description,
                          c.mention_count = 1
            ON MATCH SET  c.mention_count = c.mention_count + 1
            """,
            name=canonical,
            description=description,
        )
 
 
def store_mentions(chunk_id: str, concept_name: str) -> None:
    """Create MENTIONS edge — normalises concept name before storing."""
    from graphlens.graphrag.entity_normaliser import normalise_concept
 
    canonical = normalise_concept(concept_name)
    if not canonical:
        return  # skip noise concepts
 
    driver = _get_driver()
    with driver.session() as session:
        session.run(
            """
            MATCH (chunk:Chunk {chunk_id: $chunk_id})
            MERGE (concept:Concept {name: $concept_name})
            MERGE (chunk)-[:MENTIONS]->(concept)
            """,
            chunk_id=chunk_id,
            concept_name=canonical,
        )
 
 
def store_relationship(
    from_concept: str,
    rel_type: str,
    to_concept: str,
) -> None:
    """Create relationship between two concepts — normalises both names."""
    from graphlens.graphrag.entity_normaliser import normalise_concept
 
    from_canonical = normalise_concept(from_concept)
    to_canonical   = normalise_concept(to_concept)
 
    if not from_canonical or not to_canonical:
        return  # skip if either is noise
 
    if from_canonical == to_canonical:
        return  # skip self-relationships
 
    valid_types = {"USES", "TRAINS", "REQUIRES", "RELATES_TO", "PART_OF", "OPTIMIZES"}
    if rel_type not in valid_types:
        rel_type = "RELATES_TO"
 
    driver = _get_driver()
    with driver.session() as session:
        query = f"""
            MERGE (a:Concept {{name: $from_name}})
            MERGE (b:Concept {{name: $to_name}})
            MERGE (a)-[:{rel_type}]->(b)
        """
        session.run(
            query,
            from_name=from_canonical,
            to_name=to_canonical,
        )


# ---------------------------------------------------------------------------
# Graph expansion — the core GraphRAG query
# ---------------------------------------------------------------------------

def get_related_chunk_ids(
    chunk_ids: List[str],
    limit: int = 10,
) -> List[str]:
    """
    1-hop graph expansion:
    Given a list of chunk_ids (from ChromaDB retrieval),
    find other chunks that share concepts with them.

    This is the core GraphRAG query — it finds chunks that
    are conceptually related even if not retrieved by cosine similarity.

    Returns list of related chunk_ids NOT already in the input list.
    """
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (seed:Chunk)-[:MENTIONS]->(concept:Concept)
                  <-[:MENTIONS]-(related:Chunk)
            WHERE seed.chunk_id IN $chunk_ids
              AND NOT related.chunk_id IN $chunk_ids
            RETURN DISTINCT related.chunk_id AS chunk_id,
                   count(concept) AS shared_concepts
            ORDER BY shared_concepts DESC
            LIMIT $limit
            """,
            chunk_ids=chunk_ids,
            limit=limit,
        )
        return [record["chunk_id"] for record in result]


def get_concept_graph(
    concept_name: str,
    scope_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch the concept graph for visualization in the frontend.
    Returns nodes and edges centered on concept_name.
    """
    from graphlens.graphrag.entity_normaliser import normalise_concept

    raw_name = concept_name.lower().strip()
    canonical_name = normalise_concept(raw_name) or raw_name
    seed_terms = [raw_name, canonical_name]
    for term in (raw_name, canonical_name):
        for token in re.findall(r"[a-z0-9]+", term):
            if len(token) >= 3 and token not in {"and", "the", "for", "with"}:
                seed_terms.append(token)
                if token.endswith("s") and len(token) > 4:
                    seed_terms.append(token[:-1])
    seed_terms = list(dict.fromkeys(seed_terms))

    driver = _get_driver()
    with driver.session() as session:
        chunks_result = session.run(
            """
            MATCH (chunk:Chunk)-[:MENTIONS]->(seed:Concept)
            WHERE seed.name IN $seed_terms
            WITH DISTINCT chunk
            WHERE $scope_id IS NULL
               OR $scope_id = ""
               OR chunk.video_id = $scope_id
               OR chunk.course_id = $scope_id
               OR chunk.doc_id = $scope_id
            RETURN collect(DISTINCT {
                       chunk_id: chunk.chunk_id,
                       start_seconds: chunk.start_seconds,
                       source_url: chunk.source_url,
                       video_id: chunk.video_id,
                       course_id: chunk.course_id,
                       doc_id: chunk.doc_id
                   }) AS chunks
            """,
            seed_terms=seed_terms,
            scope_id=scope_id,
        )
        chunks_record = chunks_result.single()
        chunks = chunks_record["chunks"] if chunks_record else []
        if not chunks:
            return {"concept": raw_name, "nodes": [], "edges": [], "chunks": []}

        direct_result = session.run(
            """
            MATCH (seed:Concept)
            WHERE seed.name IN $seed_terms
            OPTIONAL MATCH (seed)-[r]-(neighbor:Concept)
            WITH collect(DISTINCT CASE
                WHEN neighbor IS NULL OR neighbor.name IN $seed_terms THEN NULL
                ELSE {name: neighbor.name, rel: type(r)}
            END) AS neighbors
            RETURN neighbors
            """,
            seed_terms=seed_terms,
        )
        direct_record = direct_result.single()
        neighbors = [
            item
            for item in (direct_record["neighbors"] if direct_record else [])
            if item and _is_display_concept(item.get("name", ""), seed_terms)
        ]

        if not neighbors:
            related_result = session.run(
                """
                MATCH (chunk:Chunk)-[:MENTIONS]->(seed:Concept)
                WHERE seed.name IN $seed_terms
                WITH DISTINCT chunk
                WHERE $scope_id IS NULL
                   OR $scope_id = ""
                   OR chunk.video_id = $scope_id
                   OR chunk.course_id = $scope_id
                   OR chunk.doc_id = $scope_id
                MATCH (chunk)-[:MENTIONS]->(related:Concept)
                WHERE NOT related.name IN $seed_terms
                RETURN related.name AS name, count(DISTINCT chunk) AS shared_chunks
                ORDER BY shared_chunks DESC, name ASC
                LIMIT 12
                """,
                seed_terms=seed_terms,
                scope_id=scope_id,
            )
            neighbors = [
                {"name": record["name"], "rel": "RELATES_TO"}
                for record in related_result
                if _is_display_concept(record["name"], seed_terms)
            ]

        center_name = raw_name
        nodes = [{"name": center_name, "type": "center"}]
        edges = []

        seen_nodes = {center_name}
        for neighbor in neighbors:
            neighbor_name = neighbor["name"]
            if neighbor_name and neighbor_name not in seen_nodes:
                seen_nodes.add(neighbor_name)
                nodes.append({"name": neighbor_name, "type": "neighbor"})
                edges.append({
                    "from": center_name,
                    "to": neighbor_name,
                    "type": neighbor["rel"],
                })

        return {
            "concept": center_name,
            "nodes": nodes,
            "edges": edges,
            "chunks": chunks,
        }


def ping() -> bool:
    """Test Neo4j connection. Returns True if connected."""
    try:
        driver = _get_driver()
        with driver.session() as session:
            session.run("RETURN 1")
        return True
    except Exception as e:
        print(f"[neo4j] Connection failed: {e}")
        return False
