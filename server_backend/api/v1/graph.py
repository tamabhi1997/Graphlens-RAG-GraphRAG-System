from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from graphlens.graphrag.neo4j_client import get_concept_graph, ping

router = APIRouter()


@router.get("/graph/concept")
def get_concept_graph_endpoint(
    concept: str = Query(..., description="Concept name to center the graph on"),
    scope_id: Optional[str] = Query(default=None, description="Optional scope filter"),
):
    """
    Fetch knowledge graph centered on a concept for frontend visualization.

    Usage:
        GET /api/v1/graph/concept?concept=gradient+descent&scope_id=abc123

    Returns nodes and edges for the graph UI tab.
    Each node has a name and type (center / neighbor).
    Each edge has from, to, and relationship type.
    Chunks include chunk_id, start_seconds, source_url for timestamp links.
    """
    if not concept or len(concept.strip()) < 2:
        raise HTTPException(status_code=400, detail="concept must be at least 2 characters.")

    try:
        graph = get_concept_graph(
            concept_name=concept.strip().lower(),
            scope_id=scope_id,
        )
        return {
            "concept": concept.strip().lower(),
            "nodes": graph["nodes"],
            "edges": graph["edges"],
            "chunks": graph["chunks"],
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/health")
def graph_health():
    """Check Neo4j connection status."""
    connected = ping()
    if not connected:
        raise HTTPException(
            status_code=503,
            detail="Neo4j is not reachable. Check NEO4J_URI and NEO4J_PASSWORD in .env"
        )
    return {"neo4j": "connected"}