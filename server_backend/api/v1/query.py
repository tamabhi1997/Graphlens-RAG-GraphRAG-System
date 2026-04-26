# v1 query endpoints.
# Accepts a user question, retrieves relevant chunks from the vector DB, and returns an answer.
# Later can be upgraded to GraphRAG by expanding retrieval using graph relations.

from fastapi import APIRouter, HTTPException
from server_backend.schemas.query import QueryRequest, QueryResponse
from graphlens.pipelines.query_v1 import query_v1
# from graphlens.pipelines.query_v1 import query_v1
from graphlens.pipelines.query_v2 import query_v2
# from graphlens.pipelines.query_v2 import query_v2 as query_v1


router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        if req.use_graph:
            try:
                out = query_v2(
                    question=req.question,
                    scope_type=req.scope_type,
                    scope_id=req.scope_id,
                    collection_name=req.collection_name,
                    use_graph=True,
                )
            except Exception as graph_error:
                out = query_v1(
                    question=req.question,
                    scope_type=req.scope_type,
                    scope_id=req.scope_id,
                    collection_name=req.collection_name,
                )
                out["graph_expansion"] = {
                    "expanded_chunks": 0,
                    "method": "fallback_plain_rag",
                    "error": str(graph_error),
                }
        else:
            out = query_v1(
                question=req.question,
                scope_type=req.scope_type,
                scope_id=req.scope_id,
                collection_name=req.collection_name,
            )
        return QueryResponse(**out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
