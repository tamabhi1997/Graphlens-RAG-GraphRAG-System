# v1 query endpoints.
# Accepts a user question, retrieves relevant chunks from the vector DB, and returns an answer.
# Later can be upgraded to GraphRAG by expanding retrieval using graph relations.

from fastapi import APIRouter, HTTPException
from server_backend.schemas.query import QueryRequest, QueryResponse
from graphlens.pipelines.query_v1 import query_v1

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        out = query_v1(
            question=req.question,
            scope_type=req.scope_type,
            scope_id=req.scope_id,
            collection_name=req.collection_name,
        )
        return QueryResponse(**out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))