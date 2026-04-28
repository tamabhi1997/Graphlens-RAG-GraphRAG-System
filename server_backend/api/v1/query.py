from fastapi import APIRouter, HTTPException
from server_backend.schemas.query import QueryRequest, QueryResponse
from graphlens.pipelines.query_v2 import query_v2

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        out = query_v2(
            question=req.question,
            scope_type=req.scope_type,
            scope_id=req.scope_id,
            collection_name=req.collection_name,
            use_graph=req.use_graph,
        )
        return QueryResponse(**out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
