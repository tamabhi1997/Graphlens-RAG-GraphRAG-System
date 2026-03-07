# Central router aggregator for the API.
# Includes and mounts versioned routers (e.g., v1 transcript/ingest/query).
# Keeps main.py clean and endpoints organized.

from fastapi import APIRouter
from server_backend.api.v1.transcript import router as transcript_router
from server_backend.api.v1.youtube_index import router as youtube_index_router
from server_backend.api.v1.query import router as query_router

api_router = APIRouter()
api_router.include_router(transcript_router, prefix="/v1", tags=["transcript"])
api_router.include_router(youtube_index_router, prefix="/v1", tags=["youtube"])
api_router.include_router(query_router, prefix="/v1", tags=["query"])