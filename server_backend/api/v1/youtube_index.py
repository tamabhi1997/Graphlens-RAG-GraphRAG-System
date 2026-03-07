from fastapi import APIRouter, HTTPException
from server_backend.schemas.index import IndexYoutubeRequest, IndexYoutubeResponse
from graphlens.pipelines.ingest_youtube_v1 import ingest_youtube_url_v1

router = APIRouter()

@router.post("/youtube/index", response_model=IndexYoutubeResponse)
def youtube_index(req: IndexYoutubeRequest):
    try:
        out = ingest_youtube_url_v1(
            url=req.url,
            languages=req.languages,
            collection_name=req.collection_name,
            force_reindex=req.force_reindex,
            chunk_cfg=req.chunk_cfg,
            course_id=req.course_id,
        )
        return IndexYoutubeResponse(**out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))