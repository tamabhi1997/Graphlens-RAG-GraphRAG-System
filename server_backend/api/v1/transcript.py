# v1 transcript endpoints.
# Accepts a YouTube URL and returns transcript data (timestamps + text) in a normalized format.
# This is the first step of the RAG ingestion pipeline.

from fastapi import APIRouter, HTTPException
from server_backend.schemas.transcript import TranscriptRequest, TranscriptResponse

from graphlens.utils.youtube_v1 import (
    get_transcript_segments_v1,
    InvalidYouTubeUrl,
)

router = APIRouter()


@router.post("/transcript", response_model=TranscriptResponse)
def transcript(req: TranscriptRequest) -> TranscriptResponse:
    """
    v1: minimal transcript fetch (URL -> timestamped segments)
    """
    try:
        doc = get_transcript_segments_v1(req.url, languages=req.languages)
        return TranscriptResponse(**doc)

    except InvalidYouTubeUrl as e:
        # Client sent a bad URL
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Transcript not available / disabled / video unavailable etc.
        # For v1 we keep it simple: treat as "not found"
        raise HTTPException(status_code=404, detail=f"Transcript unavailable: {str(e)}")