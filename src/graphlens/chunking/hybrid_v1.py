#v2 later = tokens + punctuation-aware boundary selection.
#currently this is 
# stop when duration >= max_seconds OR text >= max_chars
    # cut only at segment boundaries
    # keep a small segment overlap so ideas aren’t broken

# max_chars: int = 2200,
#     min_chars: int = 800,
#     max_seconds: float = 75.0,
#     overlap_chars: int = 250,

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    start_seconds: float
    end_seconds: float
    metadata: Dict[str, Any]


def chunk_transcript_doc(
    doc: Dict[str, Any],
    *,
    max_chars: int = 2200,
    min_chars: int = 800,
    max_seconds: float = 135.0,
    overlap_chars: int = 250,
) -> List[Chunk]:
    """
    Hybrid chunker:
    - merges transcript segments into chunks
    - cuts only at segment boundaries
    - stops when time OR chars exceed threshold
    - carries a small overlap (tail segments) into next chunk

    doc is your normalized transcript dict with:
      doc["segments"] = [{start_seconds, end_seconds, text}, ...]
    """
    segments = doc.get("segments", [])
    video_id = doc.get("video_id", "unknown")
    source_url = doc.get("source_url", "")

    chunks: List[Chunk] = []
    cur_segs: List[Dict[str, Any]] = [] #cur_segs is list of dictionar with key being str and values can be anything heree
    cur_text_parts: List[str] = []
    cur_start = None
    cur_end = None

    def cur_text() -> str:
        #guys reading this ->str doesnt do anything but its for readability that means this function will return something and here its a str---
        return " ".join(cur_text_parts).strip()

    def cur_len() -> int:
        return len(cur_text())

    def cur_dur() -> float:
        if cur_start is None or cur_end is None:
            return 0.0
        return float(cur_end) - float(cur_start)

    def tail_overlap_segments() -> List[Dict[str, Any]]:
        # Keep last segments totaling ~overlap_chars (approx), so timestamps remain valid.
        kept: List[Dict[str, Any]] = []
        total = 0
        for s in reversed(cur_segs):
            t = s.get("text", "")
            kept.append(s)
            total += len(t) + 1
            if total >= overlap_chars:
                break
        kept.reverse()
        return kept

    chunk_index = 0

    def flush_chunk():
        nonlocal chunk_index, cur_segs, cur_text_parts, cur_start, cur_end

        text = cur_text()
        if not text:
            return

        cid = f"{video_id}:{chunk_index:04d}"
        meta = {
            "video_id": video_id,
            "source_url": source_url,
            "chunk_index": chunk_index,
            "start_seconds": float(cur_start or 0.0),
            "end_seconds": float(cur_end or 0.0),
        }
        chunks.append(
            Chunk(
                chunk_id=cid,
                text=text,
                start_seconds=float(cur_start or 0.0),
                end_seconds=float(cur_end or 0.0),
                metadata=meta,
            )
        )
        chunk_index += 1

        # Prepare next chunk with overlap
        overlap = tail_overlap_segments()
        cur_segs = overlap
        cur_text_parts = [s.get("text", "") for s in overlap]
        cur_start = overlap[0]["start_seconds"] if overlap else None
        cur_end = overlap[-1]["end_seconds"] if overlap else None

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        s0 = float(seg.get("start_seconds") or 0.0)
        s1 = float(seg.get("end_seconds") or s0)

        if cur_start is None:
            cur_start = s0
        cur_end = s1

        cur_segs.append({"start_seconds": s0, "end_seconds": s1, "text": text})
        cur_text_parts.append(text)

        # If we exceed limits AND we've reached minimum size, cut.
        if (cur_dur() >= max_seconds or cur_len() >= max_chars) and cur_len() >= min_chars:
            flush_chunk()

    # flush leftovers
    if cur_len() > 0:
        # avoid producing a tiny last chunk if it’s mostly overlap:
        flush_chunk()

    return chunks