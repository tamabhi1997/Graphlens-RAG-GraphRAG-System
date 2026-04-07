from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

import tiktoken

# ---------------------------------------------------------------------------
# Chunk dataclass (same structure as hybrid_v1 — downstream code unchanged)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    start_seconds: float
    end_seconds: float
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Tokenizer (lazy singleton — loaded once, reused forever)
# ---------------------------------------------------------------------------

_ENCODER = None


def _get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        # cl100k_base = same tokenizer used by text-embedding-3-small
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    """Return the number of cl100k tokens in text."""
    return len(_get_encoder().encode(text))


# ---------------------------------------------------------------------------
# Step 1A — hard boundary detector
# Detects bracket stage directions BEFORE the cleaner removes them.
# Examples: [Applause], [Music], [Laughter], [Inaudible]
# ---------------------------------------------------------------------------

_HARD_BOUNDARY_RE = re.compile(r"\[(?:[^\[\]]{1,40})\]")


def _has_hard_boundary(text: str) -> bool:
    """Return True if this segment contains a stage direction like [Applause]."""
    return bool(_HARD_BOUNDARY_RE.search(text or ""))


# ---------------------------------------------------------------------------
# Main chunker — v2
# Changes vs v1:
#   • Sizes chunks by TOKEN count (not character count)  — Step 1B
#   • Flushes on hard boundaries like [Applause]         — Step 1A
#   • Overlap is also measured in tokens (not chars)
#   • Stores token_count in chunk metadata for debugging
#   • Uses running token counter (O(n), not re-tokenize whole chunk each step)
#   • Prevents overlap-only duplicate chunk at the end
# ---------------------------------------------------------------------------


def chunk_transcript_doc_v2(
    doc: Dict[str, Any],
    *,
    max_tokens: int = 400,  # ~300 words; well under 8192 model limit
    min_tokens: int = 100,  # don't flush tiny fragments (except hard boundaries)
    overlap_tokens: int = 40,  # ~30 words of context carried into next chunk
    max_seconds: float = 135.0,  # secondary guard for very slow speech
) -> List[Chunk]:
    """
    Hybrid chunker v2 — token-aware with hard topic boundaries.

    Merges transcript segments into chunks and cuts when:
      1. A bracket stage direction is detected (hard boundary, Step 1A)
      2. Token count >= max_tokens  (Step 1B — replaces char counting)
      3. Duration >= max_seconds    (secondary guard for slow speakers)

    Cuts only at segment boundaries (never mid-sentence).
    Carries overlap_tokens of context into the next chunk (but NOT across hard boundaries).

    Args:
        doc:            Transcript dict from get_transcript_segments_v1().
                        Must contain doc["segments"] = [{start_seconds,
                        end_seconds, text}, ...]
        max_tokens:     Flush chunk when accumulated tokens reach this limit.
        min_tokens:     Never flush below this — avoids tiny orphan chunks
                        for normal size/time gates (hard boundaries still flush).
        overlap_tokens: Tokens of tail context carried into the next chunk.
        max_seconds:    Flush if a single chunk spans more than this many
                        seconds (catches slow-speech edge cases).

    Returns:
        List of Chunk dataclass instances.
    """

    segments = doc.get("segments", [])
    video_id = doc.get("video_id", "unknown")
    source_url = doc.get("source_url", "")

    chunks: List[Chunk] = []

    # Running state
    cur_segs: List[Dict[str, Any]] = []
    cur_text_parts: List[str] = []
    cur_start = None
    cur_end = None
    chunk_index = 0
    cur_token_count = 0
    has_new_content = False  # guards against overlap-only duplicate flush at end

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def cur_text() -> str:
        return " ".join(cur_text_parts).strip()

    def cur_tok() -> int:
        return int(cur_token_count)

    def cur_dur() -> float:
        if cur_start is None or cur_end is None:
            return 0.0
        return float(cur_end) - float(cur_start)

    def reset_state() -> None:
        nonlocal cur_segs, cur_text_parts, cur_start, cur_end, cur_token_count, has_new_content
        cur_segs = []
        cur_text_parts = []
        cur_start = None
        cur_end = None
        cur_token_count = 0
        has_new_content = False

    def tail_overlap_segments() -> List[Dict[str, Any]]:
        """
        Walk backwards through cur_segs, accumulating segments until
        we have at least overlap_tokens worth of text.
        Returns them in forward order.
        """
        if overlap_tokens <= 0:
            return []

        kept: List[Dict[str, Any]] = []
        total_tok = 0
        for seg in reversed(cur_segs):
            t = seg.get("text", "") or ""
            kept.append(seg)
            total_tok += count_tokens(t)
            if total_tok >= overlap_tokens:
                break
        kept.reverse()
        return kept

    def flush_chunk(*, keep_overlap: bool = True) -> None:
        nonlocal chunk_index, cur_segs, cur_text_parts, cur_start, cur_end, cur_token_count, has_new_content

        text = cur_text()
        if not text:
            return

        tok_count = cur_tok()

        cid = f"{video_id}:{chunk_index:04d}"
        meta = {
            "video_id": video_id,
            "source_url": source_url,
            "chunk_index": chunk_index,
            "start_seconds": float(cur_start or 0.0),
            "end_seconds": float(cur_end or 0.0),
            "token_count": tok_count,  # handy for debugging chunk sizes
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
        has_new_content = False  # chunk finalized

        # If we shouldn't carry overlap (hard boundaries / final flush), hard reset.
        if not keep_overlap:
            reset_state()
            return

        # Seed the next chunk with overlap
        overlap = tail_overlap_segments()
        cur_segs = overlap
        cur_text_parts = [s.get("text", "") or "" for s in overlap]
        cur_start = overlap[0]["start_seconds"] if overlap else None
        cur_end = overlap[-1]["end_seconds"] if overlap else None
        cur_token_count = sum(count_tokens(s.get("text", "") or "") for s in overlap)
        # has_new_content stays False until we add at least one new segment

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        # -- Step 1A: TRUE hard boundary --
        # If this segment contains [Applause] / [Music] / etc.,
        # flush what we have BEFORE absorbing it, do NOT carry overlap,
        # and SKIP the boundary segment itself (divider only).
        if _has_hard_boundary(text):
            if cur_text():
                flush_chunk(keep_overlap=False)
            else:
                reset_state()
            continue

        # Absorb segment into current chunk
        s0 = float(seg.get("start_seconds") or 0.0)
        s1 = float(seg.get("end_seconds") or s0)

        if cur_start is None:
            cur_start = s0
        cur_end = s1

        cur_segs.append({"start_seconds": s0, "end_seconds": s1, "text": text})
        cur_text_parts.append(text)
        cur_token_count += count_tokens(text)
        has_new_content = True

        # -- Step 1B: token-based size gate (+ time as secondary) --
        if (cur_tok() >= max_tokens or cur_dur() >= max_seconds) and cur_tok() >= min_tokens:
            flush_chunk(keep_overlap=True)

    # Flush any remaining content (avoid overlap-only duplicate)
    if has_new_content and cur_text():
        flush_chunk(keep_overlap=False)

    return chunks