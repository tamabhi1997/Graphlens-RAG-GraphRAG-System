from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Bracket-tag hard boundary pattern — same as hybrid_v2
# PDFs sometimes contain [Figure 1], [Table 2] etc. — we treat those as
# section dividers too.
# ---------------------------------------------------------------------------
_BRACKET_TAG_RE = re.compile(r"\[(?:[^\[\]]{1,40})\]")


def _looks_like_heading(text: str) -> bool:
    """
    Heuristic: a short line (< 80 chars) that is ALL CAPS or Title Case
    and has no sentence-ending punctuation is likely a section heading.
    Used as a soft hard-boundary signal in PDFs.
    """
    text = text.strip()
    if not text or len(text) > 80:
        return False
    if text[-1] in ".?!:,;":
        return False
    words = text.split()
    if len(words) < 1:
        return False
    # ALL CAPS heading
    if text.isupper() and len(words) >= 1:
        return True
    # Title Case heading (every word capitalised, none lowercased)
    if all(w[0].isupper() for w in words if w.isalpha()) and len(words) >= 2:
        return True
    return False


def _clean_extracted_text(text: str) -> str:
    """
    Light normalisation on raw PDF text before segmenting.
    - Collapse excessive blank lines
    - Normalise whitespace within lines
    - Remove page header/footer noise (short lines that repeat)
    """
    text = text.replace("\r", "\n")
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse repeated spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def generate_doc_id(filename: str, content_bytes: bytes) -> str:
    """
    Deterministic document ID from filename + first 4KB of content.
    Stable across re-uploads of the same file.
    """
    fingerprint = filename + hashlib.md5(content_bytes[:4096]).hexdigest()
    return hashlib.md5(fingerprint.encode()).hexdigest()[:16]


def extract_pdf_segments(
    pdf_bytes: bytes,
    filename: str = "document.pdf",
) -> Dict[str, Any]:
    """
    Extract text from a PDF and return a segment dict that matches
    the format produced by get_transcript_segments_v1() for YouTube.

    Each segment has:
        text        — paragraph/block text
        page_number — 1-indexed page (replaces start_seconds)
        block_index — global block counter (replaces end_seconds)

    The chunker (hybrid_v2) receives these as-is. Because PDFs have no
    real timestamps, page_number is stored in metadata instead.

    Args:
        pdf_bytes:  Raw bytes of the PDF file.
        filename:   Original filename — used for doc_id generation.

    Returns:
        A dict with keys: doc_id, filename, page_count, segment_count, segments.
    """
    doc_id = generate_doc_id(filename, pdf_bytes)

    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = pdf.page_count

    segments: List[Dict[str, Any]] = []
    block_index = 0

    for page_num in range(page_count):
        page = pdf[page_num]
        page_number = page_num + 1  # 1-indexed for humans

        # extract_text with "blocks" gives us paragraph-level blocks
        # each block: (x0, y0, x1, y1, text, block_no, block_type)
        blocks = page.get_text("blocks", sort=True)  # sort=True = reading order

        for block in blocks:
            raw_text = block[4]  # index 4 is the text content
            block_type = block[6]  # 0=text, 1=image

            # Skip image blocks and empty blocks
            if block_type != 0:
                continue

            text = _clean_extracted_text(raw_text)
            if not text or len(text) < 10:  # skip noise (page numbers, etc.)
                continue

            # Split block into lines and detect headings as boundary signals
            # We insert a [Section] tag before headings so hybrid_v2's
            # hard boundary detector fires — same mechanism as [Applause]
            lines = [l.strip() for l in text.split("\n") if l.strip()]

            for line in lines:
                if not line:
                    continue

                # Insert a synthetic boundary tag before headings
                if _looks_like_heading(line):
                    tag_text = f"[Section: {line}]"
                    segments.append({
                        "text": tag_text,
                        "page_number": page_number,
                        "block_index": block_index,
                        # hybrid_v2 uses start/end seconds — we pass page as both
                        "start_seconds": float(page_number),
                        "end_seconds": float(page_number),
                    })
                    block_index += 1

                segments.append({
                    "text": line,
                    "page_number": page_number,
                    "block_index": block_index,
                    # Store page_number in seconds fields so hybrid_v2
                    # can track start/end page of each chunk
                    "start_seconds": float(page_number),
                    "end_seconds": float(page_number),
                })
                block_index += 1

    pdf.close()

    return {
        "doc_id": doc_id,
        "filename": filename,
        "page_count": page_count,
        "segment_count": len(segments),
        "segments": segments,
        # Mimic youtube_v1 keys so ingest pipeline stays generic
        "video_id": doc_id,        # chunker uses video_id internally
        "source_url": filename,    # used in chunk metadata
    }