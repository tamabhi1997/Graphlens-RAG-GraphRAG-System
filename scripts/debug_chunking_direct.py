#used to debug the chunking processs - 
from __future__ import annotations

from graphlens.utils.youtube_v1 import get_transcript_segments_v1
from graphlens.chunking.hybrid_v1 import chunk_transcript_doc
from graphlens.chunking.chunk_cleaner import clean_for_embedding


def main():
    url = "https://www.youtube.com/watch?v=alfdI7S6wCY&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI"

    doc = get_transcript_segments_v1(url, languages=["en"])
    chunks = chunk_transcript_doc(
        doc,
        max_chars=2200,
        min_chars=800,
        max_seconds=135.0,
        overlap_chars=250,
    )

    print(f"video_id={doc['video_id']}")
    print(f"segments={doc['segment_count']}")
    print(f"chunks={len(chunks)}")
    print("-" * 80)

    for i, c in enumerate(chunks[:5]):  # show first 5
        dur = c.end_seconds - c.start_seconds
        print(f"[{i:02d}] {c.chunk_id}  {c.start_seconds:.2f}->{c.end_seconds:.2f} ({dur:.2f}s)  chars={len(c.text)}")
        print(f"     preview={c.text[:]}...")
        

    # overlap check
    if len(chunks) >= 2:
        print("-" * 80)
        print("Overlap check (tail of chunk0 vs head of chunk1):")
        print("tail:", chunks[0].text[-220:])
        print("head:", chunks[1].text[:220])

    for c in chunks[:5]:
        cleaned = clean_for_embedding(c.text)
        print("RAW:", c.text[:])
        print("clean_text:", cleaned[:])


if __name__ == "__main__":
    main()