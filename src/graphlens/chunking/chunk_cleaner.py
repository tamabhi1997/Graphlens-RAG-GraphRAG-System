from __future__ import annotations

import re

# Common stage directions in transcripts: [Applause], [Music], [Laughter], etc.
_BRACKET_TAG_RE = re.compile(r"\[(?:[^\[\]]{1,40})\]")

# Filler words: remove only when they appear as standalone tokens
_FILLER_RE = re.compile(r"\b(?:uh|um|erm|er|ah|hmm)\b", flags=re.IGNORECASE)

# Collapse any repeated whitespace
_WS_RE = re.compile(r"\s+")


def clean_for_embedding(text: str) -> str:
    """
    Clean transcript text for embeddings / retrieval.

    Conservative cleaning:
    - removes bracketed stage directions like [Applause]
    - removes common filler words like 'uh', 'um' when standalone
    - normalizes whitespace

    Does NOT try to remove phrases like "you know" aggressively (can harm meaning).
    """
    if not text:
        return ""

    # Normalize non-breaking spaces and line breaks
    text = text.replace("\u00a0", " ").replace("\r", " ").replace("\n", " ")

    # Remove stage directions
    text = _BRACKET_TAG_RE.sub(" ", text)

    # Remove filler words
    text = _FILLER_RE.sub(" ", text)

    # Clean up stray punctuation spaces: " ,", " ."
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Collapse whitespace
    text = _WS_RE.sub(" ", text).strip()

    return text