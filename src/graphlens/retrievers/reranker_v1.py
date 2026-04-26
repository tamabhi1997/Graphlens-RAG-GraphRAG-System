from __future__ import annotations

import os
from typing import List, Dict, Any

# Lazy import — model loads only on first call, not at import time
_RERANKER = None


def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        # The cross-encoder runs on PyTorch. Prevent Transformers from importing
        # TensorFlow/Keras, which can fail in environments with Keras 3 installed.
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        from sentence_transformers import CrossEncoder
        # Lightweight cross-encoder pre-trained on MS MARCO QA dataset
        # Runs on CPU in ~200ms for 20 chunks — no GPU needed
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _RERANKER


def rerank(
    question: str,
    sources: List[Dict[str, Any]],
    top_n: int = 4,
) -> List[Dict[str, Any]]:
    """
    Rerank retrieved sources using a cross-encoder.

    Takes the candidate sources from ChromaDB (top 20) and reorders them
    by true relevance to the question. Returns the top_n most relevant.

    The cross-encoder reads question + chunk text TOGETHER with full
    attention — far more accurate than cosine similarity alone.

    Args:
        question:  The user's original question string.
        sources:   List of source dicts from ChromaDB retrieval.
                   Each must have a "text" key.
        top_n:     How many to return after reranking.

    Returns:
        Reranked and trimmed list of source dicts, best first.
        Each source gets a new "rerank_score" field added.
    """
    if not sources:
        return sources

    reranker = _get_reranker()

    # Build pairs: [question, chunk_text] for each source
    pairs = [[question, s["text"]] for s in sources]

    # Cross-encoder scores all pairs in one batched forward pass
    scores = reranker.predict(pairs)  # returns numpy array of floats

    # Attach score to each source and sort descending
    for source, score in zip(sources, scores):
        source["rerank_score"] = float(score)

    reranked = sorted(sources, key=lambda s: s["rerank_score"], reverse=True)

    return reranked[:top_n]
