# summarize_v1.py
# Deterministic summarization — no LLM call.
# Better than original: picks representative sentences across chunks,
# filters noisy topics, supports bigrams.
# Will be replaced by Gemini after integration.

import re
from collections import Counter
from typing import List, Optional

# ---------------------------------------------------------------------------
# Stopwords — expanded to catch lecture/PDF filler and common short words
# ---------------------------------------------------------------------------
STOPWORDS = {
    # articles / conjunctions / prepositions
    "the","a","an","and","or","but","if","then","so","to","of","in","on","for",
    "with","as","at","by","from","into","about","through","between","among",
    "after","before","during","without","within","along","across","behind",
    # pronouns
    "it","this","that","these","those","we","you","i","they","he","she","them",
    "his","her","our","your","their","its","my","me","us","who","which","what",
    # verbs
    "is","are","was","were","be","been","being","can","could","should","would",
    "will","just","not","do","does","did","have","has","had","may","might",
    "shall","get","got","let","use","used","using","make","made","see","show",
    # lecture/PDF filler words that slip through
    "right","okay","know","going","want","need","think","look","call","also",
    "however","therefore","thus","hence","whereas","since","although","though",
    "example","note","remark","chapter","section","figure","table","page",
    "draft","version","copyright","reserved","press","university","book",
    # math filler
    "given","where","such","let","denote","define","assume","follows","result",
    "proof","theorem","lemma","corollary","equation","follows","case","well",
    # names that appear in textbook headers/footers
    "marc","peter","deisenroth","faisal","aldo","cambridge",
}

# Minimum word length for topics
MIN_WORD_LEN = 4

# Words that look like topics but aren't useful
_NOISE_RE = re.compile(r"^[0-9]|^\W|[0-9]{2,}")


# ---------------------------------------------------------------------------
# Utility helpers (unchanged — used by pipelines)
# ---------------------------------------------------------------------------

def seconds_to_hhmmss(seconds) -> Optional[str]:
    if seconds is None:
        return None
    s = int(round(float(seconds)))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def estimate_duration_seconds(transcript_doc) -> Optional[float]:
    segs = transcript_doc.get("segments", [])
    if not segs:
        return None
    return float(segs[-1].get("end_seconds", 0.0))


# ---------------------------------------------------------------------------
# Summary — pick best sentence from each of the first N chunks
# ---------------------------------------------------------------------------

def _score_sentence(sentence: str, word_counts: Counter) -> float:
    """
    Score a sentence by the sum of its words' corpus frequencies.
    High-frequency content words = more representative sentence.
    """
    words = re.findall(r"[a-z][a-z\-]{2,}", sentence.lower())
    content = [w for w in words if w not in STOPWORDS and len(w) >= MIN_WORD_LEN]
    if not content:
        return 0.0
    return sum(word_counts.get(w, 0) for w in content) / len(content)


def make_summary_from_text(
    text: str,
    max_sentences: int = 3,
    max_chars: int = 600,
) -> str:
    """
    Single-chunk summary fallback (kept for compatibility).
    Picks the highest-scoring sentence rather than always the first.
    """
    text = (text or "").strip()
    if not text:
        return ""
    return _best_sentences(text, max_sentences, max_chars)


def make_summary_from_chunks(
    clean_texts: List[str],
    max_sentences: int = 3,
    max_chars: int = 600,
) -> str:
    """
    Multi-chunk summary — picks the single best sentence from each of
    the first few chunks, then joins them into a coherent summary.

    This is what ingest pipelines should call instead of
    make_summary_from_text(clean_texts[0]).
    """
    if not clean_texts:
        return ""

    # Build a global word frequency map across first 10 chunks
    sample = clean_texts[:10]
    all_words = []
    for t in sample:
        words = re.findall(r"[a-z][a-z\-]{2,}", t.lower())
        all_words.extend([w for w in words if w not in STOPWORDS])
    word_counts = Counter(all_words)

    picked_sentences = []
    seen_starts = set()  # avoid near-duplicate sentences

    for chunk_text in clean_texts[:20]:
        sentences = re.split(r"(?<=[.!?])\s+", chunk_text.strip())
        # Filter: min 40 chars, must look like a real sentence
        candidates = [
            s for s in sentences
            if len(s) >= 40 and re.search(r"[a-zA-Z]{4,}", s)
        ]
        if not candidates:
            continue

        # Pick highest scoring sentence from this chunk
        best = max(candidates, key=lambda s: _score_sentence(s, word_counts))

        # Deduplicate by first 30 chars
        start_key = best[:30].lower()
        if start_key in seen_starts:
            continue
        seen_starts.add(start_key)

        picked_sentences.append(best.strip())
        if len(picked_sentences) >= max_sentences:
            break

    summary = " ".join(picked_sentences)

    # Trim to max_chars
    if len(summary) > max_chars:
        summary = summary[:max_chars].rstrip() + "..."

    return summary if summary else (clean_texts[0][:max_chars] if clean_texts else "")


def _best_sentences(text: str, n: int, max_chars: int) -> str:
    """Pick the n best sentences from a single text block."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    candidates = [s for s in sentences if len(s) >= 40]
    if not candidates:
        return text[:max_chars].strip()

    words = re.findall(r"[a-z][a-z\-]{2,}", text.lower())
    word_counts = Counter(w for w in words if w not in STOPWORDS)

    scored = sorted(candidates, key=lambda s: _score_sentence(s, word_counts), reverse=True)
    picked = " ".join(scored[:n]).strip()

    if len(picked) > max_chars:
        picked = picked[:max_chars].rstrip() + "..."
    return picked


# ---------------------------------------------------------------------------
# Topic extraction — expanded stopwords, bigrams, better filtering
# ---------------------------------------------------------------------------

def extract_key_topics(texts: List[str], top_n: int = 8) -> List[str]:
    """
    Improved keyword + bigram extraction:
    - Removes expanded stopword set
    - Requires minimum word length
    - Extracts bigrams (two-word phrases) that appear 2+ times
    - Prefers bigrams over unigrams when available
    - Filters noise patterns (numbers, single chars, etc.)
    """
    all_words = []
    all_bigrams = []

    for t in texts:
        t = (t or "").lower()
        words = re.findall(r"[a-z][a-z\-]{2,}", t)
        content_words = [
            w for w in words
            if w not in STOPWORDS
            and len(w) >= MIN_WORD_LEN
            and not _NOISE_RE.search(w)
        ]
        all_words.extend(content_words)

        # Bigrams from content words
        for w1, w2 in zip(content_words, content_words[1:]):
            all_bigrams.append(f"{w1} {w2}")

    word_counts = Counter(all_words)
    bigram_counts = Counter(all_bigrams)

    topics = []
    used_words = set()

    # First: add bigrams that appear 2+ times
    for bigram, count in bigram_counts.most_common(30):
        if count < 2:
            break
        w1, w2 = bigram.split()
        # Skip if either word is already covered
        if w1 in used_words or w2 in used_words:
            continue
        topics.append(bigram)
        used_words.add(w1)
        used_words.add(w2)
        if len(topics) >= top_n:
            break

    # Fill remaining slots with top unigrams not already covered
    for word, _ in word_counts.most_common(50):
        if len(topics) >= top_n:
            break
        if word in used_words:
            continue
        if len(word) < MIN_WORD_LEN:
            continue
        topics.append(word)
        used_words.add(word)

    return topics[:top_n]