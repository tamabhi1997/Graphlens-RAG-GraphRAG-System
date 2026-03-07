# this is without the llm call can not be perfect but will remove this after the gemini integration 
import re
from collections import Counter

# Very small stopword list (enough for MVP)
STOPWORDS = {
    "the","a","an","and","or","but","if","then","so","to","of","in","on","for","with","as","at","by",
    "is","are","was","were","be","been","being","it","this","that","these","those","we","you","i",
    "they","he","she","them","his","her","our","your","their","from","into","about","can","could",
    "should","would","will","just","not","do","does","did","have","has","had","what","why","how",
}

def seconds_to_hhmmss(seconds):
    if seconds is None:
        return None
    s = int(round(float(seconds)))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"

def estimate_duration_seconds(transcript_doc):
    # transcript_doc["segments"] has end_seconds; last end_seconds is the duration estimate
    segs = transcript_doc.get("segments", [])
    if not segs:
        return None
    return float(segs[-1].get("end_seconds", 0.0))

def make_summary_from_text(text, max_sentences=3, max_chars=600):
    """
    Deterministic MVP summary:
    - take first N sentences (or first max_chars if sentence split is weak)
    """
    text = (text or "").strip()
    if not text:
        return ""

    # naive sentence split
    sentences = re.split(r"(?<=[.!?])\s+", text)
    picked = " ".join(sentences[:max_sentences]).strip()

    if len(picked) < 80:
        # fallback to first chunk of text
        picked = text[:max_chars].strip()

    if len(picked) > max_chars:
        picked = picked[:max_chars].rstrip() + "..."
    return picked

def extract_key_topics(texts, top_n=8):
    """
    Simple keyword extraction:
    - tokenize words
    - remove stopwords/numbers
    - return top_n frequent terms
    """
    all_words = []
    for t in texts:
        t = (t or "").lower()
        words = re.findall(r"[a-z][a-z\-]{1,}", t)  # basic words
        all_words.extend([w for w in words if w not in STOPWORDS])

    counts = Counter(all_words)
    # remove very short/common leftovers
    topics = []
    for word, _ in counts.most_common(50):
        if len(word) < 3:
            continue
        topics.append(word)
        if len(topics) >= top_n:
            break
    return topics