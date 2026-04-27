# summarize_v1.py
# Hybrid summarization:
#   - make_summary_from_chunks → Gemini LLM (representative sampling)
#   - extract_key_topics → deterministic bigram extraction (free, fast)
#   - All other helpers unchanged

import os
import re
import json
from collections import Counter
from typing import List, Optional

# ---------------------------------------------------------------------------
# Stopwords — expanded to catch lecture/PDF filler and common short words
# ---------------------------------------------------------------------------
STOPWORDS = {
    "the","a","an","and","or","but","if","then","so","to","of","in","on","for",
    "with","as","at","by","from","into","about","through","between","among",
    "after","before","during","without","within","along","across","behind",
    "it","this","that","these","those","we","you","i","they","he","she","them",
    "his","her","our","your","their","its","my","me","us","who","which","what",
    "is","are","was","were","be","been","being","can","could","should","would",
    "will","just","not","do","does","did","have","has","had","may","might",
    "shall","get","got","let","use","used","using","make","made","see","show",
    "right","okay","know","going","want","need","think","look","call","also",
    "however","therefore","thus","hence","whereas","since","although","though",
    "example","note","remark","chapter","section","figure","table","page",
    "draft","version","copyright","reserved","press","university","book",
    "given","where","such","let","denote","define","assume","follows","result",
    "proof","theorem","lemma","corollary","equation","follows","case","well",
    "marc","peter","deisenroth","faisal","aldo","cambridge",
}

MIN_WORD_LEN = 4
_NOISE_RE = re.compile(r"^[0-9]|^\W|[0-9]{2,}")


# ---------------------------------------------------------------------------
# Utility helpers (unchanged)
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
# Representative chunk sampling
# ---------------------------------------------------------------------------

def _sample_chunks(clean_texts: List[str], n_samples: int = 20) -> List[str]:
    """
    Sample n_samples chunks spread evenly across the full document.
    Always includes first 3 (intro) and last 2 (conclusion).
    Middle chunks are evenly spaced — gives a representative overview
    of a 400-page book without exceeding Gemini context limits.
    """
    total = len(clean_texts)
    if total <= n_samples:
        return clean_texts

    head = clean_texts[:3]
    tail = clean_texts[-2:]
    middle = clean_texts[3:-2]
    middle_slots = n_samples - 5

    if middle_slots <= 0 or not middle:
        return head + tail

    step = max(1, len(middle) // middle_slots)
    sampled_middle = [middle[i] for i in range(0, len(middle), step)][:middle_slots]

    return head + sampled_middle + tail


# ---------------------------------------------------------------------------
# Gemini summary call
# ---------------------------------------------------------------------------
def _call_gemini_summary(sampled_texts: List[str]) -> Optional[dict]:
    """
    Call Gemini to generate a structured overview summary from sampled chunks.
    Returns None on failure so caller falls back to deterministic method.
    """
    try:
        from google import genai
        from google.genai.types import HttpOptions, GenerateContentConfig

        project  = os.getenv("GOOGLE_CLOUD_PROJECT", "graphlens")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        model    = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=HttpOptions(api_version="v1"),
        )

        evidence = "\n\n---\n\n".join(
            f"[Excerpt {i+1}]\n{t.strip()}"
            for i, t in enumerate(sampled_texts)
            if t.strip()
        )

        prompt = f"""You are an intelligent document analysis assistant.
Based on the following excerpts sampled from across the document, provide:

1. SUMMARY: A clear overall summary (3-5 sentences) covering what this document
   is about, its main themes, and who would benefit from it. Write in clear,
   accessible language suited to the content type.

2. KEY_TOPICS: The most important topics covered in this document.
   Format each topic EXACTLY like this:
   "<topic name>: <1-2 sentences explaining what it covers and why it matters>"
   Extract as many topics as the content warrants. Do not force a fixed number.

Respond ONLY in this JSON format, no other text:
{{
  "summary": "overall summary here",
  "key_topics": [
    "<topic name>: <description>",
    "<topic name>: <description>"
  ]
}}

EXCERPTS:
{evidence}"""

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.2
            ),
        )
        raw = response.text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(raw)
        return {
            "summary": parsed.get("summary", ""),
            "key_topics": parsed.get("key_topics", [])
        }

    except Exception as e:
        print(f"[summarize] Gemini summary failed, falling back to deterministic: {e}")
        return None


# ---------------------------------------------------------------------------
# Summary — Gemini primary, deterministic fallback
# ---------------------------------------------------------------------------

def make_summary_from_text(
    text: str,
    max_sentences: int = 3,
    max_chars: int = 600,
) -> str:
    """Single-chunk summary fallback (kept for compatibility)."""
    text = (text or "").strip()
    if not text:
        return ""
    return _best_sentences(text, max_sentences, max_chars)


# def make_summary_from_chunks(
#     clean_texts: List[str],
#     max_sentences: int = 3,
#     max_chars: int = 600,
# ) -> str:
#     """
#     Generate a document summary using Gemini LLM with representative sampling.

#     Samples ~20 chunks spread evenly across the full document and passes
#     them to Gemini for a proper overview summary. Falls back to deterministic
#     sentence scoring if Gemini is unavailable.

#     Call sites in ingest_youtube_v1.py and ingest_pdf_v1.py are unchanged.
#     """
#     if not clean_texts:
#         return ""

#     # Sample representative chunks from across the whole document
#     sampled = _sample_chunks(clean_texts, n_samples=20)

#     # Try Gemini first
#     gemini_summary = _call_gemini_summary(sampled)
#     if gemini_summary:
#         return gemini_summary

#     # Fallback: deterministic sentence scoring
#     sample = clean_texts[:10]
#     all_words = []
#     for t in sample:
#         words = re.findall(r"[a-z][a-z\-]{2,}", t.lower())
#         all_words.extend([w for w in words if w not in STOPWORDS])
#     word_counts = Counter(all_words)

#     picked_sentences = []
#     seen_starts = set()

#     for chunk_text in clean_texts[:20]:
#         sentences = re.split(r"(?<=[.!?])\s+", chunk_text.strip())
#         candidates = [
#             s for s in sentences
#             if len(s) >= 40 and re.search(r"[a-zA-Z]{4,}", s)
#         ]
#         if not candidates:
#             continue
#         best = max(candidates, key=lambda s: _score_sentence(s, word_counts))
#         start_key = best[:30].lower()
#         if start_key in seen_starts:
#             continue
#         seen_starts.add(start_key)
#         picked_sentences.append(best.strip())
#         if len(picked_sentences) >= max_sentences:
#             break

#     summary = " ".join(picked_sentences)
#     if len(summary) > max_chars:
#         summary = summary[:max_chars].rstrip() + "..."
#     return summary if summary else (clean_texts[0][:max_chars] if clean_texts else "")

def make_summary_from_chunks(clean_texts: List[str]) -> tuple:
    """
    Generate summary and key topics using Gemini LLM with representative sampling.
    Returns a tuple: (summary_string, key_topics_list)
    Falls back to deterministic summary + empty topics if Gemini fails.
    """
    if not clean_texts:
        return "", []

    # Sample representative chunks from across the whole document
    sampled = _sample_chunks(clean_texts, n_samples=20)

    # Try Gemini first
    result = _call_gemini_summary(sampled)
    if result:
        return result["summary"], result["key_topics"]

    # Fallback — deterministic sentence scoring, no topics
    sample = clean_texts[:10]
    all_words = []
    for t in sample:
        words = re.findall(r"[a-z][a-z\-]{2,}", t.lower())
        all_words.extend([w for w in words if w not in STOPWORDS])
    word_counts = Counter(all_words)

    picked_sentences = []
    seen_starts = set()

    for chunk_text in clean_texts[:20]:
        sentences = re.split(r"(?<=[.!?])\s+", chunk_text.strip())
        candidates = [
            s for s in sentences
            if len(s) >= 40 and re.search(r"[a-zA-Z]{4,}", s)
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda s: _score_sentence(s, word_counts))
        start_key = best[:30].lower()
        if start_key in seen_starts:
            continue
        seen_starts.add(start_key)
        picked_sentences.append(best.strip())
        if len(picked_sentences) >= 3:
            break

    summary = " ".join(picked_sentences)
    return summary, []

def _score_sentence(sentence: str, word_counts: Counter) -> float:
    words = re.findall(r"[a-z][a-z\-]{2,}", sentence.lower())
    content = [w for w in words if w not in STOPWORDS and len(w) >= MIN_WORD_LEN]
    if not content:
        return 0.0
    return sum(word_counts.get(w, 0) for w in content) / len(content)


def _best_sentences(text: str, n: int, max_chars: int) -> str:
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
# Topic extraction — unchanged, deterministic bigram extraction
# ---------------------------------------------------------------------------

def extract_key_topics(texts: List[str], top_n: int = 8) -> List[str]:
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
        for w1, w2 in zip(content_words, content_words[1:]):
            all_bigrams.append(f"{w1} {w2}")

    word_counts = Counter(all_words)
    bigram_counts = Counter(all_bigrams)

    topics = []
    used_words = set()

    for bigram, count in bigram_counts.most_common(30):
        if count < 2:
            break
        w1, w2 = bigram.split()
        if w1 in used_words or w2 in used_words:
            continue
        topics.append(bigram)
        used_words.add(w1)
        used_words.add(w2)
        if len(topics) >= top_n:
            break

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