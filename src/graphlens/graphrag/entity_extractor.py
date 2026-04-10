from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import spacy

# ---------------------------------------------------------------------------
# spaCy model — lazy singleton
# ---------------------------------------------------------------------------

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


# ---------------------------------------------------------------------------
# Concept normalisation helpers
# ---------------------------------------------------------------------------

# Words that look like concepts but aren't useful
_NOISE_WORDS = {
    "example", "figure", "table", "section", "chapter", "page",
    "note", "remark", "definition", "theorem", "proof", "result",
    "case", "step", "way", "type", "kind", "form", "part", "set",
    "number", "value", "point", "line", "term", "word", "thing",
}

# Relationship verb → canonical type mapping
_VERB_TO_REL = {
    "use":        "USES",
    "apply":      "USES",
    "employ":     "USES",
    "require":    "REQUIRES",
    "need":       "REQUIRES",
    "depend":     "REQUIRES",
    "train":      "TRAINS",
    "optimize":   "OPTIMIZES",
    "minimise":   "OPTIMIZES",
    "minimize":   "OPTIMIZES",
    "relate":     "RELATES_TO",
    "connect":    "RELATES_TO",
    "extend":     "RELATES_TO",
    "generalise": "RELATES_TO",
    "generalize": "RELATES_TO",
    "contain":    "PART_OF",
    "include":    "PART_OF",
    "comprise":   "PART_OF",
}


def _clean_concept(text: str) -> str:
    """Normalise a concept name — lowercase, strip, remove articles."""
    text = text.lower().strip()
    # Remove leading articles
    text = re.sub(r"^(the|a|an)\s+", "", text)
    return text.strip()


def _is_valid_concept(text: str) -> bool:
    """Filter out noise concepts."""
    text = text.lower().strip()
    if len(text) < 3:
        return False
    if text in _NOISE_WORDS:
        return False
    if re.match(r"^[0-9\W]+$", text):
        return False
    return True


def _verb_to_rel_type(verb_lemma: str) -> str:
    """Map a verb lemma to a canonical relationship type."""
    return _VERB_TO_REL.get(verb_lemma.lower(), "RELATES_TO")


# ---------------------------------------------------------------------------
# spaCy SVO extraction — primary method (free, local)
# ---------------------------------------------------------------------------

def _extract_noun_chunks(doc) -> List[str]:
    """Extract meaningful noun chunks as concept candidates."""
    concepts = []
    for chunk in doc.noun_chunks:
        # Use root of noun chunk for cleaner names
        concept = _clean_concept(chunk.root.text)
        if _is_valid_concept(concept):
            concepts.append(concept)
        # Also try the full chunk (for multi-word concepts like "gradient descent")
        full = _clean_concept(chunk.text)
        if full != concept and _is_valid_concept(full) and len(full.split()) <= 3:
            concepts.append(full)
    return list(dict.fromkeys(concepts))  # deduplicate preserving order


def _extract_svo_triples(doc) -> List[Dict[str, str]]:
    """
    Extract Subject-Verb-Object triples from parsed sentence.
    These become graph relationships: Subject -[VERB]-> Object
    """
    triples = []

    for token in doc:
        # Only look at main verbs (ROOT or relcl/advcl)
        if token.pos_ != "VERB":
            continue

        verb_lemma = token.lemma_.lower()
        rel_type = _verb_to_rel_type(verb_lemma)

        # Find subjects (nsubj, nsubjpass)
        subjects = [
            child for child in token.children
            if child.dep_ in ("nsubj", "nsubjpass")
        ]

        # Find objects (dobj, pobj, attr)
        objects = [
            child for child in token.children
            if child.dep_ in ("dobj", "pobj", "attr")
        ]

        for subj in subjects:
            for obj in objects:
                subj_text = _clean_concept(subj.text)
                obj_text = _clean_concept(obj.text)

                if _is_valid_concept(subj_text) and _is_valid_concept(obj_text):
                    triples.append({
                        "from": subj_text,
                        "type": rel_type,
                        "to": obj_text,
                    })

    return triples


def spacy_extract(chunk_text: str) -> Dict[str, Any]:
    """
    Primary extraction using spaCy.
    Returns concepts (nodes) and relationships (edges).
    """
    nlp = _get_nlp()

    # Process in sentence chunks (spaCy works best on sentences)
    doc = nlp(chunk_text[:5000])  # cap at 5000 chars for speed

    concepts = _extract_noun_chunks(doc)[:8]  # max 8 concepts per chunk
    relationships = _extract_svo_triples(doc)[:10]  # max 10 relationships

    return {
        "concepts": [{"name": c, "description": ""} for c in concepts],
        "relationships": relationships,
        "method": "spacy",
    }


# ---------------------------------------------------------------------------
# LLM fallback — called when spaCy finds no relationships
# ---------------------------------------------------------------------------

def llm_extract(chunk_text: str) -> Dict[str, Any]:
    """
    Fallback extraction using OpenAI when spaCy returns no relationships.
    Uses gpt-4o-mini for cost efficiency (~$0.0002 per chunk).
    """
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""Extract concepts and relationships from this text.
Return ONLY valid JSON, no other text, no markdown:

{{
  "concepts": [
    {{"name": "concept_name_lowercase", "description": "one sentence definition"}}
  ],
  "relationships": [
    {{"from": "concept1", "type": "USES", "to": "concept2"}}
  ]
}}

Rules:
- Max 5 concepts, max 5 relationships
- concept names must be lowercase
- relationship type must be one of: USES, TRAINS, REQUIRES, RELATES_TO, PART_OF, OPTIMIZES
- only extract what is explicitly in the text

Text: {chunk_text[:1500]}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        result["method"] = "llm"
        return result

    except Exception as e:
        print(f"[entity_extractor] LLM extraction failed: {e}")
        return {"concepts": [], "relationships": [], "method": "llm_failed"}


# ---------------------------------------------------------------------------
# Main entry point — spaCy first, LLM fallback
# ---------------------------------------------------------------------------

def extract_entities(chunk_text: str) -> Dict[str, Any]:
    """
    Extract entities and relationships from a chunk.

    Strategy:
      1. Try spaCy SVO extraction (free, local, fast)
      2. If spaCy finds no relationships → fall back to LLM (~$0.0002)

    Returns:
        {
          "concepts": [{"name": str, "description": str}],
          "relationships": [{"from": str, "type": str, "to": str}],
          "method": "spacy" | "llm" | "llm_failed"
        }
    """
    result = spacy_extract(chunk_text)

    # Fall back to LLM if spaCy found no relationships
    if len(result["relationships"]) == 0:
        print(f"[entity_extractor] spaCy found no relationships, trying LLM...")
        result = llm_extract(chunk_text)

    return result