from __future__ import annotations

import json
import os
import re
from typing import Optional

# ---------------------------------------------------------------------------
# spaCy lemmatizer — lazy singleton
# Handles plurals automatically: "matrices" → "matrix", "gradients" → "gradient"
# ---------------------------------------------------------------------------

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            _NLP = None
    return _NLP


# ---------------------------------------------------------------------------
# Alias map — loaded from aliases.json (configurable, no code changes needed)
# To add a new alias: open aliases.json and add "abbreviation": "full name"
# ---------------------------------------------------------------------------

_ALIAS_MAP: dict = {}
_ALIASES_PATH = os.path.join(os.path.dirname(__file__), "aliases.json")


def _get_alias_map() -> dict:
    global _ALIAS_MAP
    if not _ALIAS_MAP:
        if os.path.exists(_ALIASES_PATH):
            with open(_ALIASES_PATH, "r") as f:
                _ALIAS_MAP = json.load(f)
    return _ALIAS_MAP


# ---------------------------------------------------------------------------
# Regex rules
# ---------------------------------------------------------------------------

_LEADING_ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_SEPARATORS       = re.compile(r"[_\-]+")
_NOISE_CHARS      = re.compile(r"[^a-z0-9\s]")
_MULTI_SPACE      = re.compile(r"\s{2,}")


# ---------------------------------------------------------------------------
# Main normalisation function
# ---------------------------------------------------------------------------

def normalise_concept(name: str) -> Optional[str]:
    """
    Normalise a concept name to a canonical form.

    Pipeline:
      1. Lowercase
      2. Replace underscores/hyphens with spaces
      3. Strip leading articles (the, a, an)
      4. Remove noise characters
      5. Lemmatize using spaCy  ← handles all plurals automatically
      6. Collapse whitespace
      7. Apply alias map from aliases.json
      8. Return None if result is too short or invalid

    Args:
        name: Raw concept name from entity extraction

    Returns:
        Normalised concept name, or None if the concept is noise/invalid
    """
    if not name:
        return None

    text = name.strip().lower()

    # Replace separators with spaces
    text = _SEPARATORS.sub(" ", text)

    # Strip leading articles
    text = _LEADING_ARTICLES.sub("", text)

    # Remove noise characters
    text = _NOISE_CHARS.sub("", text)

    # Collapse whitespace
    text = _MULTI_SPACE.sub(" ", text).strip()

    if not text:
        return None

    # Apply alias map BEFORE lemmatization
    # (so "ml" → "machine learning" before spaCy sees it)
    alias_map = _get_alias_map()
    if text in alias_map:
        text = alias_map[text]

    # Lemmatize — spaCy converts plurals to singular automatically
    # "matrices" → "matrix", "gradients" → "gradient", "neural networks" → "neural network"
    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text)
        text = " ".join(token.lemma_ for token in doc)

    # Collapse whitespace again after lemmatization
    text = _MULTI_SPACE.sub(" ", text).strip()

    # Filter noise
    if len(text) < 3:
        return None
    if text.isdigit():
        return None

    return text
