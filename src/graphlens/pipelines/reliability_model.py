from __future__ import annotations

import os
import re
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Feature names — canonical order used by both training and inference
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "best_similarity",
    "rerank_score",
    "mean_similarity",
    "worst_similarity",
    "similarity_gap",
    "num_sources",
    "expanded_chunks",
    "graph_fired",
    "expansion_ratio",
    "num_citations",
    "citation_coverage",
    "answer_length",
]

# ---------------------------------------------------------------------------
# Feature extraction from query response
# ---------------------------------------------------------------------------

def extract_features(query_response: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract reliability features from a query pipeline response.
    All features computed directly from the API response — no extra calls.
    """
    sources  = query_response.get("sources", []) or []
    answer   = query_response.get("answer") or ""
    graph    = query_response.get("graph_expansion") or {}
    refused  = query_response.get("refused", True)

    # Retrieval features
    similarities  = [s.get("similarity", 0.0) for s in sources if s.get("similarity") is not None]
    rerank_scores = [s.get("rerank_score", 0.0) for s in sources if s.get("rerank_score") is not None]

    best_similarity  = max(similarities)  if similarities else 0.0
    worst_similarity = min(similarities)  if similarities else 0.0
    mean_similarity  = float(np.mean(similarities)) if similarities else 0.0
    similarity_gap   = best_similarity - worst_similarity
    num_sources      = float(len(sources))
    rerank_score     = max(rerank_scores) if rerank_scores else 0.0

    # Graph features
    expanded_chunks = float(graph.get("expanded_chunks", 0))
    graph_fired     = 1.0 if expanded_chunks > 0 else 0.0
    expansion_ratio = (expanded_chunks / num_sources) if num_sources > 0 else 0.0

    # Answer features
    citation_nums     = re.findall(r"\[(\d+)\]", answer)
    num_citations     = float(len(set(citation_nums)))
    citation_coverage = (num_citations / num_sources) if num_sources > 0 else 0.0
    answer_length     = float(len(answer))

    # Zero out answer features if refused
    if refused:
        num_citations     = 0.0
        citation_coverage = 0.0
        answer_length     = 0.0

    return {
        "best_similarity":   best_similarity,
        "rerank_score":      rerank_score,
        "mean_similarity":   mean_similarity,
        "worst_similarity":  worst_similarity,
        "similarity_gap":    similarity_gap,
        "num_sources":       num_sources,
        "expanded_chunks":   expanded_chunks,
        "graph_fired":       graph_fired,
        "expansion_ratio":   expansion_ratio,
        "num_citations":     num_citations,
        "citation_coverage": citation_coverage,
        "answer_length":     answer_length,
    }


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dict to numpy array in canonical FEATURE_NAMES order."""
    return np.array([features.get(name, 0.0) for name in FEATURE_NAMES], dtype=float)


# ---------------------------------------------------------------------------
# Reliability model
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "reliability_model.pkl")


class ReliabilityModel:
    """
    Logistic regression confidence scorer.
    Predicts P(answer is grounded | retrieval + graph features).

    Before training → uses heuristic scoring.
    After training  → uses logistic regression from reliability_model.pkl.
    """

    def __init__(self, clf=None, scaler=None):
        self.clf    = clf
        self.scaler = scaler

    def predict(self, query_response: Dict[str, Any]) -> float:
        """
        Returns float in [0, 1]. Higher = more grounded answer.
        Returns 0.0 for refused queries.
        """
        if query_response.get("refused", True):
            return 0.0

        if self.clf is None:
            return self._heuristic_score(query_response)

        features = extract_features(query_response)
        X = features_to_vector(features).reshape(1, -1)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        try:
            prob = self.clf.predict_proba(X)[0][1]
            return float(round(prob, 4))
        except Exception as exc:
            print(f"[reliability] model inference failed, using heuristic fallback: {exc}")
            return self._heuristic_score(query_response)

    def predict_with_explanation(self, query_response: Dict[str, Any]) -> Dict[str, Any]:
        """Predict confidence + show which features drove the score."""
        if self.clf is None:
            return {
                "confidence": self._heuristic_score(query_response),
                "explanation": "heuristic — run train_reliability.py to enable full model"
            }

        features = extract_features(query_response)
        X = features_to_vector(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X) if self.scaler else X
        try:
            prob = float(self.clf.predict_proba(X_scaled)[0][1])
        except Exception as exc:
            return {
                "confidence": self._heuristic_score(query_response),
                "explanation": f"heuristic fallback — trained model inference failed: {exc}",
                "feature_values": {k: round(v, 4) for k, v in features.items()},
            }

        contributions = {
            name: float(round(self.clf.coef_[0][i] * X_scaled[0][i], 4))
            for i, name in enumerate(FEATURE_NAMES)
        }
        sorted_contributions = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return {
            "confidence":     round(prob, 4),
            "feature_values": {k: round(v, 4) for k, v in features.items()},
            "contributions":  sorted_contributions,
        }

    def _heuristic_score(self, query_response: Dict[str, Any]) -> float:
        """Fallback when model not yet trained."""
        if query_response.get("refused", True):
            return 0.0
        best_sim = query_response.get("best_similarity") or 0.0
        rerank   = query_response.get("rerank_score") or 0.0
        expanded = (query_response.get("graph_expansion") or {}).get("expanded_chunks", 0)
        rerank_norm = max(0.0, min(1.0, (rerank + 5) / 15))
        graph_bonus = min(0.1, expanded * 0.02)
        score = (0.5 * best_sim) + (0.4 * rerank_norm) + (0.1 * graph_bonus)
        return float(round(min(1.0, score), 4))

    def save(self, path: str = MODEL_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump({"clf": self.clf, "scaler": self.scaler}, f)
        print(f"[reliability] Model saved to {path}")

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "ReliabilityModel":
        if not os.path.exists(path):
            print(f"[reliability] No model found at {path} — using heuristic scoring")
            return cls(clf=None, scaler=None)
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"[reliability] Loaded trained model from {path}")
        return cls(clf=data["clf"], scaler=data["scaler"])
