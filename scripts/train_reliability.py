"""
GraphLens Reliability Model — Training Script
==============================================
Run this once when Sayali's evaluation CSV is ready.

Usage:
    python scripts/train_reliability.py --csv path/to/reliability_training_data_SAYALI.csv

What it does:
    1. Loads evaluation CSV
    2. Computes derived features (graph_fired, expansion_ratio, etc.)
    3. Sets label=0 for all refused rows automatically
    4. Trains logistic regression with cross-validation
    5. Prints feature weights for the report
    6. Saves reliability_model.pkl to src/graphlens/pipelines/

After running:
    Restart uvicorn — every query response will include a confidence score.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphlens.pipelines.reliability_model import (
    FEATURE_NAMES,
    MODEL_PATH,
    ReliabilityModel,
)


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the derived feature columns from raw API response data.
    These columns may not be present in Sayali's CSV — we compute them here.
    """
    # Graph features
    if "graph_fired" not in df.columns:
        df["graph_fired"] = (df["expanded_chunks"] > 0).astype(float)

    if "expansion_ratio" not in df.columns:
        df["expansion_ratio"] = df["expanded_chunks"] / df["num_sources"].replace(0, 1)

    # Answer features — count [1][2][3] in answer text
    if "num_citations" not in df.columns:
        def count_citations(answer):
            if pd.isna(answer) or answer == "":
                return 0.0
            return float(len(set(re.findall(r"\[(\d+)\]", str(answer)))))
        df["num_citations"] = df["answer"].apply(count_citations)

    if "citation_coverage" not in df.columns:
        df["citation_coverage"] = df["num_citations"] / df["num_sources"].replace(0, 1)

    if "answer_length" not in df.columns:
        df["answer_length"] = df["answer"].apply(
            lambda x: float(len(str(x))) if pd.notna(x) else 0.0
        )

    # Zero out answer features for refused rows
    refused_mask = df["refused"] == True
    df.loc[refused_mask, "num_citations"]     = 0.0
    df.loc[refused_mask, "citation_coverage"] = 0.0
    df.loc[refused_mask, "answer_length"]     = 0.0

    # Similarity derived features — approximate if individual source similarities not available
    if "mean_similarity" not in df.columns:
        # Use top_source_similarity as proxy if available, else use best_similarity
        proxy = df.get("top_source_similarity", df["best_similarity"])
        df["mean_similarity"] = (df["best_similarity"] + proxy) / 2

    if "worst_similarity" not in df.columns:
        df["worst_similarity"] = df.get("top_source_similarity", df["best_similarity"])

    if "similarity_gap" not in df.columns:
        df["similarity_gap"] = df["best_similarity"] - df["worst_similarity"]

    return df


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Auto-label refused rows as 0
    refused_before = df["label"].isna().sum()
    df.loc[df["refused"] == True, "label"] = 0.0
    print(f"Auto-labelled {refused_before} refused rows as label=0")

    # Compute derived features
    df = compute_derived_features(df)

    # Final label check
    still_missing = df["label"].isna().sum()
    if still_missing > 0:
        print(f"WARNING: {still_missing} rows still have no label — dropping them")
        df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)
    print(f"Final training set: {len(df)} rows")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df


def train(csv_path: str, output_path: str = MODEL_PATH) -> None:
    print("=" * 60)
    print("GraphLens — Reliability Model Training")
    print("=" * 60)

    # 1. Load and prepare
    df = load_and_prepare(csv_path)

    X = df[FEATURE_NAMES].fillna(0.0).values
    y = df["label"].values

    # 2. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Train
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    # 4. Cross-validation
    n_splits = min(5, int(y.sum()), int((len(y) - y.sum())))
    n_splits = max(2, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")

    print(f"\nCross-validation AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Individual folds:     {[round(s, 3) for s in cv_scores]}")

    # 5. Final fit
    clf.fit(X_scaled, y)
    y_pred = clf.predict(X_scaled)

    print("\nClassification Report (training set):")
    print(classification_report(y, y_pred, target_names=["not grounded", "grounded"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    # 6. Feature weights — goes into the report
    print("\n" + "=" * 60)
    print("Feature Weights (sorted by importance):")
    print("=" * 60)
    weights = list(zip(FEATURE_NAMES, clf.coef_[0]))
    for name, weight in sorted(weights, key=lambda x: abs(x[1]), reverse=True):
        direction = "↑ increases confidence" if weight > 0 else "↓ decreases confidence"
        print(f"  {name:<25} {weight:+.4f}   {direction}")
    print(f"\n  Intercept: {clf.intercept_[0]:+.4f}")

    # 7. Save
    model = ReliabilityModel(clf=clf, scaler=scaler)
    model.save(output_path)

    print("\n" + "=" * 60)
    print("Done! Copy the output above into your report.")
    print(f"Model saved to: {output_path}")
    print("Restart uvicorn — confidence scores will now appear in all query responses.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to Sayali's evaluation CSV")
    parser.add_argument("--output", default=MODEL_PATH, help="Where to save the model")
    args = parser.parse_args()
    train(args.csv, args.output)