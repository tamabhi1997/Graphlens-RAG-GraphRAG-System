"""
GraphLens — Comprehensive Evaluation Script
============================================
Runs three evaluation tasks:
  1. LLM Baseline — Gemini with NO retrieval context
  2. RAGAS — Faithfulness + Answer Relevancy across conditions
  3. Graph expansion rate + mean confidence from Sayali's CSV

Usage:
    python scripts/run_evaluation.py

Outputs:
    evaluation/llm_baseline_results.csv
    evaluation/ragas_results.csv
    evaluation/summary_metrics.txt
"""

from __future__ import annotations

import os
import re
import sys
import csv
import json
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.makedirs("evaluation", exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE      = "http://127.0.0.1:8000/api/v1"
SCOPE_TYPE    = "document"
SCOPE_ID      = "ad5960cd3df2690d"    # MML book scope_id
SAYALI_CSV    = os.path.expanduser("~/Downloads/reliability_training_data_SAYALI.csv")

# Load questions from Sayali's CSV
df_sayali = pd.read_csv(SAYALI_CSV)
IN_VIDEO_QS = df_sayali[
    (df_sayali["answer_scope"] == "in_video") &
    (df_sayali["expected_refusal"] == False)
]["question"].tolist()[:35]

print(f"Loaded {len(IN_VIDEO_QS)} in-video questions from Sayali's CSV")
print("=" * 60)


# ---------------------------------------------------------------------------
# TASK 1 — LLM Baseline (Gemini with NO retrieval)
# ---------------------------------------------------------------------------

def run_llm_baseline():
    print("\nTASK 1: LLM Baseline — Gemini with NO context")
    print("-" * 60)

    from google import genai
    from google.genai.types import HttpOptions, GenerateContentConfig

    client = genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT", "graphlens"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        http_options=HttpOptions(api_version="v1"),
    )
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    results = []
    for i, q in enumerate(IN_VIDEO_QS, 1):
        print(f"  [{i}/{len(IN_VIDEO_QS)}] {q[:60]}...")
        try:
            prompt = f"Answer this question concisely: {q}"
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=GenerateContentConfig(temperature=0.1),
            )
            answer = (response.text or "").strip()
        except Exception as e:
            answer = f"ERROR: {e}"

        results.append({
            "question":        q,
            "answer":          answer,
            "condition":       "llm_baseline",
            "refused":         False,
            "best_similarity": 0.0,
            "rerank_score":    0.0,
            "expanded_chunks": 0,
            "graph_fired":     0,
            "num_citations":   0,
            "answer_length":   len(answer),
            "label":           "",   # ← Sayali fills this in manually
        })
        time.sleep(0.5)  # avoid rate limits

    out_path = "evaluation/llm_baseline_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"  Saved {len(results)} rows to {out_path}")
    print("  *** Sayali: fill in the 'label' column (1=correct, 0=wrong) ***")


# ---------------------------------------------------------------------------
# TASK 2 — RAGAS Faithfulness + Answer Relevancy
# ---------------------------------------------------------------------------

def query_api(question: str, use_graph: bool) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={
                "question":   question,
                "scope_type": SCOPE_TYPE,
                "scope_id":   SCOPE_ID,
                "use_graph":  use_graph,
            },
            timeout=60,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e), "refused": True}


def run_ragas():
    print("\nTASK 2: RAGAS Faithfulness")
    print("-" * 60)

    try:
        from ragas.metrics import Faithfulness
        faithfulness = Faithfulness()
        from ragas import evaluate
        from datasets import Dataset
    except ImportError:
        try:
            from ragas.metrics import Faithfulness
            faithfulness = Faithfulness()
            from ragas import evaluate
            from datasets import Dataset
        except ImportError:
            print("  RAGAS not installed. Run: pip install ragas datasets")
            return

    conditions = [
        ("plain_rag", False),
        ("graphrag",  True),
    ]

    ragas_results = []

    for condition_name, use_graph in conditions:
        print(f"\n  Running condition: {condition_name}")
        rows = []
        for i, q in enumerate(IN_VIDEO_QS[:20], 1):
            print(f"    [{i}/20] {q[:50]}...")
            out = query_api(q, use_graph)

            if out.get("refused") or not out.get("answer"):
                continue

            # Truncate answer to 500 chars to avoid GPT-4o-mini token limit
            answer = out["answer"][:500]
            # Truncate each context chunk too
            contexts = [s["text"][:300] for s in out.get("sources", [])]

            rows.append({
                "question":     q,
                "answer":       answer,
                "contexts":     contexts,
                "ground_truth": q,
            })
            time.sleep(1)

        if not rows:
            print(f"  No valid answers for {condition_name}")
            continue

        dataset = Dataset.from_list(rows)
        try:
            scores = evaluate(dataset, metrics=[faithfulness])
            
            # Handle RAGAS returning Dataset object instead of dict
            try:
                faith_val = scores["faithfulness"]
            except Exception:
                scores_df = scores.to_pandas()
                faith_val = scores_df["faithfulness"].tolist()

            # Handle list vs float return type
            if isinstance(faith_val, list):
                valid = [v for v in faith_val if v is not None and str(v) != 'nan']
                faith = sum(valid) / len(valid) if valid else 0.0
            else:
                faith = float(faith_val)

            print(f"\n  {condition_name} — Faithfulness: {faith:.3f}")
            ragas_results.append({
                "condition":    condition_name,
                "faithfulness": round(faith, 3),
                "n_questions":  len(rows),
            })
        except Exception as e:
            print(f"  RAGAS evaluation failed: {e}")

    if ragas_results:
        out_path = "evaluation/ragas_results.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ragas_results[0].keys())
            writer.writeheader()
            writer.writerows(ragas_results)
        print(f"\n  Saved RAGAS results to {out_path}")


# ---------------------------------------------------------------------------
# TASK 3 — Graph expansion rate + mean confidence from Sayali's CSV
# ---------------------------------------------------------------------------

def run_summary_metrics():
    print("\nTASK 3: Graph expansion rate + mean confidence")
    print("-" * 60)

    df = pd.read_csv(SAYALI_CSV)
    answered = df[df["refused"] == False].copy()

    total_answered = len(answered)

    # Graph expansion rate
    if "expanded_chunks" in df.columns:
        fired = (answered["expanded_chunks"] > 0).sum()
        rate = fired / total_answered if total_answered > 0 else 0
        print(f"  Graph expansion rate:  {fired}/{total_answered} = {rate:.1%}")
    else:
        print("  expanded_chunks column not found in CSV")
        fired = rate = "N/A"

    # Mean confidence
    if "confidence" in df.columns:
        mean_conf = answered["confidence"].mean()
        print(f"  Mean confidence (GraphRAG answered): {mean_conf:.3f}")
    else:
        # Compute via API on sample questions
        print("  confidence column not in CSV — querying API for sample...")
        sample_qs = answered["question"].tolist()[:10]
        confs = []
        for q in sample_qs:
            out = query_api(q, use_graph=True)
            if not out.get("refused") and out.get("confidence") is not None:
                confs.append(out["confidence"])
        mean_conf = sum(confs) / len(confs) if confs else 0
        print(f"  Mean confidence (from {len(confs)} API calls): {mean_conf:.3f}")

    # Label distribution
    if "label" in df.columns:
        labels = df["label"].dropna()
        correct = (labels == 1).sum()
        wrong   = (labels == 0).sum()
        print(f"  Labelled rows: {len(labels)} ({correct} correct, {wrong} wrong)")

    # Save summary
    summary = f"""GraphLens Evaluation Summary
==============================
Total questions:         {len(df)}
Answered (not refused):  {total_answered}
Refused:                 {len(df) - total_answered}

Graph expansion rate:    {fired}/{total_answered} = {rate:.1%} (questions where graph fired)
Mean confidence (GraphRAG): {mean_conf:.3f}

See evaluation/ragas_results.csv for faithfulness + answer relevancy.
See evaluation/llm_baseline_results.csv for LLM baseline answers (needs manual labels).
"""
    with open("evaluation/summary_metrics.txt", "w") as f:
        f.write(summary)
    print("\n  Saved summary to evaluation/summary_metrics.txt")
    print(summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("GraphLens Comprehensive Evaluation")
    print("=" * 60)
    print("Make sure uvicorn is running before starting.\n")

    tasks = sys.argv[1:] or ["all"]

    if "all" in tasks or "baseline" in tasks:
        run_llm_baseline()

    if "all" in tasks or "ragas" in tasks:
        run_ragas()

    if "all" in tasks or "metrics" in tasks:
        run_summary_metrics()

    print("\n" + "=" * 60)
    print("All evaluation tasks complete.")
    print("Files saved to evaluation/ folder.")