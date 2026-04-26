from __future__ import annotations

import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig

load_dotenv()


# ---------------------------------------------------------------------------
# Lazy singleton client — uses Application Default Credentials (gcloud auth)
# Connects to global endpoint required for Gemini 3 Flash Preview
# ---------------------------------------------------------------------------

_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "graphlens")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        _CLIENT = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=HttpOptions(api_version="v1"),
        )
    return _CLIENT


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(question: str, sources: List[Dict[str, Any]]) -> str:
    evidence_blocks = []
    for i, source in enumerate(sources, 1):
        text = source.get("text", "").strip()
        start = source.get("start_seconds") or 0
        url = source.get("source_url", "")
        is_pdf = source.get("doc_id") is not None
        loc = f"Page {int(start)}" if is_pdf else f"{int(start)}s"
        evidence_blocks.append(f"[{i}] ({loc} — {url})\n{text}")

    evidence_str = "\n\n".join(evidence_blocks)

    return f"""You are a grounded question-answering assistant for an educational system.
Answer questions strictly based on the provided evidence passages. Do NOT use any outside knowledge.

STRICT RULES:
1. Answer ONLY using information from the evidence passages below.
2. If the evidence is insufficient, respond with exactly:
   "I don't have enough evidence to answer this question."
3. Answer as thoroughly as the evidence allows. Explain the concept 
   fully and clearly. If the evidence supports a detailed explanation, 
   provide one. Let the depth of the evidence determine your answer length.
4. After your answer, cite which passages you used, e.g. [1], [2].
5. Never fabricate facts not present in the evidence.

QUESTION:
{question}

EVIDENCE:
{evidence_str}

ANSWER:"""


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    sources: List[Dict[str, Any]],
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    Generate a grounded answer using Gemini 3 Flash Preview via Vertex AI.
    Uses google-genai SDK with global endpoint + Application Default Credentials.
    Draws from $300 Google Cloud credits.
    """
    model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

    if not sources:
        return {
            "answer": None,
            "refused": True,
            "citations": [],
            "model": model_name,
        }

    try:
        client = _get_client()
        prompt = _build_prompt(question, sources)

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=1024,
            ),
        )

        answer_text = (response.text or "").strip()
        if not answer_text and response.candidates:
            parts = getattr(response.candidates[0].content, "parts", []) or []
            answer_text = "".join(getattr(part, "text", "") or "" for part in parts).strip()
        if not answer_text:
            raise RuntimeError("Gemini returned an empty answer.")

        refused = "don't have enough evidence" in answer_text.lower()

        # Extract citation numbers e.g. [1], [2]
        citation_nums = [int(n) for n in re.findall(r"\[(\d+)\]", answer_text)]
        citations = list(dict.fromkeys(
            n - 1 for n in citation_nums if 1 <= n <= len(sources)
            ))

        return {
            "answer": None if refused else answer_text,
            "refused": refused,
            "citations": citations,
            "model": model_name,
        }

    except Exception as e:
        print(f"[generator] Gemini call failed: {e}")
        return {
            "answer": None,
            "refused": False,
            "citations": [],
            "model": model_name,
            "error": str(e),
        }
