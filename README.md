# GraphLens — Hybrid RAG + GraphRAG Framework

> Grounded, citation-anchored question answering over long-form video, course lectures, and academic documents.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-green)
![Neo4j](https://img.shields.io/badge/Neo4j-knowledge--graph-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-frontend-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

GraphLens is an end-to-end hybrid GraphRAG system that enables grounded, citation-anchored question answering across three heterogeneous content modalities — long-form YouTube lectures, multi-video course playlists, and PDF academic documents.

Standard RAG systems retrieve by vector similarity alone, missing conceptual relationships that span chapters, lectures, and documents. GraphLens addresses this through a 7-stage pipeline that combines dense vector retrieval, cross-encoder reranking, Neo4j knowledge graph expansion, citation-mandatory answer generation via Gemini 2.5 Flash, and a logistic regression reliability model that assigns a calibrated confidence score to every response.

---

## Demo

<!-- Replace with your actual demo video link -->
[![Watch Demo](assets/demo_thumbnail.png)](https://youtu.be/YOUR_DEMO_LINK)

---

## Screenshots

| PDF Tile | Concept Map |
|---|---|
| ![PDF Viewer](assets/screenshot_pdf.png) | ![Concept Map](assets/screenshot_graph.png) |

---

## Results

Evaluated on a 50-question benchmark across four retrieval conditions:

| Condition | Behavior Acc | Refusal Acc | Silver R@4 | Faithfulness |
|---|---|---|---|---|
| LLM Baseline | 0.618 | 0.000 | N/A | N/A |
| Plain RAG | 0.70 | 0.000 | 0.886 | 0.600 |
| RAG + Reranking | 0.72 | 0.133 | 1.000 | — |
| **GraphRAG** | **0.92** | **1.000** | **1.000** | **0.622** |

**Reliability Model:** Logistic regression trained on 12 retrieval and citation features. Cross-validation AUC: **0.984 ± 0.020**

**Key finding:** Graph expansion improves answer quality and produces perfect refusal accuracy as an emergent property — not an engineered rule.

---

## Architecture

![System Architecture](assets/architecture.png)

### 7-Stage Pipeline

| Stage | Component | Detail |
|---|---|---|
| 1 | **Ingest** | YouTube Transcript API · PyMuPDF · citation anchors |
| 2 | **Chunk** | tiktoken cl100k_base · 400 tokens · hard topic boundaries |
| 3 | **Retrieve** | ChromaDB · OpenAI text-embedding-3-small · top-20 |
| 4 | **Rerank** | ms-marco-MiniLM-L-6-v2 cross-encoder · top-4 |
| 5 | **Expand** | Neo4j 1-hop knowledge graph expansion · up to 6 chunks |
| 6 | **Generate** | Gemini 2.5 Flash · evidence-only · mandatory citations |
| 7 | **Score** | Logistic regression reliability model · calibrated 0–1 |

### Three Content Tiles

| Feature | YouTube URL | Course Playlist | PDF Document |
|---|---|---|---|
| Input | Any YouTube link | Multi-video playlist | Any PDF |
| Retrieval | Plain RAG | Full GraphRAG | Full GraphRAG |
| Citations | Timestamps | Lecture + timestamp | Page numbers |
| Latency | ~4s | ~6s | ~6s |

---

## Tech Stack

- **Backend:** FastAPI, Python 3.11
- **Frontend:** Streamlit
- **Vector Store:** ChromaDB
- **Knowledge Graph:** Neo4j Desktop
- **Embeddings:** OpenAI text-embedding-3-small
- **Reranker:** ms-marco-MiniLM-L-6-v2 (cross-encoder)
- **Generator:** Gemini 2.5 Flash via Vertex AI
- **Entity Extraction:** spaCy en_core_web_sm
- **Chunking:** tiktoken cl100k_base
- **Reliability Model:** scikit-learn logistic regression

---

## Project Structure

```
GraphLens/
├── server_backend/          # FastAPI backend
│   ├── api/v1/              # API endpoints
│   └── schemas/             # Pydantic schemas
├── src/graphlens/
│   ├── pipelines/           # Ingest + query pipelines
│   ├── graphrag/            # Neo4j + entity extraction
│   └── chunking/            # Token-aware chunker
├── app/
│   └── index.py             # Streamlit frontend
├── scripts/                 # Evaluation scripts
├── evaluation/              # Results and metrics
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11
- Neo4j Desktop (bolt://localhost:7687)
- OpenAI API key
- Google Cloud / Vertex AI credentials

### Installation

```bash
# Clone the repo
git clone https://github.com/tamabhi1997/Graphlens-RAG-GraphRAG-System.git
cd Graphlens-RAG-GraphRAG-System

# Create virtual environment
conda create -n graphlens_env python=3.11
conda activate graphlens_env

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Fill in your API keys in .env
```

### Running the System

```bash
# Terminal 1 — Backend
uvicorn server_backend.main:app --reload

# Terminal 2 — Frontend
streamlit run app/index.py
```

Make sure Neo4j Desktop is running with the GraphLens database active on bolt://localhost:7687.

---

## Evaluation

```bash
# Run full evaluation suite
python scripts/run_evaluation.py

# Run RAGAS faithfulness only
python scripts/run_evaluation.py ragas

# Train reliability model
python scripts/train_reliability.py
```

Results saved to `evaluation/` folder.

---

## Key Findings

- **Graph ≠ Retrieval** — GraphRAG does not improve Silver R@4 beyond reranking (both 1.000). The graph improves answer quality, not chunk selection.
- **Refusal is emergent** — Perfect refusal accuracy (1.000) was not engineered — it emerged from the interaction between graph expansion and the similarity-based gate.
- **Citations > Similarity** — Citation coverage (+0.711) is a stronger groundedness predictor than best_similarity (+0.406). Optimize for evidence usage, not retrieval score.

---

## Actionable Recommendations for RAG System Design

1. **Prioritize citation completeness over retrieval similarity** — systems should maximize how much of the retrieved evidence the generator actually cites
2. **Deploy hard refusal gates independently of the LLM** — delegating refusal to the language model produces zero refusal accuracy
3. **Use 400-token chunks for mixed-modality corpora** — 300 tokens fragments evidence, 600 tokens introduces topical noise

---


## Citation

If you use GraphLens in your research, please cite:

```bibtex
@misc{graphlens2026,
  title={GraphLens: A Hybrid RAG + GraphRAG Framework for Grounded Question Answering over Long-Form Video, Course Lectures, and Documents},
  author={Yadav, Abhishek Subhash and Gurav, Sayali Sunil and Gangrade, Khushi Himanshu and Lakawade, Prathamesh Arun and Chintapalli, Ganesh Kumar},
  year={2026},
  institution={Arizona State University}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
