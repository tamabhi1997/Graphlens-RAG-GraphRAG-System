# GraphLens

Starter template for a Python-based data project with:

- RAG pipelines
- Vector database integrations
- Streamlit or Gradio frontends
- Local scripts, notebooks, and test scaffolding

## Project Structure

```text
GraphLens/
├── app/                    # Frontend entry points
├── server_backend/         # FastAPI backend (API endpoints for transcript/ingest/query)
├── config/                 # YAML / TOML runtime configs
├── data/                   # Local project data (gitignored except placeholders)
├── docs/                   # Notes, architecture docs, ADRs
├── notebooks/              # Exploration notebooks
├── scripts/                # CLI helpers for ingest, index, eval
├── src/graphlens/          # Main Python package (core logic)
├── tests/                  # Test suite
├── .env.example            # Environment variable template
├── .gitignore
└── pyproject.toml
```

## Quick Start

1. Create a virtual environment.
2. Install the project in editable mode: `pip install -e .`
3. Copy `.env.example` to `.env` and fill in provider keys.
4. Start a UI:
   - Streamlit: `streamlit run app/streamlit_app.py`
   - Gradio: `python app/gradio_app.py`

# server backend

FastAPI Backend (API)

The FastAPI backend exposes a simple RAG pipeline as HTTP endpoints so the frontend can call it directly.

Main endpoints (v1):

POST /api/v1/transcript → YouTube URL → timestamped transcript (normalized)

POST /api/v1/ingest → transcript/URL → chunks + embeddings → stored in vector DB

POST /api/v1/query → question → retrieves chunks from vector DB → returns answer

Run the API locally:

uvicorn server_backend.main:app --reload

Swagger docs:

http://127.0.0.1:8000/docs

Code layout:

server_backend/ → API layer (routes + schemas)

src/graphlens/ → core logic (transcripts/chunking/embeddings/vector store)