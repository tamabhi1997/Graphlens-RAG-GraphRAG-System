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
├── config/                 # YAML / TOML runtime configs
├── data/                   # Local project data (gitignored except placeholders)
├── docs/                   # Notes, architecture docs, ADRs
├── notebooks/              # Exploration notebooks
├── scripts/                # CLI helpers for ingest, index, eval
├── src/graphlens/          # Main Python package
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
