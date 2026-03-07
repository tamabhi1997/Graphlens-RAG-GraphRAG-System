# Entry point for the FastAPI application.
# Creates the FastAPI app instance and mounts the top-level API router.
# Run via: uvicorn server_backend.main:app --reload
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from server_backend.api.router import api_router


app = FastAPI(title="GraphLens API")
app.include_router(api_router, prefix="/api")