# Entry point for the FastAPI application.
# Creates the FastAPI app instance and mounts the top-level API router.
# Run via: uvicorn server_backend.main:app --reload
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server_backend.api.router import api_router


app = FastAPI(title="GraphLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/")
def root():
    return {
        "name": "GraphLens API",
        "base_url": "/api/v1",
        "docs": "/docs",
    }
