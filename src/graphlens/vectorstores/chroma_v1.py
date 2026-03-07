from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import chromadb


class ChromaStore:
    """
    Minimal persistent Chroma wrapper.
    Stores embeddings + documents + metadata for retrieval.
    """

    def __init__(
        self,
        persist_path: str = "data/chroma",
        collection_name: str = "graphlens_chunks",
    ):
        self.persist_path = os.getenv("CHROMA_PERSIST_PATH", persist_path)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=self.persist_path)

        # Prefer cosine space (newer and older Chroma versions differ slightly in config API)
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                configuration={"hnsw": {"space": "cosine"}},
            )
        except TypeError:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def delete_by_video_id(self, video_id: str) -> None:
        # Deletes all records where metadata.video_id == video_id
        self.collection.delete(where={"video_id": video_id})

    def upsert(
        self,
        *,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        *,
        query_embedding: List[float],
        top_k: int = 6,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )