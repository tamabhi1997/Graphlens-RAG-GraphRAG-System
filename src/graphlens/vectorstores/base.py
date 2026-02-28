"""Base interface for vector store clients."""

from typing import Protocol


class VectorStoreClient(Protocol):
    def upsert(self, documents: list[dict]) -> None:
        ...

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        ...
