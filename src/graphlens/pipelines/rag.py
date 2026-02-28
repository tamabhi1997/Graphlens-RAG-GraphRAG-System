"""RAG pipeline scaffold."""

from graphlens.retrievers.hybrid import retrieve_context


def answer_query(query: str) -> str:
    context = retrieve_context(query)
    return f"Query: {query}\n\nRetrieved context:\n{context}"
