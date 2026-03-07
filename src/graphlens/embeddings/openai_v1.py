# from __future__ import annotations

# import os
# from typing import List

# from openai import OpenAI


# class OpenAIEmbedder:
#     """
#     OpenAI embedding wrapper.
#     - Model defaults to EMBEDDING_MODEL env var if set, else 'text-embedding-3-small'
#     - Keeps embedding logic isolated from pipelines/endpoints
#     """

#     def __init__(self, model: str | None = None):
#         env_model = os.getenv("EMBEDDING_MODEL")
#         self.model = model or env_model or "text-embedding-3-small"

#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env and load_dotenv().")

#         self.client = OpenAI(api_key=api_key)

#     def embed_texts(self, texts: List[str]) -> List[List[float]]:
#         # Batch embed: one call for many texts
#         resp = self.client.embeddings.create(model=self.model, input=texts)
#         return [item.embedding for item in resp.data]

#     def embed_query(self, text: str) -> List[float]:
#         return self.embed_texts([text])[0]

import os
from openai import OpenAI

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def embed_texts(texts):
    """
    Input:  list[str]
    Output: list[list[float]]
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Make sure load_dotenv() runs before calling embed_texts()."
        )

    client = OpenAI(api_key=api_key)  # create client only when needed

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


def embed_query(text):
    return embed_texts([text])[0]