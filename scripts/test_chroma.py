from dotenv import load_dotenv
load_dotenv()

from graphlens.vectorstores.chroma_v1 import ChromaStore

def main():
    store = ChromaStore(
        persist_path="data/chroma",
        collection_name="graphlens_chunks_test",
    )

    # Fake vectors (length 3 just for testing; in real life it will be 1536)
    ids = ["a", "b"]
    embeddings = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.29]]
    documents = ["hello world", "hello there"]
    metadatas = [
        {"video_id": "vid1", "start_seconds": 0.0, "end_seconds": 10.0},
        {"video_id": "vid1", "start_seconds": 10.0, "end_seconds": 20.0},
    ]

    store.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    # Query using a vector close to "b"
    res = store.query(query_embedding=[0.1, 0.2, 0.291], top_k=2, where={"video_id": "vid1"})
    print(res)

if __name__ == "__main__":
    main()