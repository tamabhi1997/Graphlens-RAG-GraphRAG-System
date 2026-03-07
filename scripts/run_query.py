from dotenv import load_dotenv
load_dotenv()

from graphlens.embeddings.openai_v1 import embed_query
from graphlens.vectorstores.chroma_v1 import ChromaStore


def main():
    video_id = "alfdI7S6wCY"          # the one you ingested
    collection_name = "graphlens_chunks"
    question = "What is deep learning and how is it different from AI and machine learning?"

    top_k = 6
    min_similarity = 0.25  # refusal threshold (we can tune)

    # 1) Embed the question
    qvec = embed_query(question)

    # 2) Query Chroma (filter to this video)
    store = ChromaStore(persist_path="data/chroma", collection_name=collection_name)
    res = store.query(query_embedding=qvec, top_k=top_k, where={"video_id": video_id})

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    # 3) Convert distance -> similarity
    sims = [1.0 - float(d) for d in dists]
    best_sim = sims[0] if sims else None

    print("\nQUERY RESULT")
    print("question:", question)
    print("video_id:", video_id)
    print("top_k:", top_k)
    print("best_similarity:", best_sim)

    # 4) Refusal gate
    if best_sim is None or best_sim < min_similarity:
        print("\nREFUSED ❌")
        print("Reason: Not enough relevant evidence in indexed content.")
        return

    print("\nACCEPTED ✅ Top matches:\n")
    for i, (cid, sim, doc, meta) in enumerate(zip(ids, sims, docs, metas)):
        start = meta.get("start_seconds")
        end = meta.get("end_seconds")
        preview = doc[:220].replace("\n", " ")
        print(f"[{i}] chunk_id={cid}  sim={sim:.3f}  {start:.2f}s -> {end:.2f}s")
        print("    preview:", preview)
        print()


if __name__ == "__main__":
    main()