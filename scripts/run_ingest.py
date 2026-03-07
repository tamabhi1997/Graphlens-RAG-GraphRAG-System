from dotenv import load_dotenv
load_dotenv()

from graphlens.pipelines.ingest_youtube_v1 import ingest_youtube_url_v1


def main():
    url = "https://www.youtube.com/watch?v=alfdI7S6wCY"  # change to any video

    result = ingest_youtube_url_v1(
        url=url,
        languages=["en"],
        collection_name="graphlens_chunks",
        force_reindex=True,
        chunk_cfg={
            "max_seconds": 135.0,
            "max_chars": 2200,
            "min_chars": 1200,
            "overlap_chars": 250,
        },
        batch_size=64,
    )

    print("\nINDEXING DONE ✅")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
