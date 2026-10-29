
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDINGS_PATH = Path("Data/processed/constitution_embeddings.jsonl")
MODEL_NAME = "BAAI/bge-m3"
TOP_K = 5


def load_chunks():
    with EMBEDDINGS_PATH.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_PATH}"
        )

    print("Loading Constitution embeddings...")
    chunks = load_chunks()

    vectors = np.array(
        [chunk["embedding"] for chunk in chunks],
        dtype=np.float32,
    )

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("\nConstitution Semantic Search")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        query_vector = model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        # Embeddings are normalized, so dot product = cosine similarity.
        scores = vectors @ query_vector
        top_indices = np.argsort(scores)[::-1][:TOP_K]

        print("\nMost relevant passages:\n")

        for rank, index in enumerate(top_indices, start=1):
            chunk = chunks[index]

            print(f"--- Result {rank} ---")
            print(f"Similarity: {scores[index]:.4f}")
            print(
                f"PDF page: {chunk['page_start']}"
                f" | Chunk ID: {chunk['chunk_id']}"
            )
            print(chunk["text"][:1200])
            print()


if __name__ == "__main__":
    main()
