
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

INPUT_PATH = Path("Data/processed/constitution_chunks.jsonl")
OUTPUT_PATH = Path("Data/processed/constitution_embeddings.jsonl")

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 8


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    chunks = [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    texts = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for chunk, embedding in zip(chunks, embeddings):
            record = {
                **chunk,
                "embedding": embedding.tolist(),
                "embedding_model": MODEL_NAME,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Chunks embedded:", len(embeddings))
    print("Embedding dimensions:", embeddings.shape[1])
    print("Output file:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
