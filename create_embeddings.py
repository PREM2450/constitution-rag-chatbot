
import json
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

INPUT_PATH = Path("Data/processed/constitution_chunks.jsonl")
OUTPUT_PATH = Path("Data/processed/constitution_embeddings.jsonl")

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 8


def embedding_text(chunk):
    """Include structural labels so Article and Schedule queries affect retrieval."""
    metadata = " ".join(dict.fromkeys(
        value for value in (
            chunk.get("article"), chunk.get("section"), chunk.get("title")
        ) if value
    ))
    return f"{metadata}\n{chunk['text']}" if metadata else chunk["text"]


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    load_dotenv(Path(__file__).resolve().parent / ".env")
    model_name = os.getenv("EMBEDDING_MODEL", MODEL_NAME)
    device = os.getenv("RAG_DEVICE") or None
    print(f"Loading embedding model: {model_name} (device: {device or 'auto'})")
    model = SentenceTransformer(model_name, device=device)

    chunks = [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    texts = [embedding_text(chunk) for chunk in chunks]

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
                "embedding_model": model_name,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Chunks embedded:", len(embeddings))
    print("Embedding dimensions:", embeddings.shape[1])
    print("Output file:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
