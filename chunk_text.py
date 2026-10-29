
import json
from pathlib import Path

INPUT_PATH = Path("Data/processed/constitution_pages_clean.jsonl")
OUTPUT_PATH = Path("Data/processed/constitution_chunks.jsonl")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks, preferably at word boundaries."""
    text = text.strip()
    chunks = []

    if not text:
        return chunks

    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Avoid cutting a word in half when possible.
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - overlap

        # Ensure the loop always moves forward.
        start = max(next_start, start + 1)

    return chunks


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    total_chunks = 0

    with INPUT_PATH.open("r", encoding="utf-8") as source, \
         OUTPUT_PATH.open("w", encoding="utf-8") as output:

        for line in source:
            if not line.strip():
                continue

            page = json.loads(line)
            text = page.get("text", "")
            page_number = page["page_number"]

            chunks = split_text(text)

            for chunk_index, chunk in enumerate(chunks):
                record = {
                    "chunk_id": f"page_{page_number}_chunk_{chunk_index}",
                    "text": chunk,
                    "page_start": page_number,
                    "page_end": page_number,
                    "source_file": page.get(
                        "source_file", "constitution-of-india.pdf"
                    ),
                }

                output.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                total_chunks += 1

            total_pages += 1

    print("Pages processed:", total_pages)
    print("Chunks created:", total_chunks)
    print("Output file:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
