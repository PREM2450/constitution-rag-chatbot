
import json
import re
from pathlib import Path

INPUT_PATH = Path("Data/processed/constitution_pages.jsonl")
OUTPUT_PATH = Path("Data/processed/constitution_pages_clean.jsonl")


def clean_text(text: str) -> str:
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove repeated spaces and tabs without merging words
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces around line breaks
    text = re.sub(r" *\n *", "\n", text)

    # Reduce excessive blank lines, preserving paragraph boundaries
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with INPUT_PATH.open("r", encoding="utf-8") as source, \
         OUTPUT_PATH.open("w", encoding="utf-8") as destination:

        for line in source:
            record = json.loads(line)
            record["text"] = clean_text(record["text"])

            destination.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            count += 1

    print(f"Pages cleaned: {count}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
