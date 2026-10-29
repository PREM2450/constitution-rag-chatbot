
import json
from pathlib import Path

import pymupdf

PDF_PATH = Path("Data/raw/constitution-of-india.pdf")
OUTPUT_PATH = Path("Data/processed/constitution_pages.jsonl")


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    empty_pages = []

    with pymupdf.open(PDF_PATH) as pdf, OUTPUT_PATH.open(
        "w", encoding="utf-8"
    ) as output_file:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            record = {
                "page_number": page_number,
                "text": text,
                "source_file": PDF_PATH.name,
            }

            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            total_pages += 1

            if not text:
                empty_pages.append(page_number)

    print(f"Total pages processed: {total_pages}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Empty pages: {len(empty_pages)}")

    if empty_pages:
        print(f"Pages needing inspection: {empty_pages}")


if __name__ == "__main__":
    main()
