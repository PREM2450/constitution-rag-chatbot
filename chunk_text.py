
import json
import re
from pathlib import Path

from extract_articles import SOURCE_TITLE, SOURCE_URL, SOURCE_VERSION

INPUT_PATH = Path("Data/processed/constitution_articles.jsonl")
PAGES_PATH = Path("Data/processed/constitution_pages_clean.jsonl")
OUTPUT_PATH = Path("Data/processed/constitution_chunks.jsonl")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
SCHEDULE_HEADING = re.compile(
    r"^\s*(?:\d+\[)?(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|"
    r"EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH)\s+SCHEDULE\b",
    re.IGNORECASE,
)
APPENDIX_HEADING = re.compile(r"^\s*(?:\d+\[)?APPENDIX\b", re.IGNORECASE)


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


def build_chunks(articles, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split Articles while preserving their identifiers and source metadata."""
    records = []
    for article in articles:
        page_numbers = sorted(set(article.get("page_numbers", [])))
        page_start = page_numbers[0] if page_numbers else None
        page_end = page_numbers[-1] if page_numbers else None

        for chunk_index, text in enumerate(
            split_text(article["text"], chunk_size=chunk_size, overlap=overlap)
        ):
            article_number = article["article_number"]
            records.append({
                "chunk_id": f"article_{article_number}_chunk_{chunk_index}",
                "text": text,
                "article": article.get("article", f"Article {article_number}"),
                "article_number": article_number,
                "title": article.get("title", ""),
                "part": article.get("part"),
                "page_start": page_start,
                "page_end": page_end,
                "source_file": article.get("source_file", "constitution-of-india.pdf"),
                "source_title": article.get("source_title", "The Constitution of India"),
                "source_url": article.get("source_url", ""),
                "source_version": article.get("source_version", "unknown"),
            })
    return records


def build_supplement_chunks(
    pages, after_page, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP
):
    """Keep schedules and appendices searchable as page-scoped chunks."""
    records = []
    section = "Schedules"
    for page in pages:
        page_number = int(page["page_number"])
        if page_number <= after_page:
            continue

        text = page["text"].strip()
        if not text:
            continue
        for line in text.splitlines():
            schedule = SCHEDULE_HEADING.match(line)
            if schedule:
                section = f"{schedule.group(1).title()} Schedule"
            elif APPENDIX_HEADING.match(line):
                section = "Appendices"

        for chunk_index, chunk in enumerate(
            split_text(text, chunk_size=chunk_size, overlap=overlap)
        ):
            records.append({
                "chunk_id": f"supplement_page_{page_number}_chunk_{chunk_index}",
                "text": chunk,
                "article": None,
                "article_number": None,
                "title": section,
                "part": None,
                "section": section,
                "page_start": page_number,
                "page_end": page_number,
                "source_file": "constitution-of-india.pdf",
                "source_title": SOURCE_TITLE,
                "source_url": SOURCE_URL,
                "source_version": SOURCE_VERSION,
            })
    return records


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    if not PAGES_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {PAGES_PATH}")

    articles = [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pages = [
        json.loads(line)
        for line in PAGES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    article_end_page = max(
        (page for article in articles for page in article.get("page_numbers", [])),
        default=0,
    )
    chunks = build_chunks(articles)
    chunks.extend(build_supplement_chunks(pages, after_page=article_end_page))
    if not chunks:
        raise ValueError("No Article chunks were produced; refusing to write an empty corpus.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("Articles processed:", len(articles))
    print("Chunks created:", len(chunks))
    print("Output file:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
