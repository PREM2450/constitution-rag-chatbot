"""Extract numbered Constitution Articles from the cleaned, page-wise corpus."""

import json
import re
from pathlib import Path

INPUT_PATH = Path("Data/processed/constitution_pages_clean.jsonl")
OUTPUT_PATH = Path("Data/processed/constitution_articles.jsonl")
SOURCE_TITLE = "The Constitution of India"
SOURCE_URL = (
    "https://www.legislative.gov.in/static/uploads/2025/08/"
    "b9a933fb88c0a5afba254256bc55a3f0.pdf"
)
SOURCE_VERSION = "2026-09-10"
MAX_ARTICLE_NUMBER = 395

INLINE_HEADING = re.compile(r"^\s*[\W\d]*?(\d{1,3}[A-Z]?)\.\s*(.*)$")
NUMBER_ONLY = re.compile(r"^\s*\[?(\d{1,3}[A-Z]?)\.?\]?\s*$")
HEADING_DASH = re.compile(r"—|⎯|–|--")
PART_HEADING = re.compile(r"^\s*PART\s+([IVXLCDM]+A?)\b", re.IGNORECASE)
FIRST_SCHEDULE = re.compile(r"^\s*(?:\d+\[)?FIRST\s+SCHEDULE\b", re.IGNORECASE)
PREAMBLE_START = re.compile(r"\bWE,\s*THE PEOPLE OF INDIA\b", re.IGNORECASE)


def _heading_at(lines: list[str], index: int):
    line = lines[index]
    match = INLINE_HEADING.match(line)
    number_only = match is None
    if match:
        number, first_title_line = match.groups()
        title_lines = [first_title_line.strip()]
        scan_start = index
    else:
        number_match = NUMBER_ONLY.match(line)
        if number_match is None:
            return None
        number = number_match.group(1)
        title_lines = []
        scan_start = index + 1

    if int(number.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) > MAX_ARTICLE_NUMBER:
        return None

    for cursor in range(scan_start, min(scan_start + 12, len(lines))):
        candidate = lines[cursor]
        if number_only and (
            candidate.strip().upper() == "THE CONSTITUTION OF INDIA"
            or re.match(r"^\s*\(Part\s+[IVXLCDM]+\b", candidate, re.IGNORECASE)
        ):
            return None
        dash = HEADING_DASH.search(candidate)
        if dash:
            if cursor == index and INLINE_HEADING.match(line):
                inline = INLINE_HEADING.match(line)
                title_lines = [candidate[inline.start(2):dash.start()].strip()]
            else:
                before_dash = candidate[:dash.start()].strip()
                if before_dash:
                    title_lines.append(before_dash)
            body = candidate[dash.end():].strip()
            return {
                "number": number,
                "title": " ".join(part for part in title_lines if part).strip(),
                "body": body,
                "next_line": cursor + 1,
            }

        if cursor != index or not INLINE_HEADING.match(line):
            title_lines.append(candidate.strip())

    return None


def parse_articles(pages: list[dict]) -> list[dict]:
    """Parse ordered Article headings, ignoring the table of contents."""
    articles = []
    current = None
    current_part = None
    started = False
    seen_numbers = set()

    def save_current():
        if current is not None and current["text"].strip():
            current["text"] = current["text"].strip()
            articles.append(current.copy())

    def append_line(text: str, page_number: int):
        if current is None or not text.strip():
            return
        current["text"] += "\n" + text.strip()
        if page_number not in current["page_numbers"]:
            current["page_numbers"].append(page_number)

    for page in pages:
        page_number = page["page_number"]
        lines = page["text"].splitlines()
        if not started:
            if PREAMBLE_START.search(page["text"]):
                started = True
            continue

        index = 0
        while index < len(lines):
            line = lines[index].strip()
            part_match = PART_HEADING.match(line)
            if part_match:
                current_part = "Part " + part_match.group(1).upper()
                index += 1
                continue

            if (
                FIRST_SCHEDULE.match(line)
                and current is not None
                and current["article_number"] == "395"
            ):
                save_current()
                return articles

            heading = _heading_at(lines, index)
            if heading is None:
                if line.upper() != "THE CONSTITUTION OF INDIA" and not re.fullmatch(
                    r"\(?Part\s+[IVXLCDM]+\b.*\)?", line, re.IGNORECASE
                ):
                    append_line(line, page_number)
                index += 1
                continue

            if heading["number"] in seen_numbers:
                append_line(line, page_number)
                index += 1
                continue

            save_current()
            current = {
                "article_number": heading["number"],
                "article": f"Article {heading['number']}",
                "title": heading["title"],
                "text": heading["body"],
                "part": current_part,
                "page_numbers": [page_number],
                "source_file": "constitution-of-india.pdf",
                "source_title": SOURCE_TITLE,
                "source_url": SOURCE_URL,
                "source_version": SOURCE_VERSION,
            }
            seen_numbers.add(heading["number"])
            index = heading["next_line"]

    save_current()
    return articles


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    pages = [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    articles = parse_articles(pages)
    if not articles:
        raise ValueError("No Article headings were found; refusing to write an empty corpus.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for article in articles:
            output.write(json.dumps(article, ensure_ascii=False) + "\n")

    print("Articles extracted:", len(articles))
    print("Output file:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
