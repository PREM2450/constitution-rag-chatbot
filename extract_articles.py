
import json
import re
from pathlib import Path

INPUT_PATH = Path("Data/processed/constitution_pages_clean.jsonl")
OUTPUT_PATH = Path("Data/processed/constitution_articles.jsonl")

# Candidate headings such as:
# 1. Name and territory of the Union.—
# 21A. Right to education.—


ARTICLE_PATTERN = re.compile(
    r"(?m)(?<!\S)(\d{1,3}[A-Z]?)\.\s+"
    r"([^\n—]*(?:\n[^\n—]*){0,4}?)—"
)




def main():
    pages = [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
    ]

    articles = []
    current = None

    def save_current():
        if current is not None and current["text"].strip():
            current["text"] = current["text"].strip()
            articles.append(current.copy())

    for page in pages:
        text = page["text"]
        matches = list(ARTICLE_PATTERN.finditer(text))

        # If no heading is found, treat this page as continuation text.
        if not matches:
            if current is not None:
                current["text"] += "\n" + text
                current["page_numbers"].append(page["page_number"])
            continue

        # Preserve text before the first detected heading as continuation
        # of the previous Article, if one exists.
        prefix = text[:matches[0].start()].strip()
        if current is not None and prefix:
            current["text"] += "\n" + prefix
            current["page_numbers"].append(page["page_number"])

        for index, match in enumerate(matches):
            save_current()

            start = match.end()
            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(text)
            )

            current = {
                "article_number": match.group(1),
                "title": match.group(2).replace("\n", " ").strip(),
                "text": text[start:end].strip(),
                "page_numbers": [page["page_number"]],
                "source_file": page["source_file"],
            }

    save_current()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for article in articles:
            output.write(json.dumps(article, ensure_ascii=False) + "\n")

    print("Candidate records created:", len(articles))
    print("Output file:", OUTPUT_PATH)
    print("Note: this is still a candidate parser; Schedule headings and")
    print("PDF extraction errors need validation before using it for RAG.")


if __name__ == "__main__":
    main()
