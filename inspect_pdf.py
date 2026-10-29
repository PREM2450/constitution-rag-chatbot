
from pathlib import Path
import pymupdf

PDF_PATH = Path("Data/raw/constitution-of-india.pdf")


def main():
    with pymupdf.open(PDF_PATH) as pdf:
        for page_number, page in enumerate(pdf):
            text = page.get_text("text")
            normalized = " ".join(text.split()).lower()

            if "equality before law" in normalized:
                print(f"\nPDF page: {page_number + 1}")
                print(text[:2500])
                print("-" * 50)


if __name__ == "__main__":
    main()
