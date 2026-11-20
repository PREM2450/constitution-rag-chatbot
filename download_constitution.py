"""Download the verified 2026 English Constitution PDF from the Legislative Department."""

import hashlib
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

PDF_URL = (
    "https://www.legislative.gov.in/static/uploads/2025/08/"
    "b9a933fb88c0a5afba254256bc55a3f0.pdf"
)
EXPECTED_SHA256 = "c2e0f5f7a670d362038190d6792a688a37c25b3918b62f924cb7fe7444ae2608"
OUTPUT_PATH = Path("Data/raw/constitution-of-india.pdf")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_pdf(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        actual = _sha256(output_path)
        if actual != EXPECTED_SHA256:
            raise FileExistsError(
                f"{output_path} already exists but does not match the verified source; "
                "move it aside before downloading a different version."
            )
        print(f"Verified Constitution PDF already exists: {output_path}")
        return output_path

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".constitution-", dir=output_path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            request = Request(PDF_URL, headers={"User-Agent": "Constitution-RAG/1.0"})
            with urlopen(request, timeout=60) as response:
                if response.status != 200:
                    raise OSError(f"Official PDF download returned HTTP {response.status}.")
                while block := response.read(1024 * 1024):
                    temporary.write(block)

        actual = _sha256(temporary_path)
        if actual != EXPECTED_SHA256:
            raise ValueError("Downloaded PDF failed the verified SHA-256 check.")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"Downloaded and verified official Constitution PDF: {output_path}")
    return output_path


if __name__ == "__main__":
    download_pdf()
