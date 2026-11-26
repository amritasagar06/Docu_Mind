"""Load supported local document formats without hiding extraction errors."""

from pathlib import Path
import re

import pymupdf
from bs4 import BeautifulSoup

from documind.ingestion.models import Document


class DocumentLoadError(RuntimeError):
    """Raised when a source cannot be read or extracted."""


_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise DocumentLoadError(f"Could not read document '{path}': {exc}") from exc


def _load_markdown_or_text(path: Path) -> list[Document]:
    text = _read_text(path)
    headings = [match.group(1).strip() for line in text.splitlines() if (match := _MARKDOWN_HEADING.match(line))]
    return [Document.from_path(path, text, headings=headings, content_type="text/markdown" if path.suffix.lower() == ".md" else "text/plain")]


def _load_html(path: Path) -> list[Document]:
    source = _read_text(path)
    try:
        soup = BeautifulSoup(source, "html.parser")
        headings = [node.get_text(" ", strip=True) for node in soup.find_all(re.compile(r"^h[1-6]$"))]
        for node in soup(["script", "style", "noscript"]):
            node.decompose()
        text = soup.get_text("\n", strip=True)
    except Exception as exc:
        raise DocumentLoadError(f"Could not extract HTML document '{path}': {exc}") from exc
    return [Document.from_path(path, text, headings=headings, content_type="text/html")]


def _load_pdf(path: Path) -> list[Document]:
    try:
        with pymupdf.open(path) as pdf:
            if pdf.is_encrypted:
                raise DocumentLoadError(f"PDF is encrypted and cannot be read without a password: '{path}'")
            records = [
                Document.from_path(path, page.get_text("text"), page_number=index + 1, content_type="application/pdf")
                for index, page in enumerate(pdf)
            ]
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(f"Could not extract PDF document '{path}': {exc}") from exc
    return records


def load_document(path: str | Path) -> list[Document]:
    """Extract a supported document. PDF pages are returned as separate records.

    Raises:
        DocumentLoadError: if the source is missing, unsupported, or malformed.
    """
    source = Path(path)
    if not source.is_file():
        raise DocumentLoadError(f"Document does not exist or is not a file: '{source}'")

    suffix = source.suffix.lower()
    if suffix in {".md", ".txt"}:
        return _load_markdown_or_text(source)
    if suffix in {".htm", ".html"}:
        return _load_html(source)
    if suffix == ".pdf":
        return _load_pdf(source)
    raise DocumentLoadError(f"Unsupported document type '{suffix or '(no extension)'}': '{source}'")
