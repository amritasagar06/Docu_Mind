from pathlib import Path

import pymupdf
import pytest

from documind.ingestion import DocumentLoadError, load_document


def test_loads_markdown_text_and_headings(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Setup\nInstall the package.\n## Verify\nRun checks.\n", encoding="utf-8")

    [document] = load_document(path)

    assert document.text.startswith("# Setup")
    assert document.headings == ["Setup", "Verify"]
    assert document.source_name == "guide.md"
    assert document.source_uri == path.resolve().as_uri()
    assert document.content_type == "text/markdown"
    assert document.page_number is None


def test_loads_plain_text_and_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    [document] = load_document(path)

    assert document.text == ""
    assert document.content_type == "text/plain"


def test_loads_html_headings_and_omits_script_content(tmp_path: Path) -> None:
    path = tmp_path / "guide.html"
    path.write_text(
        "<html><body><h1>Install</h1><p>Use pip.</p><script>secret()</script></body></html>",
        encoding="utf-8",
    )

    [document] = load_document(path)

    assert document.headings == ["Install"]
    assert "Use pip." in document.text
    assert "secret()" not in document.text
    assert document.content_type == "text/html"


def test_loads_pdf_as_one_record_per_page(tmp_path: Path) -> None:
    path = tmp_path / "guide.pdf"
    pdf = pymupdf.open()
    for page_text in ("First page", "Second page"):
        page = pdf.new_page()
        page.insert_text((72, 72), page_text)
    path.write_bytes(pdf.tobytes())
    pdf.close()

    documents = load_document(path)

    assert [document.page_number for document in documents] == [1, 2]
    assert "First page" in documents[0].text
    assert "Second page" in documents[1].text
    assert all(document.content_type == "application/pdf" for document in documents)


def test_malformed_pdf_raises_informative_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a PDF")

    with pytest.raises(DocumentLoadError, match="Could not extract PDF"):
        load_document(path)


def test_encrypted_pdf_raises_informative_error(tmp_path: Path) -> None:
    path = tmp_path / "encrypted.pdf"
    pdf = pymupdf.open()
    pdf.new_page()
    path.write_bytes(
        pdf.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner-password",
            user_pw="user-password",
        )
    )
    pdf.close()

    with pytest.raises(DocumentLoadError, match="PDF is encrypted"):
        load_document(path)


def test_invalid_utf8_text_raises_informative_error(tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    path.write_bytes(b"valid text followed by invalid utf-8: \xff")

    with pytest.raises(DocumentLoadError, match="Could not read document"):
        load_document(path)


def test_missing_and_unsupported_sources_raise(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError, match="does not exist"):
        load_document(tmp_path / "missing.txt")

    unsupported = tmp_path / "archive.zip"
    unsupported.write_bytes(b"data")
    with pytest.raises(DocumentLoadError, match="Unsupported document type"):
        load_document(unsupported)
