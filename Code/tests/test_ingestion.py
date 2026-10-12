"""Tests for PDF Ingestion, checksumming, validation, and OCR fallback."""

import pytest
from pathlib import Path
from src.ingestion import (
    PDFIngestionPipeline,
    compute_file_hash,
    normalize_text,
    extract_headings,
    check_tesseract_available,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = PROJECT_ROOT / "Input_Data" / "synthetic_autosar_hld.pdf"


def test_compute_file_hash():
    data = b"AUTOSAR Classic Platform Test"
    h1 = compute_file_hash(data)
    h2 = compute_file_hash(data)
    assert h1 == h2
    assert len(h1) == 64


def test_normalize_text():
    raw = "Section 1:\tExecutive\u00a0Summary\r\n\r\n\r\n\r\nDetails here   with spaces."
    clean = normalize_text(raw)
    assert "\u00a0" not in clean
    assert "\r" not in clean
    assert "\n\n\n" not in clean
    assert "Details here with spaces." in clean


def test_extract_headings():
    sample_text = (
        "Section 1: Executive Summary & Architectural Scope\n"
        "Some body text...\n"
        "Section 3: Software Components (SWCs) and Port Definitions\n"
        "Details...\n"
    )
    headings = extract_headings(sample_text)
    assert len(headings) >= 2
    assert "Section 1: Executive Summary & Architectural Scope" in headings


def test_ingest_valid_pdf():
    assert SAMPLE_PDF.exists(), "Sample PDF must exist for ingestion test"
    with open(SAMPLE_PDF, "rb") as f:
        pdf_bytes = f.read()

    pipeline = PDFIngestionPipeline(enable_ocr=False)
    doc_data = pipeline.process_pdf(pdf_bytes, SAMPLE_PDF.name)

    assert doc_data.doc_id.startswith("doc_")
    assert doc_data.filename == SAMPLE_PDF.name
    assert doc_data.total_pages == 6
    assert len(doc_data.pages) == 6

    # Verify page metadata preserved
    page1 = doc_data.pages[0]
    assert page1.page_number == 1
    assert "Section 1: Executive Summary" in page1.text
    assert page1.char_count > 100
    assert page1.extraction_method == "pdf_text"


def test_ingest_empty_bytes_raises():
    pipeline = PDFIngestionPipeline()
    with pytest.raises(ValueError, match="is empty"):
        pipeline.process_pdf(b"", "empty.pdf")


def test_ingest_invalid_pdf_signature_raises():
    pipeline = PDFIngestionPipeline()
    with pytest.raises(ValueError, match="not a valid PDF"):
        pipeline.process_pdf(b"NOT A REAL PDF FILE HEADER", "corrupt.pdf")


def test_ocr_availability_check():
    # Should safely return boolean without throwing
    is_avail = check_tesseract_available()
    assert isinstance(is_avail, bool)
