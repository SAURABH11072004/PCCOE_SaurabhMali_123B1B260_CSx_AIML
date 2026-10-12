"""Tests for context-aware chunking and metadata preservation."""

from pathlib import Path
from src.ingestion import PDFIngestionPipeline
from src.chunking import ContextAwareChunker, TextChunk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = PROJECT_ROOT / "Input_Data" / "synthetic_autosar_hld.pdf"


def test_chunking_metadata_and_stable_ids():
    with open(SAMPLE_PDF, "rb") as f:
        pdf_bytes = f.read()

    pipeline = PDFIngestionPipeline()
    doc_data = pipeline.process_pdf(pdf_bytes, SAMPLE_PDF.name)

    chunker = ContextAwareChunker(target_chunk_size=400, chunk_overlap=80)
    chunks = chunker.chunk_document(doc_data)

    assert len(chunks) > 0

    for idx, c in enumerate(chunks):
        assert isinstance(c, TextChunk)
        assert c.chunk_id.startswith(f"{doc_data.doc_id}_p")
        assert c.doc_id == doc_data.doc_id
        assert c.filename == SAMPLE_PDF.name
        assert 1 <= c.page_num <= doc_data.total_pages
        assert len(c.section) > 0
        assert c.chunk_index == idx
        assert c.doc_hash == doc_data.file_hash
        assert c.char_count > 0

        # Check ChromaDB-compatible metadata dictionary serialization
        meta = c.to_metadata()
        assert meta["chunk_id"] == c.chunk_id
        assert isinstance(meta["page_num"], int)
        assert isinstance(meta["chunk_index"], int)


def test_chunking_respects_sections():
    with open(SAMPLE_PDF, "rb") as f:
        pdf_bytes = f.read()

    doc_data = PDFIngestionPipeline().process_pdf(pdf_bytes, SAMPLE_PDF.name)
    chunker = ContextAwareChunker(target_chunk_size=600)
    chunks = chunker.chunk_document(doc_data)

    sections_found = {c.section for c in chunks}
    # Check that major section titles are retained in chunk metadata
    has_section = any("section" in s.lower() for s in sections_found)
    assert has_section, "Chunks should inherit section headers"
