"""Tests for Vector Store, indexing, duplicate prevention, and safe deletion."""

import tempfile
from pathlib import Path
from src.ingestion import PDFIngestionPipeline
from src.chunking import ContextAwareChunker
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = PROJECT_ROOT / "Input_Data" / "synthetic_autosar_hld.pdf"


def test_vector_store_indexing_and_duplicate_prevention():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        with open(SAMPLE_PDF, "rb") as f:
            pdf_bytes = f.read()

        doc_data = PDFIngestionPipeline().process_pdf(pdf_bytes, SAMPLE_PDF.name)
        chunks = ContextAwareChunker().chunk_document(doc_data)

        emb_mgr = EmbeddingManager()
        store = VectorStoreManager(persist_directory=tmpdir, embedding_manager=emb_mgr, collection_name="test_col")

        # First indexing
        indexed_count = store.index_chunks(chunks)
        assert indexed_count == len(chunks)
        assert store.is_document_indexed(doc_data.file_hash) is True

        # Second indexing of same document must be prevented
        duplicate_count = store.index_chunks(chunks)
        assert duplicate_count == 0, "Duplicate document indexing must be rejected"


def test_vector_store_retrieval_and_filtering():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        with open(SAMPLE_PDF, "rb") as f:
            pdf_bytes = f.read()

        doc_data = PDFIngestionPipeline().process_pdf(pdf_bytes, SAMPLE_PDF.name)
        chunks = ContextAwareChunker().chunk_document(doc_data)

        store = VectorStoreManager(persist_directory=tmpdir, collection_name="test_search")
        store.index_chunks(chunks)

        # Query relevant to ThrottleControl
        results = store.search(query="ThrottleControl_SWC accelerator pedal angle", top_k=2)
        assert len(results) > 0
        top_meta = results[0]["metadata"]
        assert top_meta["doc_id"] == doc_data.doc_id
        assert 1 <= int(top_meta["page_num"]) <= 6
        assert "similarity" in results[0]

        # Query with document filter
        filtered_results = store.search(
            query="BrakeManager", top_k=2, doc_id_filter=doc_data.doc_id
        )
        assert len(filtered_results) > 0
        assert filtered_results[0]["metadata"]["doc_id"] == doc_data.doc_id

        # Query with non-matching filter should return empty
        non_matching = store.search(query="Throttle", top_k=2, doc_id_filter="doc_non_existent")
        assert len(non_matching) == 0


def test_vector_store_safe_deletion():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        with open(SAMPLE_PDF, "rb") as f:
            pdf_bytes = f.read()

        doc_data = PDFIngestionPipeline().process_pdf(pdf_bytes, SAMPLE_PDF.name)
        chunks = ContextAwareChunker().chunk_document(doc_data)

        store = VectorStoreManager(persist_directory=tmpdir, collection_name="test_del")
        store.index_chunks(chunks)
        assert store.get_total_chunk_count() > 0

        # Delete document
        deleted = store.delete_document(doc_data.doc_id)
        assert deleted is True
        assert store.get_total_chunk_count() == 0
        assert store.is_document_indexed(doc_data.file_hash) is False
