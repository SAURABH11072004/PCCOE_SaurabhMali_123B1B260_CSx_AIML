"""Tests for DataManager, catalog persistence, and cross-tab data synchronization."""

import tempfile
from pathlib import Path
from src.ingestion import PDFIngestionPipeline
from src.chunking import ContextAwareChunker
from src.vector_store import VectorStoreManager
from src.data_manager import DataManager, classify_document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = PROJECT_ROOT / "Input_Data" / "synthetic_autosar_hld.pdf"


def test_classify_document():
    assert classify_document("synthetic_autosar_hld.pdf") == "Synthetic Test Data"
    assert classify_document("powertrain_swc_spec.pdf", "AUTOSAR SWC Layer") == "AUTOSAR Specification"
    assert classify_document("student_chapter_report.pdf", "Meeting minutes") == "General Document"


def test_data_manager_workflow():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        vstore = VectorStoreManager(persist_directory=tmpdir, collection_name="test_dm")
        dm = DataManager(vector_store=vstore)

        with open(SAMPLE_PDF, "rb") as f:
            pdf_bytes = f.read()

        doc_data = PDFIngestionPipeline().process_pdf(pdf_bytes, SAMPLE_PDF.name)
        chunks = ContextAwareChunker().chunk_document(doc_data)
        vstore.index_chunks(chunks)

        ents = dm.entity_extractor.extract_from_chunks(chunks)
        issues = dm.issue_detector.detect_issues(doc_data, chunks, ents)

        # Register document
        dm.register_document(doc_data, chunks, ents, issues)

        # Check summary
        summaries = dm.get_documents_summary()
        assert len(summaries) >= 1
        doc_entry = next(d for d in summaries if d["doc_id"] == doc_data.doc_id)
        assert doc_entry["filename"] == SAMPLE_PDF.name
        assert doc_entry["doc_type"] == "Synthetic Test Data"

        # Check entities
        entities_retrieved = dm.get_entities(doc_id_filter=doc_data.doc_id)
        assert len(entities_retrieved) == len(ents)
        assert len(entities_retrieved) > 0

        # Check issues
        issues_retrieved = dm.get_issues(doc_id_filter=doc_data.doc_id)
        assert len(issues_retrieved) == len(issues)
        assert len(issues_retrieved) > 0

        # Check filter with non-matching ID
        assert len(dm.get_entities(doc_id_filter="doc_non_existent")) == 0
        assert len(dm.get_issues(doc_id_filter="doc_non_existent")) == 0

        # Check document removal
        removed = dm.remove_document(doc_data.doc_id)
        assert removed is True
        assert len(dm.get_entities(doc_id_filter=doc_data.doc_id)) == 0
        assert len(dm.get_issues(doc_id_filter=doc_data.doc_id)) == 0


def test_ensure_baseline_document_indexes_when_empty():
    """Test automatic indexing of baseline document into an empty library."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        vstore = VectorStoreManager(persist_directory=tmpdir, collection_name="test_auto_init")
        dm = DataManager(vector_store=vstore)
        chunker = ContextAwareChunker()

        assert len(dm.get_documents_summary()) == 0
        assert vstore.get_total_chunk_count() == 0

        # Run auto-initialization
        result = dm.ensure_baseline_document(chunker=chunker, baseline_pdf_path=SAMPLE_PDF)
        assert result is True

        # Verify document is now in catalog
        summaries = dm.get_documents_summary()
        assert len(summaries) == 1
        assert summaries[0]["filename"] == "synthetic_autosar_hld.pdf"
        assert summaries[0]["chunk_count"] == 6

        # Verify chunks in vector store
        assert vstore.get_total_chunk_count() == 6

        # Verify entities and issues extracted
        entities = dm.get_entities()
        assert len(entities) > 0
        swc_names = {e.name for e in entities}
        assert "ThrottleControl_SWC" in swc_names

        issues = dm.get_issues()
        assert len(issues) > 0

        # Verify search functionality and citation retrieval
        search_results = vstore.search("ThrottleControl_SWC", top_k=4)
        assert len(search_results) > 0
        assert any(r["metadata"]["page_num"] == 3 for r in search_results)


def test_ensure_baseline_document_idempotent_no_duplicates():
    """Test that repeated restarts/calls do not duplicate chunks or catalog entries."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        vstore = VectorStoreManager(persist_directory=tmpdir, collection_name="test_idempotent")
        dm = DataManager(vector_store=vstore)
        chunker = ContextAwareChunker()

        # First call: indexes document
        assert dm.ensure_baseline_document(chunker=chunker, baseline_pdf_path=SAMPLE_PDF) is True
        assert vstore.get_total_chunk_count() == 6
        assert len(dm.get_documents_summary()) == 1

        # Second call: should be a no-op and not create duplicates
        assert dm.ensure_baseline_document(chunker=chunker, baseline_pdf_path=SAMPLE_PDF) is True
        assert vstore.get_total_chunk_count() == 6
        assert len(dm.get_documents_summary()) == 1

        # Third call: simulate restart
        vstore2 = VectorStoreManager(persist_directory=tmpdir, collection_name="test_idempotent")
        dm2 = DataManager(vector_store=vstore2)
        assert dm2.ensure_baseline_document(chunker=chunker, baseline_pdf_path=SAMPLE_PDF) is True
        assert vstore2.get_total_chunk_count() == 6
        assert len(dm2.get_documents_summary()) == 1


def test_ensure_baseline_document_missing_file_handling():
    """Test that a missing baseline file returns False without fabricating data or crashing."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        vstore = VectorStoreManager(persist_directory=tmpdir, collection_name="test_missing")
        dm = DataManager(vector_store=vstore)
        non_existent = Path(tmpdir) / "does_not_exist.pdf"

        result = dm.ensure_baseline_document(baseline_pdf_path=non_existent)
        assert result is False
        assert len(dm.get_documents_summary()) == 0
        assert vstore.get_total_chunk_count() == 0
        assert not non_existent.exists()


def test_removal_and_safe_recovery():
    """Test removing document via UI delete workflow and re-indexing safely."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        vstore = VectorStoreManager(persist_directory=tmpdir, collection_name="test_remove_recover")
        dm = DataManager(vector_store=vstore)
        chunker = ContextAwareChunker()

        # Initialize
        dm.ensure_baseline_document(chunker=chunker, baseline_pdf_path=SAMPLE_PDF)
        assert len(dm.get_documents_summary()) == 1
        doc_id = dm.get_documents_summary()[0]["doc_id"]

        # User deletes document through UI
        deleted = dm.remove_document(doc_id)
        assert deleted is True
        assert len(dm.get_documents_summary()) == 0
        assert vstore.get_total_chunk_count() == 0

        # Safe recovery / re-index
        recovered = dm.ensure_baseline_document(chunker=chunker, baseline_pdf_path=SAMPLE_PDF, force=True)
        assert recovered is True
        assert len(dm.get_documents_summary()) == 1
        assert vstore.get_total_chunk_count() == 6

