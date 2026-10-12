"""Tests for document revision comparison."""

from pathlib import Path
from src.ingestion import PDFIngestionPipeline
from src.chunking import ContextAwareChunker
from src.entity_extractor import EntityExtractor
from src.doc_comparison import DocumentComparator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_V1 = PROJECT_ROOT / "Input_Data" / "synthetic_autosar_hld.pdf"
PDF_V2 = PROJECT_ROOT / "Input_Data" / "synthetic_autosar_hld_v2.pdf"


def test_revision_comparison_v1_vs_v2():
    with open(PDF_V1, "rb") as f:
        doc1 = PDFIngestionPipeline().process_pdf(f.read(), PDF_V1.name)
    with open(PDF_V2, "rb") as f:
        doc2 = PDFIngestionPipeline().process_pdf(f.read(), PDF_V2.name)

    chunks1 = ContextAwareChunker().chunk_document(doc1)
    chunks2 = ContextAwareChunker().chunk_document(doc2)

    extractor = EntityExtractor()
    ent1 = extractor.extract_from_chunks(chunks1)
    ent2 = extractor.extract_from_chunks(chunks2)

    comparator = DocumentComparator()
    report = comparator.compare_documents(doc1, doc2, ent1, ent2)

    assert report.doc_a_name == PDF_V1.name
    assert report.doc_b_name == PDF_V2.name
    assert len(report.section_diffs) > 0

    # Revision 2 adds Section 7 (Cyber Security)
    added_sections = [s for s in report.section_diffs if s.change_type == "ADDED"]
    assert len(added_sections) >= 1

    # SecurityManager_SWC should be detected as added entity
    added_entities = [e for e in report.entity_diffs if e.change_type == "ADDED"]
    assert any("SecurityManager_SWC" in e.entity_name for e in added_entities)

    # Check DataFrame export
    df = comparator.diff_to_dataframe(report.section_diffs)
    assert len(df) == len(report.section_diffs)
    assert "Section" in df.columns
