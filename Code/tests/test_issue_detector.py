"""Tests for potential issue and architectural inconsistency detection."""

from pathlib import Path
from src.ingestion import PDFIngestionPipeline
from src.chunking import ContextAwareChunker
from src.entity_extractor import EntityExtractor
from src.issue_detector import IssueDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = PROJECT_ROOT / "Input_Data" / "synthetic_autosar_hld.pdf"


def test_issue_detection_finds_deliberate_inconsistencies():
    with open(SAMPLE_PDF, "rb") as f:
        doc = PDFIngestionPipeline().process_pdf(f.read(), SAMPLE_PDF.name)

    chunks = ContextAwareChunker().chunk_document(doc)
    extractor = EntityExtractor()
    entities = extractor.extract_from_chunks(chunks)

    detector = IssueDetector()
    issues = detector.detect_issues(doc, chunks, entities)

    assert len(issues) > 0

    issue_types = {i.issue_type for i in issues}
    # Deliberately injected issues must be flagged:
    # 1. Conflicting Definition: Sig_ThrottleAngle 10ms vs 50ms
    # 2. Undefined Interface: If_RefrigerantPressure
    # Verify deduplication: no duplicate (type, description) pairs
    unique_keys = {(i.issue_type, i.description) for i in issues}
    assert len(unique_keys) == len(issues), "Issues must be deduplicated"

    # Verify DataFrame conversion
    df = detector.issues_to_dataframe(issues)
    assert len(df) == len(issues)
    assert "Issue ID" in df.columns
    assert "Severity" in df.columns
    assert "Supporting Evidence" in df.columns
