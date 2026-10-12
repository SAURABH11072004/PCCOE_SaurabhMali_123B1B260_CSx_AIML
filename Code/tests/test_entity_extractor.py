"""Tests for Architectural Entity Extractor and Pydantic validation."""

from pathlib import Path
from src.ingestion import PDFIngestionPipeline
from src.chunking import ContextAwareChunker
from src.entity_extractor import EntityExtractor, ArchitecturalEntity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = PROJECT_ROOT / "Input_Data" / "synthetic_autosar_hld.pdf"


def test_entity_extraction_from_pdf():
    with open(SAMPLE_PDF, "rb") as f:
        pdf_bytes = f.read()

    doc_data = PDFIngestionPipeline().process_pdf(pdf_bytes, SAMPLE_PDF.name)
    chunks = ContextAwareChunker().chunk_document(doc_data)

    extractor = EntityExtractor()
    entities = extractor.extract_from_chunks(chunks)

    assert len(entities) > 0

    types_found = {e.type for e in entities}
    assert "Component" in types_found
    assert "Port" in types_found
    assert "Signal" in types_found
    assert "Interface" in types_found

    # Check specific component presence
    comp_names = {e.name for e in entities if e.type == "Component"}
    assert "ThrottleControl_SWC" in comp_names
    assert "BrakeManager_SWC" in comp_names

    # Check port presence
    port_names = {e.name for e in entities if e.type == "Port"}
    assert "PpThrottlePos" in port_names or "RpPedalSensor" in port_names

    # Verify DataFrame conversion
    df = extractor.entities_to_dataframe(entities)
    assert len(df) == len(entities)
    assert "Name" in df.columns
    assert "Type" in df.columns
    assert "Page" in df.columns
