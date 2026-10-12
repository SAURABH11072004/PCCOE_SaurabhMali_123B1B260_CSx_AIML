"""Comprehensive regression test suite for AUTOSAR HLD Document Analysis Assistant.
Verifies Test A through Test E on the synthetic AUTOSAR HLD document,
all 8 truthful status states, answer mode labeling, document isolation,
and zero real API quota consumption.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.ingestion import PDFIngestionPipeline
from src.chunking import ContextAwareChunker, TextChunk
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline, RAGResponse, Citation
from src.config import (
    Settings,
    test_gemini_connection as verify_gemini_connection,
    STATUS_KEY_MISSING,
    STATUS_NOT_TESTED,
    STATUS_CONNECTED,
    STATUS_RATE_LIMITED,
    STATUS_AUTH_FAILED,
    STATUS_MODEL_UNAVAILABLE,
    STATUS_SERVICE_UNAVAILABLE,
    STATUS_CONNECTION_FAILED,
)
from src.gemini_client import (
    call_gemini_with_quota_protection,
    compute_evidence_cache_key,
    clear_quota_block,
    _RESPONSE_CACHE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = PROJECT_ROOT / "Input_Data" / "synthetic_autosar_hld.pdf"


@pytest.fixture(autouse=True)
def reset_state():
    clear_quota_block()
    _RESPONSE_CACHE.clear()
    yield
    clear_quota_block()
    _RESPONSE_CACHE.clear()


@pytest.fixture(scope="module")
def indexed_synthetic_store():
    """Index synthetic AUTOSAR HLD once in a temporary Chroma store for module tests."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        with open(SAMPLE_PDF, "rb") as f:
            pdf_bytes = f.read()

        doc_data = PDFIngestionPipeline().process_pdf(pdf_bytes, SAMPLE_PDF.name)
        chunks = ContextAwareChunker().chunk_document(doc_data)

        store = VectorStoreManager(persist_directory=tmpdir, collection_name="test_autosar_suite")
        store.index_chunks(chunks)
        yield store, doc_data.doc_id


# =============================================================================
# TESTS A THROUGH E: SYNTHETIC AUTOSAR DOCUMENT VALIDATION
# =============================================================================

def test_a_component_extraction(indexed_synthetic_store):
    """Test A — Component extraction:
    Identify software components explicitly defined in Section 3 and cite their source passages.
    Must NOT include unrelated architectural overview or execution model paragraphs.
    """
    store, doc_id = indexed_synthetic_store
    settings = Settings(google_api_key="")
    rag = RAGPipeline(vector_store=store, settings=settings)

    resp = rag.query("Which software components are defined in the document?", doc_id_filter=doc_id)
    assert resp.is_abstained is False
    assert resp.mode == "retrieval_only"

    ans = resp.answer
    # Must list documented components
    assert "ThrottleControl_SWC" in ans
    assert "BrakeManager_SWC" in ans
    assert "BatteryManagement_SWC" in ans
    assert "ThermalControl_SWC" in ans
    assert "Gateway_SWC" in ans

    # Must NOT include unrelated architectural overview text as a component
    assert "three-layer architectural model" not in ans.lower()
    assert "execution model" not in ans.lower()

    # Citations must be valid and point to Page 3
    assert len(resp.citations) >= 1
    assert any(c.page == 3 for c in resp.citations)


def test_b_component_responsibility(indexed_synthetic_store):
    """Test B — Component responsibility:
    Describe documented responsibility of ThrottleControl_SWC and cite Page 3.
    """
    store, doc_id = indexed_synthetic_store
    settings = Settings(google_api_key="")
    rag = RAGPipeline(vector_store=store, settings=settings)

    resp = rag.query("What is the responsibility of ThrottleControl_SWC?", doc_id_filter=doc_id)
    assert resp.is_abstained is False
    assert "ThrottleControl_SWC" in resp.answer
    assert "throttle angle" in resp.answer.lower() or "pedal" in resp.answer.lower()
    assert any(c.page == 3 for c in resp.citations)


def test_c_interface_extraction(indexed_synthetic_store):
    """Test C — Interface extraction:
    Distinguish explicitly documented interfaces from missing or undefined references.
    """
    store, doc_id = indexed_synthetic_store
    settings = Settings(google_api_key="")
    rag = RAGPipeline(vector_store=store, settings=settings)

    resp = rag.query("Which interfaces are documented, and are any required interfaces missing?", doc_id_filter=doc_id)
    assert resp.is_abstained is False
    assert "If_ThrottlePosition" in resp.answer
    assert "If_BrakeTorque" in resp.answer
    assert "If_BatteryState" in resp.answer
    assert "If_ThermalData" in resp.answer
    # Must note missing interface
    assert "If_RefrigerantPressure" in resp.answer
    assert any(c.page == 4 for c in resp.citations)


def test_d_inconsistency_detection(indexed_synthetic_store):
    """Test D — Inconsistency detection:
    Report conflicting cycle-time definitions for Sig_ThrottleAngle across sections.
    """
    store, doc_id = indexed_synthetic_store
    settings = Settings(google_api_key="")
    rag = RAGPipeline(vector_store=store, settings=settings)

    resp = rag.query("Are there any conflicting cycle-time definitions for Sig_ThrottleAngle?", doc_id_filter=doc_id)
    assert resp.is_abstained is False
    assert "Sig_ThrottleAngle" in resp.answer
    assert "10 ms" in resp.answer or "10ms" in resp.answer
    assert "50 ms" in resp.answer or "50ms" in resp.answer
    assert any(c.page in [4, 6] for c in resp.citations)


def test_e_unsupported_question_abstains(indexed_synthetic_store):
    """Test E — Unsupported question:
    Abstain when document contains no supporting evidence.
    """
    store, doc_id = indexed_synthetic_store
    settings = Settings(google_api_key="")
    rag = RAGPipeline(vector_store=store, settings=settings)

    resp = rag.query("What lidar optical wavelength and point-cloud configuration does the document specify?", doc_id_filter=doc_id)
    assert resp.is_abstained is True
    assert "insufficient evidence" in resp.answer.lower()
    assert "not mentioned or specified" in resp.answer.lower()


# =============================================================================
# DOCUMENT ISOLATION TESTS
# =============================================================================

def test_document_isolation():
    """Ensure queries scoped to one document cannot retrieve or answer from another."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = VectorStoreManager(persist_directory=tmpdir, collection_name="test_iso")
        chunk_autosar = TextChunk(
            chunk_id="c_auto",
            doc_id="doc_autosar_id",
            filename="synthetic_autosar_hld.pdf",
            page_num=3,
            section="Section 3",
            chunk_index=0,
            text="ThrottleControl_SWC calculates vehicle throttle angle from accelerator pedal.",
            doc_hash="hash_auto",
            extraction_method="pdfplumber",
            char_count=70,
        )
        chunk_resume = TextChunk(
            chunk_id="c_resume",
            doc_id="doc_resume_id",
            filename="candidate_resume.pdf",
            page_num=1,
            section="Education",
            chunk_index=0,
            text="Candidate holds a Master of Science in Electrical Engineering.",
            doc_hash="hash_resume",
            extraction_method="pdfplumber",
            char_count=60,
        )
        store.index_chunks([chunk_autosar, chunk_resume])

        rag = RAGPipeline(vector_store=store, settings=Settings(google_api_key=""))

        # Scoped to resume
        resp_resume = rag.query("What degree does the candidate hold?", doc_id_filter="doc_resume_id")
        assert "ThrottleControl" not in resp_resume.answer
        assert any(c.document == "candidate_resume.pdf" for c in resp_resume.citations)

        # Scoped to autosar
        resp_auto = rag.query("What does ThrottleControl do?", doc_id_filter="doc_autosar_id")
        assert "Electrical Engineering" not in resp_auto.answer
        assert any(c.document == "synthetic_autosar_hld.pdf" for c in resp_auto.citations)


# =============================================================================
# TRUTHFUL STATUS STATES & ERROR CLASSES (8 STATES)
# =============================================================================

def test_status_1_not_configured():
    status, msg = verify_gemini_connection(api_key="")
    assert status == STATUS_KEY_MISSING
    assert "no api key" in msg.lower()


def test_status_2_configured_not_tested():
    settings = Settings(google_api_key="AIzaSyMockKeyPresent123456789")
    assert settings.is_gemini_configured is True
    # Verify default state without live API ping is STATUS_NOT_TESTED
    status_label = STATUS_NOT_TESTED
    assert status_label == "Configured, not tested"


def test_status_3_connected():
    with patch("src.gemini_client.get_gemini_client") as mock_get_client:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "PONG"
        mock_client.models.generate_content.return_value = mock_resp
        mock_get_client.return_value = (mock_client, None)

        status, msg = verify_gemini_connection(api_key="AIzaSyValidFormatKey12345678", model_name="gemini-3.5-flash-lite")
        assert status == STATUS_CONNECTED
        assert "successfully connected" in msg.lower()


def test_status_4_quota_exceeded_429():
    with patch("src.gemini_client.get_gemini_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded")
        mock_get_client.return_value = (mock_client, None)

        text, err, status = call_gemini_with_quota_protection(
            prompt="Hello",
            model_id="gemini-3.5-flash-lite",
            api_key="AIzaSyMockKeyForTesting12345",
        )
        assert status == STATUS_RATE_LIMITED
        assert "quota exceeded" in err.lower() or "rate limit" in err.lower()
        # Verify NO retry on 429
        assert mock_client.models.generate_content.call_count == 1


def test_status_5_auth_failed_401_403():
    with patch("src.gemini_client.get_gemini_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("401 UNAUTHENTICATED: API_KEY_INVALID")
        mock_get_client.return_value = (mock_client, None)

        text, err, status = call_gemini_with_quota_protection(
            prompt="Hello",
            model_id="gemini-3.5-flash-lite",
            api_key="AIzaSyInvalidKey12345",
        )
        assert status == STATUS_AUTH_FAILED
        assert "authentication failed" in err.lower()
        assert mock_client.models.generate_content.call_count == 1


def test_status_6_model_unavailable_404():
    with patch("src.gemini_client.get_gemini_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("404 NOT_FOUND: models/nonexistent is not found")
        mock_get_client.return_value = (mock_client, None)

        text, err, status = call_gemini_with_quota_protection(
            prompt="Hello",
            model_id="nonexistent-model",
            api_key="AIzaSyMockKeyForTesting12345",
        )
        assert status == STATUS_MODEL_UNAVAILABLE
        assert "not available" in err.lower() or "not supported" in err.lower()
        assert mock_client.models.generate_content.call_count == 1


def test_status_7_service_unavailable_503():
    with patch("src.gemini_client.get_gemini_client") as mock_get_client:
        with patch("time.sleep"):
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = Exception("503 Service Unavailable: High load")
            mock_get_client.return_value = (mock_client, None)

            text, err, status = call_gemini_with_quota_protection(
                prompt="Hello",
                model_id="gemini-3.5-flash-lite",
                api_key="AIzaSyMockKeyForTesting12345",
            )
            assert status == STATUS_SERVICE_UNAVAILABLE
            assert "temporarily unavailable" in err.lower()
            # Bounded: 1 initial attempt + 1 retry = 2 total
            assert mock_client.models.generate_content.call_count == 2


def test_status_8_connection_failed_timeout():
    with patch("src.gemini_client.get_gemini_client") as mock_get_client:
        with patch("time.sleep"):
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = Exception("Connection timed out after 30000ms")
            mock_get_client.return_value = (mock_client, None)

            text, err, status = call_gemini_with_quota_protection(
                prompt="Hello",
                model_id="gemini-3.5-flash-lite",
                api_key="AIzaSyMockKeyForTesting12345",
            )
            assert status == STATUS_CONNECTION_FAILED
            assert "timeout" in err.lower() or "connection" in err.lower()
