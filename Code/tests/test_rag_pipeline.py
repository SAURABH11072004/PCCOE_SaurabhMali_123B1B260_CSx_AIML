"""Tests for RAG pipeline, grounding, citation validation, query validation,
semantic intent mapping, and deterministic evidence-extraction fallback.
"""

import tempfile
from pathlib import Path
from src.ingestion import PDFIngestionPipeline
from src.chunking import ContextAwareChunker, TextChunk
from src.vector_store import VectorStoreManager
from src.rag_pipeline import (
    RAGPipeline,
    validate_user_query,
    get_contextual_suggestions,
)
from src.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = PROJECT_ROOT / "Input_Data" / "synthetic_autosar_hld.pdf"


def make_test_chunk(cid: str, doc_id: str, fname: str, page: int, section: str, text: str) -> TextChunk:
    return TextChunk(
        chunk_id=cid,
        doc_id=doc_id,
        filename=fname,
        page_num=page,
        section=section,
        chunk_index=0,
        text=text,
        doc_hash=f"hash_{cid}_{doc_id}",
        extraction_method="pdfplumber",
        char_count=len(text),
    )


def test_query_validation():
    # Empty query
    is_valid, err = validate_user_query("")
    assert is_valid is False
    assert "enter a question" in err.lower()

    # Short incomplete query
    is_valid, err = validate_user_query("what is")
    assert is_valid is False
    assert "at least 3 words" in err.lower()

    # Valid query
    is_valid, err = validate_user_query("What is the function of ThrottleControl_SWC?")
    assert is_valid is True
    assert err is None


def test_contextual_suggestions():
    # AUTOSAR document
    autosar_suggs = get_contextual_suggestions("synthetic_autosar_hld.pdf", "AUTOSAR Specification")
    assert len(autosar_suggs) >= 3
    assert any("ThrottleControl_SWC" in s for s in autosar_suggs)

    # General ACM document
    acm_suggs = get_contextual_suggestions("ACM student chapter.pdf", "General Document")
    assert len(acm_suggs) >= 2
    assert any("activities" in s.lower() or "workshops" in s.lower() for s in acm_suggs)


def test_rag_retrieval_and_citations():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        with open(SAMPLE_PDF, "rb") as f:
            pdf_bytes = f.read()

        doc_data = PDFIngestionPipeline().process_pdf(pdf_bytes, SAMPLE_PDF.name)
        chunks = ContextAwareChunker().chunk_document(doc_data)

        store = VectorStoreManager(persist_directory=tmpdir, collection_name="test_rag")
        store.index_chunks(chunks)

        test_settings = Settings(google_api_key="", gemini_model="gemini-1.5-flash")
        rag = RAGPipeline(vector_store=store, settings=test_settings)

        resp = rag.query("What is the function of ThrottleControl_SWC?")
        assert resp.mode == "retrieval_only"
        assert len(resp.retrieved_chunks) > 0
        assert len(resp.citations) > 0
        assert resp.is_abstained is False

        for c in resp.citations:
            assert c.document == SAMPLE_PDF.name
            assert 1 <= c.page <= 6
            assert c.is_valid is True


def test_rag_abstention_on_unsupported_query():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        with open(SAMPLE_PDF, "rb") as f:
            pdf_bytes = f.read()

        doc_data = PDFIngestionPipeline().process_pdf(pdf_bytes, SAMPLE_PDF.name)
        chunks = ContextAwareChunker().chunk_document(doc_data)

        store = VectorStoreManager(persist_directory=tmpdir, collection_name="test_rag_abstain")
        store.index_chunks(chunks)

        test_settings = Settings(google_api_key="")
        rag = RAGPipeline(vector_store=store, settings=test_settings)

        resp = rag.query("What is the lidar optical laser wavelength and point cloud format?")
        assert resp.is_abstained is True
        assert "insufficient evidence" in resp.answer.lower()
        assert len(resp.citations) == 0


def test_activities_query_matches_workshops_evidence():
    """Regression test: User asks for 'activities', text mentions 'workshops'."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = VectorStoreManager(persist_directory=tmpdir, collection_name="test_workshops")
        chunk = make_test_chunk(
            cid="chunk_acm_01",
            doc_id="doc_acm_test",
            fname="ACM_Student_Chapter_Report.pdf",
            page=1,
            section="Activities Overview",
            text="The team actively organized 7 technical workshops on cloud computing and competitive programming.",
        )
        store.index_chunks([chunk])

        test_settings = Settings(google_api_key="")
        rag = RAGPipeline(vector_store=store, settings=test_settings)

        resp = rag.query("Which activities were conducted by the student chapter?", doc_id_filter="doc_acm_test")
        assert resp.is_abstained is False
        assert resp.mode == "retrieval_only"
        assert "workshops" in resp.answer.lower()
        assert len(resp.citations) == 1
        assert resp.citations[0].document == "ACM_Student_Chapter_Report.pdf"
        assert resp.citations[0].page == 1


def test_tools_query_matches_github_and_kaggle():
    """Regression test: Query for tools/platforms matches GitHub and Kaggle."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = VectorStoreManager(persist_directory=tmpdir, collection_name="test_tools")
        chunk = make_test_chunk(
            cid="chunk_tools_01",
            doc_id="doc_tools_test",
            fname="ACM_Student_Chapter_Report.pdf",
            page=2,
            section="Tools & Platforms",
            text="Sessions focused on hands-on tools such as GitHub usage for version control and Kaggle for data science.",
        )
        store.index_chunks([chunk])

        test_settings = Settings(google_api_key="")
        rag = RAGPipeline(vector_store=store, settings=test_settings)

        resp = rag.query("Which tools and platforms are mentioned in the sessions?", doc_id_filter="doc_tools_test")
        assert resp.is_abstained is False
        assert "github" in resp.answer.lower()
        assert "kaggle" in resp.answer.lower()
        assert resp.citations[0].page == 2


def test_general_doc_has_no_hardcoded_autosar_wording():
    """Regression test: ACM / General documents must never say 'AUTOSAR HLD documentation'."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = VectorStoreManager(persist_directory=tmpdir, collection_name="test_no_autosar")
        chunk = make_test_chunk(
            cid="chunk_general_01",
            doc_id="doc_general_test",
            fname="Student_Activity_Summary.pdf",
            page=1,
            section="Executive Summary",
            text="The annual coding competition attracted over 250 registered students across all engineering branches.",
        )
        store.index_chunks([chunk])

        test_settings = Settings(google_api_key="")
        rag = RAGPipeline(vector_store=store, settings=test_settings)

        resp = rag.query("How many students participated in the coding competition?", doc_id_filter="doc_general_test")
        assert "autosar" not in resp.answer.lower()
        assert "Student_Activity_Summary.pdf" in resp.answer


def test_partial_evidence_handling():
    """Regression test: Partial evidence answers supported facts and notes missing facts."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = VectorStoreManager(persist_directory=tmpdir, collection_name="test_partial")
        chunk = make_test_chunk(
            cid="chunk_part_01",
            doc_id="doc_part_test",
            fname="Chapter_Events.pdf",
            page=2,
            section="Events",
            text="The committee hosted 3 hackathons during the autumn semester for undergraduate participants.",
        )
        store.index_chunks([chunk])

        test_settings = Settings(google_api_key="")
        rag = RAGPipeline(vector_store=store, settings=test_settings)

        resp = rag.query("What hackathons were hosted and what was the financial sponsorship budget?", doc_id_filter="doc_part_test")
        assert resp.is_abstained is False
        assert "hackathons" in resp.answer.lower()
        assert "note on partial evidence" in resp.answer.lower()
        assert len(resp.citations) == 1


def test_no_supporting_evidence_abstains():
    """Regression test: Completely unrelated topic triggers clean, document-aware abstention."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = VectorStoreManager(persist_directory=tmpdir, collection_name="test_unrelated")
        chunk = make_test_chunk(
            cid="chunk_unrel_01",
            doc_id="doc_unrel_test",
            fname="Chapter_Schedule.pdf",
            page=1,
            section="Weekly Schedule",
            text="ACM student chapter conducts weekly coding practice and algorithm study sessions.",
        )
        store.index_chunks([chunk])

        test_settings = Settings(google_api_key="")
        rag = RAGPipeline(vector_store=store, settings=test_settings)

        resp = rag.query("What is the supersonic aerodynamics Mach number of the jet turbine?", doc_id_filter="doc_unrel_test")
        assert resp.is_abstained is True
        assert "insufficient evidence" in resp.answer.lower()
        assert len(resp.citations) == 0
        assert "Chapter_Schedule.pdf" in resp.answer


def test_document_filtering_across_multiple_pdfs():
    """Regression test: Strict doc_id filtering prevents cross-document contamination."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = VectorStoreManager(persist_directory=tmpdir, collection_name="test_multi_docs")
        chunk_a = make_test_chunk(
            cid="chunk_a",
            doc_id="doc_autosar_id",
            fname="autosar_hld.pdf",
            page=3,
            section="SWC",
            text="ThrottleControl_SWC provides PpThrottlePos with period 10ms.",
        )
        chunk_b = make_test_chunk(
            cid="chunk_b",
            doc_id="doc_acm_id",
            fname="acm_report.pdf",
            page=2,
            section="Workshops",
            text="ACM chapter organized GitHub profile optimization sessions.",
        )
        store.index_chunks([chunk_a, chunk_b])

        test_settings = Settings(google_api_key="")
        rag = RAGPipeline(vector_store=store, settings=test_settings)

        resp_b = rag.query("What sessions were organized?", doc_id_filter="doc_acm_id")
        assert resp_b.is_abstained is False
        assert "github" in resp_b.answer.lower()
        assert "throttle" not in resp_b.answer.lower()
        for c in resp_b.citations:
            assert c.document == "acm_report.pdf"


def test_different_activities_query_extracts_acm_evidence():
    """Bug 2 acceptance test: 'Which are the different activities?' extracts leadership, speaking,
    teaching, audiences, and event execution without abstaining.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = VectorStoreManager(persist_directory=tmpdir, collection_name="test_acm_activities")
        chunk_skills = make_test_chunk(
            cid="chunk_p3",
            doc_id="doc_acm_exp4",
            fname="ACM student chapter Exp_4 acm (9).pdf",
            page=3,
            section="developed:",
            text=(
                "Through this experience, I developed:\n"
                "● Strong leadership and public speaking skills\n"
                "● Ability to teach and simplify technical concepts\n"
                "● Experience in handling large audiences\n"
                "● Team collaboration and event execution skills"
            ),
        )
        chunk_workshops = make_test_chunk(
            cid="chunk_p2",
            doc_id="doc_acm_exp4",
            fname="ACM student chapter Exp_4 acm (9).pdf",
            page=2,
            section="Page 2",
            text=(
                "independently led 4 workshops, which focused on practical and career-oriented "
                "topics such as LinkedIn profile optimization, GitHub usage, Kaggle for data science, "
                "and Generative AI."
            ),
        )
        store.index_chunks([chunk_skills, chunk_workshops])

        test_settings = Settings(google_api_key="")
        rag = RAGPipeline(vector_store=store, settings=test_settings)

        resp = rag.query("Which are the different activities?", doc_id_filter="doc_acm_exp4")
        assert resp.is_abstained is False
        assert resp.mode == "retrieval_only"
        ans_lower = resp.answer.lower()
        assert "leadership" in ans_lower or "speaking" in ans_lower
        assert "workshops" in ans_lower
        assert len(resp.citations) >= 1
        for c in resp.citations:
            assert c.document == "ACM student chapter Exp_4 acm (9).pdf"
            assert c.page in [2, 3]


def test_workshops_and_topics_query():
    """Bug 2 & Bug 4 acceptance test: Workshops and technical topics query."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = VectorStoreManager(persist_directory=tmpdir, collection_name="test_workshops_topics")
        chunk = make_test_chunk(
            cid="chunk_ws",
            doc_id="doc_acm_ws",
            fname="ACM student chapter Exp_4 acm (9).pdf",
            page=2,
            section="Page 2",
            text=(
                "independently led 4 workshops, which focused on practical and career-oriented topics "
                "such as LinkedIn profile optimization, GitHub usage, Kaggle for data science, and Generative AI."
            ),
        )
        store.index_chunks([chunk])

        test_settings = Settings(google_api_key="")
        rag = RAGPipeline(vector_store=store, settings=test_settings)

        resp = rag.query("What workshops were organized and what technical topics were covered?", doc_id_filter="doc_acm_ws")
        assert resp.is_abstained is False
        ans_lower = resp.answer.lower()
        assert "linkedin" in ans_lower or "github" in ans_lower or "kaggle" in ans_lower
        assert resp.citations[0].page == 2


def test_missing_api_key_handling():
    """Bug 1 acceptance test: Safe handling of missing credentials without error or leakage."""
    from src.config import test_gemini_connection
    status, msg = test_gemini_connection(api_key="")
    assert status == "Not configured"
    assert "No API key" in msg


def test_invalid_api_key_handling():
    """Bug 1 acceptance test: Real/mock test showing safe auth failure message."""
    from src.config import test_gemini_connection
    status, msg = test_gemini_connection(api_key="AIzaSyFakeKeyThatWillFailAuth123456789")
    assert status in ["Authentication failed", "Connection failed", "Model unavailable"]
    assert "AIzaSy" not in msg  # No key leakage!
