"""Controlled mock tests for Gemini LLM integration, configuration, and error handling.
NOTE: Mock tests demonstrate prompt construction, citation parsing, and safe failure
modes; they are NOT presented as proof of live API connectivity.
"""

from unittest.mock import MagicMock, patch
from src.config import Settings, test_gemini_connection as verify_gemini_connection
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStoreManager


def test_missing_api_key_handling():
    settings = Settings()
    settings.google_api_key = ""
    assert settings.is_gemini_configured is False
    assert settings.masked_key_status == "Not configured"

    store = VectorStoreManager(persist_directory="./tmp_mock")
    rag = RAGPipeline(vector_store=store, settings=settings)
    ans, err = rag._call_gemini_api("Test prompt")
    assert ans is None
    assert "not configured" in err.lower()


def test_gemini_connection_status_not_configured():
    status, msg = verify_gemini_connection(api_key="")
    assert status in ["Not configured", "API key missing"]
    assert "no api key configured" in msg.lower()


def test_gemini_connection_mock_auth_failure():
    with patch("google.genai.Client") as mock_client:
        mock_client.side_effect = Exception("API_KEY_INVALID: The provided API key is not valid")
        status, msg = verify_gemini_connection(api_key="AIzaSyInvalidKey123456789", model_name="gemini-2.0-flash")
        assert status == "Authentication failed"
        assert "rejected" in msg.lower()


def test_gemini_connection_mock_model_unavailable():
    with patch("google.genai.Client") as mock_client:
        inst = MagicMock()
        inst.models.generate_content.side_effect = Exception("Model not found: models/gemini-unsupported-model")
        mock_client.return_value = inst
        status, msg = verify_gemini_connection(api_key="AIzaSyValidFormatKey12345678", model_name="gemini-unsupported-model")
        assert status == "Model unavailable"
        assert "not supported or accessible" in msg.lower() or "not available" in msg.lower()


def test_gemini_connection_mock_success():
    with patch("google.genai.Client") as mock_client:
        inst = MagicMock()
        resp = MagicMock()
        resp.text = "PONG"
        inst.models.generate_content.return_value = resp
        mock_client.return_value = inst
        status, msg = verify_gemini_connection(api_key="AIzaSyValidFormatKey12345678", model_name="gemini-2.0-flash")
        assert "connected" in status.lower()
        assert "successfully" in msg.lower()


def test_mocked_gemini_successful_response():
    mock_response = MagicMock()
    mock_response.text = (
        "ThrottleControl_SWC provides throttle position using "
        "interface If_ThrottlePosition [Doc: synthetic_autosar_hld.pdf, Page: 3, Section: Software Components]."
    )

    retrieved = [
        {
            "chunk_id": "c1",
            "text": "ThrottleControl_SWC provides PpThrottlePos",
            "metadata": {"filename": "synthetic_autosar_hld.pdf", "page_num": 3, "section": "Software Components"},
            "similarity": 0.85,
        }
    ]

    settings = Settings()
    settings.google_api_key = "AIzaSyMockKeyForTesting12345"
    settings.gemini_model = "gemini-2.0-flash"
    store = MagicMock()
    store.search.return_value = retrieved

    rag = RAGPipeline(vector_store=store, settings=settings)

    with patch.object(rag, "_call_gemini_api", return_value=(mock_response.text, None)):
        res = rag.query("What does ThrottleControl_SWC provide?")
        assert res.mode == "gemini_grounded"
        assert len(res.citations) == 1
        assert res.citations[0].is_valid is True
        assert res.citations[0].page == 3
        assert res.citation_validity_rate == 1.0


def test_mocked_gemini_abstention_response():
    mock_text = (
        "Based on the provided HLD documentation, there is insufficient evidence to answer this question. "
        "The retrieved sections do not mention autonomous parking radar frequencies."
    )
    retrieved = [
        {
            "chunk_id": "c1",
            "text": "General architecture notes",
            "metadata": {"filename": "synthetic_autosar_hld.pdf", "page_num": 1, "section": "Overview"},
            "similarity": 0.70,
        }
    ]
    settings = Settings()
    settings.google_api_key = "AIzaSyMockKeyForTesting12345"
    store = MagicMock()
    store.search.return_value = retrieved

    rag = RAGPipeline(vector_store=store, settings=settings)
    with patch.object(rag, "_call_gemini_api", return_value=(mock_text, None)):
        res = rag.query("What is the parking radar frequency?")
        assert res.is_abstained is True
        assert "insufficient evidence" in res.answer.lower()


def test_duplicate_citations_are_deduplicated():
    """Ensure identical citations referencing the same doc, page, section, and chunk are deduplicated."""
    rag = RAGPipeline(vector_store=MagicMock(), settings=Settings())
    retrieved = [
        {
            "chunk_id": "c1",
            "text": "ThrottleControl_SWC provides PpThrottlePos and calculates angle.",
            "metadata": {"filename": "synthetic_autosar_hld.pdf", "page_num": 3, "section": "Section 3"},
            "similarity": 0.88,
        }
    ]

    # Repeated identical citation tokens emitted 4 times by LLM
    answer_text = (
        "ThrottleControl_SWC calculates angle. [Doc: synthetic_autosar_hld.pdf, Page: 3, Section: Section 3]\n"
        "It provides PpThrottlePos. [Doc: synthetic_autosar_hld.pdf, Page: 3, Section: Section 3]\n"
        "It requires RpPedalSensor. [Doc: synthetic_autosar_hld.pdf, Page: 3, Section: Section 3]\n"
        "Documented in Section 3. [Doc: synthetic_autosar_hld.pdf, Page: 3, Section: Section 3]"
    )

    cits, validity = rag._validate_citations(answer_text, retrieved)
    assert len(cits) == 1
    assert cits[0].page == 3
    assert cits[0].section == "Section 3"
    assert cits[0].supporting_chunk_id == "c1"
    assert cits[0].is_valid is True
    assert validity == 1.0


def test_distinct_page_passages_preserved_as_separate_citations():
    """Ensure citations pointing to different chunks or different sections on the same page are preserved."""
    rag = RAGPipeline(vector_store=MagicMock(), settings=Settings())
    retrieved = [
        {
            "chunk_id": "chunk_sec31",
            "text": "ThrottleControl_SWC responsibility details.",
            "metadata": {"filename": "synthetic_autosar_hld.pdf", "page_num": 3, "section": "Section 3.1"},
            "similarity": 0.88,
        },
        {
            "chunk_id": "chunk_sec32",
            "text": "BrakeManager_SWC responsibility details.",
            "metadata": {"filename": "synthetic_autosar_hld.pdf", "page_num": 3, "section": "Section 3.2"},
            "similarity": 0.82,
        },
    ]

    answer_text = (
        "ThrottleControl is defined in Section 3.1 [Doc: synthetic_autosar_hld.pdf, Page: 3, Section: Section 3.1]. "
        "BrakeManager is defined in Section 3.2 [Doc: synthetic_autosar_hld.pdf, Page: 3, Section: Section 3.2]."
    )

    cits, validity = rag._validate_citations(answer_text, retrieved)
    assert len(cits) == 2
    assert {c.section for c in cits} == {"Section 3.1", "Section 3.2"}
    assert {c.supporting_chunk_id for c in cits} == {"chunk_sec31", "chunk_sec32"}
    assert validity == 1.0


def test_mock_generation_does_not_mark_live_connected():
    """When generation uses a mock/patch, is_live_gemini must be False and status must not be Connected."""
    from src.config import STATUS_NOT_TESTED
    settings = Settings(google_api_key="AIzaSyMockKeyForUnitTestingOnly12345", gemini_model="gemini-3.5-flash")
    store = MagicMock()
    store.search.return_value = [
        {
            "chunk_id": "c1",
            "text": "ThrottleControl_SWC responsibility text",
            "metadata": {"filename": "synthetic_autosar_hld.pdf", "page_num": 3, "section": "Section 3"},
            "similarity": 0.85,
        }
    ]
    rag = RAGPipeline(vector_store=store, settings=settings)

    with patch.object(rag, "_call_gemini_api", return_value=("Mock answer with [Doc: synthetic_autosar_hld.pdf, Page: 3, Section: Section 3]", None)):
        res = rag.query("What is ThrottleControl_SWC?")
        assert res.mode == "gemini_grounded"
        assert res.is_live_gemini is False
        assert res.gemini_status == STATUS_NOT_TESTED


def test_fallback_status_truthful_on_api_error():
    """When Gemini generation encounters an error, status must reflect the error and is_live_gemini must be False."""
    from src.config import STATUS_RATE_LIMITED
    settings = Settings(google_api_key="AIzaSyMockKeyForUnitTestingOnly12345", gemini_model="gemini-3.5-flash")
    store = MagicMock()
    store.search.return_value = [
        {
            "chunk_id": "c1",
            "text": "ThrottleControl_SWC responsibility text",
            "metadata": {"filename": "synthetic_autosar_hld.pdf", "page_num": 3, "section": "Section 3"},
            "similarity": 0.85,
        }
    ]
    rag = RAGPipeline(vector_store=store, settings=settings)

    with patch("src.rag_pipeline.call_gemini_with_quota_protection") as mock_call:
        mock_call.return_value = (None, "Rate limit exceeded for gemini-3.5-flash", STATUS_RATE_LIMITED)
        res = rag.query("What is ThrottleControl_SWC?")
        assert res.mode == "retrieval_only"
        assert res.is_live_gemini is False
        assert res.gemini_status == STATUS_RATE_LIMITED


def test_gemini_key_env_resolution():
    """Verify both GOOGLE_API_KEY and GEMINI_API_KEY environment variables are detected."""
    import os
    from src.config import _get_secret_or_env, Settings

    with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyGeminiKeyFromEnv12345678", "GOOGLE_API_KEY": ""}):
        key = _get_secret_or_env("GOOGLE_API_KEY")
        assert key == "AIzaSyGeminiKeyFromEnv12345678"
        s = Settings()
        assert s.is_gemini_configured is True

    with patch.dict(os.environ, {"GOOGLE_API_KEY": "AIzaSyGoogleKeyFromEnv12345678", "GEMINI_API_KEY": ""}):
        key = _get_secret_or_env("GOOGLE_API_KEY")
        assert key == "AIzaSyGoogleKeyFromEnv12345678"
        s = Settings()
        assert s.is_gemini_configured is True


def test_placeholder_keys_are_unconfigured():
    """Verify placeholder keys from template files are treated as unconfigured."""
    from src.config import Settings
    placeholders = [
        "your_google_api_key_here",
        "your_api_key_here",
        "your_gemini_api_key_here",
        "paste_your_new_key_here",
        "replace_with_a_supported_model_name",
        "",
        "short_key",
    ]
    for ph in placeholders:
        s = Settings(google_api_key=ph)
        assert s.is_gemini_configured is False, f"Expected {ph} to be unconfigured"
        assert s.masked_key_status == "Not configured"


def test_throttlecontrol_citation_and_lidar_abstention_on_index():
    """Verify ThrottleControl_SWC cites the correct page and unsupported lidar query abstains."""
    from pathlib import Path
    from src.config import PROJECT_ROOT
    from src.vector_store import VectorStoreManager

    persist_dir = str(PROJECT_ROOT / "chroma_db")
    if Path(persist_dir).exists():
        vstore = VectorStoreManager(persist_directory=persist_dir)
        settings = Settings(google_api_key="")  # Retrieval-only mode
        rag = RAGPipeline(vector_store=vstore, settings=settings)

        # 1. ThrottleControl_SWC query
        res = rag.query("What is the primary function and port mapping of ThrottleControl_SWC?")
        assert res.is_abstained is False
        assert len(res.citations) > 0
        # ThrottleControl_SWC is specified in Section 3 on Page 3
        pages = [c.page for c in res.citations]
        assert 3 in pages, f"Expected page 3 in citations, got {pages}"
        assert all(c.is_valid for c in res.citations)

        # 2. Unsupported Lidar question
        res_lidar = rag.query("What is the lidar optical laser wavelength and point cloud format?")
        assert res_lidar.is_abstained is True
        assert "insufficient evidence" in res_lidar.answer.lower()


def test_four_key_and_connection_states():
    """Explicitly verify:
    1) Missing key -> 'Not configured'
    2) Valid configured key -> 'Configured, not tested'
    3) Failed connection test -> 'Authentication failed'
    4) Successful connection test -> 'Connected'
    """
    from src.config import (
        STATUS_KEY_MISSING,
        STATUS_NOT_TESTED,
        STATUS_CONNECTED,
        STATUS_AUTH_FAILED,
    )

    # 1. Missing key
    status_missing, msg_missing = verify_gemini_connection(api_key="")
    assert status_missing == STATUS_KEY_MISSING
    assert "no api key" in msg_missing.lower()

    # 2. Valid configured key (before test)
    settings = Settings(google_api_key="AIzaSyValidDummyKey123456789")
    assert settings.is_gemini_configured is True
    assert STATUS_NOT_TESTED == "Configured, not tested"

    # 3. Failed connection test
    with patch("google.genai.Client") as mock_client:
        mock_client.side_effect = Exception("API_KEY_INVALID: The provided API key is not valid")
        status_fail, msg_fail = verify_gemini_connection(api_key="AIzaSyInvalidKey123456789", model_name="gemini-2.0-flash")
        assert status_fail == STATUS_AUTH_FAILED
        assert "rejected" in msg_fail.lower() or "authentication failed" in msg_fail.lower()

    # 4. Successful mocked response
    with patch("google.genai.Client") as mock_client:
        inst = MagicMock()
        resp = MagicMock()
        resp.text = "PONG"
        inst.models.generate_content.return_value = resp
        mock_client.return_value = inst
        status_ok, msg_ok = verify_gemini_connection(api_key="AIzaSyValidDummyKey123456789", model_name="gemini-2.0-flash")
        assert status_ok == STATUS_CONNECTED
        assert "successfully" in msg_ok.lower()


def test_st_secrets_fallback_safety():
    """Verify that when os.environ and .env have no key, st.secrets acts safely as a fallback."""
    import os
    import sys
    from src.config import _get_secret_or_env

    mock_st = MagicMock()
    mock_st.secrets = {"GOOGLE_API_KEY": "AIzaSyKeyFromSecretsToml123456"}

    with patch.dict(os.environ, {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""}):
        with patch("src.config.dotenv_values", return_value={}):
            with patch.dict(sys.modules, {"streamlit": mock_st}):
                resolved = _get_secret_or_env("GOOGLE_API_KEY")
                assert resolved == "AIzaSyKeyFromSecretsToml123456"



