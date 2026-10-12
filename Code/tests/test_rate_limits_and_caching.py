"""Automated unit tests for Gemini rate-limiting, error handling, caching, and fallback.
Uses mocked SDK clients exclusively to ensure ZERO real API quota consumption.
"""

import time
from unittest.mock import MagicMock, patch
import pytest

from src.config import Settings
from src.gemini_client import (
    call_gemini_with_quota_protection,
    compute_evidence_cache_key,
    is_quota_blocked,
    set_quota_blocked,
    clear_quota_block,
    _RESPONSE_CACHE,
    STATUS_KEY_MISSING,
    STATUS_CONNECTED,
    STATUS_RATE_LIMITED,
    STATUS_MODEL_UNAVAILABLE,
    STATUS_AUTH_FAILED,
    STATUS_SERVICE_UNAVAILABLE,
    STATUS_CONNECTION_FAILED,
)
from src.rag_pipeline import RAGPipeline


@pytest.fixture(autouse=True)
def reset_rate_limit_state():
    """Ensure clean state before each test."""
    clear_quota_block()
    _RESPONSE_CACHE.clear()
    yield
    clear_quota_block()
    _RESPONSE_CACHE.clear()


def test_missing_api_key_returns_missing_status():
    """Ensure empty or placeholder key returns STATUS_KEY_MISSING without calling SDK."""
    text, err, status = call_gemini_with_quota_protection(
        prompt="Hello",
        model_id="gemini-3.5-flash",
        api_key="",
    )
    assert text is None
    assert status == STATUS_KEY_MISSING
    assert "no gemini api key" in err.lower()


def test_successful_gemini_generation_and_caching():
    """Ensure successful generation caches result and returns STATUS_CONNECTED."""
    with patch("src.gemini_client.get_gemini_client") as mock_get_client:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "Answer from Gemini 3.5 Flash"
        mock_client.models.generate_content.return_value = mock_resp
        mock_get_client.return_value = mock_client

        cache_key = "test_cache_key_123"
        text, err, status = call_gemini_with_quota_protection(
            prompt="What is ThrottleControl_SWC?",
            model_id="gemini-3.5-flash",
            api_key="AIzaSyMockKeyForUnitTestingOnly12345",
            cache_key=cache_key,
        )

        assert text == "Answer from Gemini 3.5 Flash"
        assert err is None
        assert status == STATUS_CONNECTED
        assert mock_client.models.generate_content.call_count == 1

        # Second call with same cache_key should hit cache without calling generate_content again
        text2, err2, status2 = call_gemini_with_quota_protection(
            prompt="What is ThrottleControl_SWC?",
            model_id="gemini-3.5-flash",
            api_key="AIzaSyMockKeyForUnitTestingOnly12345",
            cache_key=cache_key,
        )
        assert text2 == "Answer from Gemini 3.5 Flash"
        assert status2 == STATUS_CONNECTED
        assert mock_client.models.generate_content.call_count == 1  # Unchanged!


def test_rate_limit_429_sets_cooldown_and_does_not_retry():
    """Ensure HTTP 429 stops immediately, sets quota block, and does not retry."""
    with patch("src.gemini_client.get_gemini_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "429 RESOURCE_EXHAUSTED: Quota exceeded for metric generate_content. retry in 10m"
        )
        mock_get_client.return_value = mock_client

        text, err, status = call_gemini_with_quota_protection(
            prompt="Hello",
            model_id="gemini-3.8-flash",
            api_key="AIzaSyMockKeyForUnitTestingOnly12345",
        )

        assert text is None
        assert status == STATUS_RATE_LIMITED
        assert "quota exceeded" in err.lower() or "rate limit" in err.lower()
        # Ensure only 1 call was made (NO retry loop on 429)
        assert mock_client.models.generate_content.call_count == 1

        # Subsequent call should be blocked immediately without calling SDK
        blocked, msg, remaining = is_quota_blocked()
        assert blocked is True
        assert remaining > 0

        text_blocked, err_blocked, status_blocked = call_gemini_with_quota_protection(
            prompt="Hello again",
            model_id="gemini-3.8-flash",
            api_key="AIzaSyMockKeyForUnitTestingOnly12345",
        )
        assert text_blocked is None
        assert status_blocked == STATUS_RATE_LIMITED
        assert mock_client.models.generate_content.call_count == 1  # Still 1!


def test_model_404_not_found_does_not_retry_or_switch():
    """Ensure HTTP 404 reports model unavailable immediately without model cycling."""
    with patch("src.gemini_client.get_gemini_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "404 NOT_FOUND: models/gemini-invalid-model is not found"
        )
        mock_get_client.return_value = mock_client

        text, err, status = call_gemini_with_quota_protection(
            prompt="Hello",
            model_id="gemini-invalid-model",
            api_key="AIzaSyMockKeyForUnitTestingOnly12345",
        )

        assert text is None
        assert status == STATUS_MODEL_UNAVAILABLE
        assert "not available" in err.lower() or "not supported" in err.lower()
        assert mock_client.models.generate_content.call_count == 1


def test_auth_failure_401_403_stops_immediately():
    """Ensure authentication errors are categorized correctly without retry."""
    with patch("src.gemini_client.get_gemini_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "401 UNAUTHENTICATED: API_KEY_INVALID"
        )
        mock_get_client.return_value = mock_client

        text, err, status = call_gemini_with_quota_protection(
            prompt="Hello",
            model_id="gemini-3.5-flash",
            api_key="AIzaSyInvalidKey12345",
        )

        assert text is None
        assert status == STATUS_AUTH_FAILED
        assert "authentication failed" in err.lower()
        assert mock_client.models.generate_content.call_count == 1


def test_service_unavailable_503_has_bounded_single_retry():
    """Ensure HTTP 503 retries once, then fails safely without runaway loops."""
    with patch("src.gemini_client.get_gemini_client") as mock_get_client:
        with patch("time.sleep") as mock_sleep:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = Exception(
                "503 Service Unavailable: High load"
            )
            mock_get_client.return_value = mock_client

            text, err, status = call_gemini_with_quota_protection(
                prompt="Hello",
                model_id="gemini-3.5-flash-lite",
                api_key="AIzaSyMockKeyForUnitTestingOnly12345",
            )

            assert text is None
            assert status in (STATUS_SERVICE_UNAVAILABLE, STATUS_CONNECTION_FAILED)
            # Bounded: 1 initial attempt + 1 retry = 2 total
            assert mock_client.models.generate_content.call_count == 2
            assert mock_sleep.call_count == 1


def test_cache_key_invalidation_on_changed_evidence():
    """Ensure cache key differs if question, scope, or evidence changes."""
    chunks_v1 = [
        {"chunk_id": "c1", "text": "ThrottleControl calculates angle", "metadata": {"filename": "doc.pdf", "page_num": 3}}
    ]
    chunks_v2 = [
        {"chunk_id": "c1", "text": "ThrottleControl modified text", "metadata": {"filename": "doc.pdf", "page_num": 3}}
    ]

    key_v1 = compute_evidence_cache_key("What is ThrottleControl?", "doc.pdf", chunks_v1, "gemini-3.5-flash")
    key_v2 = compute_evidence_cache_key("What is ThrottleControl?", "doc.pdf", chunks_v2, "gemini-3.5-flash")
    key_diff_q = compute_evidence_cache_key("What is BrakeManager?", "doc.pdf", chunks_v1, "gemini-3.5-flash")
    key_diff_model = compute_evidence_cache_key("What is ThrottleControl?", "doc.pdf", chunks_v1, "gemini-3.6-flash")

    assert key_v1 != key_v2
    assert key_v1 != key_diff_q
    assert key_v1 != key_diff_model


def test_rag_pipeline_uses_retrieval_fallback_on_gemini_failure():
    """Ensure RAG pipeline safely falls back to local evidence extraction with clear mode label."""
    retrieved = [
        {
            "chunk_id": "c1",
            "text": "Section 3.1: ThrottleControl_SWC is responsible for calculating engine throttle angle from accelerator pedal.",
            "metadata": {"filename": "synthetic_autosar_hld.pdf", "page_num": 3, "section": "Section 3: Software Components"},
            "similarity": 0.88,
        }
    ]

    mock_store = MagicMock()
    mock_store.search.return_value = retrieved

    settings = Settings(google_api_key="AIzaSyMockKeyForUnitTestingOnly12345", gemini_model="gemini-3.5-flash")
    pipeline = RAGPipeline(vector_store=mock_store, settings=settings)

    # Simulate Gemini 429 quota failure
    with patch("src.rag_pipeline.call_gemini_with_quota_protection") as mock_call:
        mock_call.return_value = (None, "Rate limit / daily quota exceeded for 'gemini-3.5-flash'.", STATUS_RATE_LIMITED)

        res = pipeline.query("What is the responsibility of ThrottleControl_SWC?")
        assert res.mode == "retrieval_only"
        assert res.error_message is not None
        assert "quota exceeded" in res.error_message.lower() or "rate limit" in res.error_message.lower()
        assert "ThrottleControl_SWC" in res.answer
        assert len(res.citations) >= 1


def test_call_origin_distinguishes_live_mock_and_cache():
    """Ensure call origin helpers truthfully identify live, mock, and cached responses."""
    from src.gemini_client import was_last_call_live, was_last_call_cached, was_last_call_mock

    # 1. Test mock client
    with patch("src.gemini_client.get_gemini_client") as mock_get_client:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "Answer from mock"
        mock_client.models.generate_content.return_value = mock_resp
        mock_get_client.return_value = mock_client

        text, err, status = call_gemini_with_quota_protection(
            prompt="Test prompt",
            model_id="gemini-3.5-flash",
            api_key="AIzaSyMockKeyForUnitTestingOnly12345",
            cache_key="origin_test_key_1",
        )
        assert text == "Answer from mock"
        assert was_last_call_mock() is True
        assert was_last_call_live() is False
        assert was_last_call_cached() is False

        # 2. Test cache hit
        text2, err2, status2 = call_gemini_with_quota_protection(
            prompt="Test prompt",
            model_id="gemini-3.5-flash",
            api_key="AIzaSyMockKeyForUnitTestingOnly12345",
            cache_key="origin_test_key_1",
        )
        assert text2 == "Answer from mock"
        assert was_last_call_cached() is True
        assert was_last_call_live() is False

    # 3. Test simulated live client (non-mock class)
    class RealClientStub:
        class Models:
            def generate_content(self, model, contents):
                class Resp:
                    text = "Live answer from model"
                return Resp()
        models = Models()

    with patch("src.gemini_client.get_gemini_client", return_value=(RealClientStub(), None)):
        text3, err3, status3 = call_gemini_with_quota_protection(
            prompt="Different prompt",
            model_id="gemini-3.5-flash",
            api_key="AIzaSyMockKeyForUnitTestingOnly12345",
            cache_key="origin_test_key_2",
        )
        assert text3 == "Live answer from model"
        assert was_last_call_live() is True
        assert was_last_call_mock() is False
        assert was_last_call_cached() is False

