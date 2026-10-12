"""Targeted tests for two fixes in src/gemini_client.py.

Fix 1 - Windows wsarecv/wsasend/stream-reading-error network errors are classified
         as STATUS_CONNECTION_FAILED and retried exactly once (bounded retry).

Fix 2 - get_available_gemini_models() iterates all models before returning.
         The old code had `return discovered` inside the for-loop so it returned
         after the first match instead of collecting all candidates.

All tests are fully mocked - zero real API calls.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.gemini_client import (
    call_gemini_with_quota_protection,
    get_available_gemini_models,
    clear_quota_block,
    _RESPONSE_CACHE,
    STATUS_CONNECTION_FAILED,
    STATUS_CONNECTED,
    STATUS_RATE_LIMITED,
    STATUS_AUTH_FAILED,
)


@pytest.fixture(autouse=True)
def clean_state():
    clear_quota_block()
    _RESPONSE_CACHE.clear()
    yield
    clear_quota_block()
    _RESPONSE_CACHE.clear()


VALID_KEY = "AIzaSyMockKeyForUnitTestingOnly12345"


def _run_network_error(error_message):
    with patch("src.gemini_client.get_gemini_client") as mock_get_client:
        with patch("time.sleep") as mock_sleep:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = Exception(error_message)
            mock_get_client.return_value = mock_client
            text, err, status = call_gemini_with_quota_protection(
                prompt="Test prompt",
                model_id="gemini-3.5-flash-lite",
                api_key=VALID_KEY,
            )
            return text, err, status, mock_client.models.generate_content.call_count, mock_sleep.call_count


def _fake_model(name):
    m = MagicMock()
    m.name = f"models/{name}"
    return m


# ---- Fix 1: wsarecv ----

WSARECV_ERROR = (
    "stream reading error: read tcp 192.168.1.44:52627->172.217.114.4:443: "
    "wsarecv: An established connection was aborted by the software in your host machine."
)

WSASEND_ERROR = (
    "request failed: Post \"https://generativelanguage.googleapis.com\": "
    "write tcp [::1]:53585->[2001:4860:4843:400::]:443: "
    "wsasend: An existing connection was forcibly closed by the remote host."
)

STREAM_ERROR = "stream reading error: eof occurred in violation of protocol"
ABORTED_ERROR = "connection aborted: remote peer closed connection unexpectedly"


def test_wsarecv_classified_as_connection_failed():
    _, _, status, _, _ = _run_network_error(WSARECV_ERROR)
    assert status == STATUS_CONNECTION_FAILED, f"got '{status}'"

def test_wsarecv_bounded_to_exactly_one_retry():
    _, _, _, calls, sleeps = _run_network_error(WSARECV_ERROR)
    assert calls == 2, f"Expected 2 generate calls, got {calls}"
    assert sleeps == 1, f"Expected 1 sleep, got {sleeps}"

def test_wsarecv_user_friendly_message():
    _, err, _, _, _ = _run_network_error(WSARECV_ERROR)
    assert err is not None
    assert "wsarecv" not in err.lower()
    assert "192.168" not in err
    assert any(kw in err.lower() for kw in ["firewall", "antivirus", "vpn", "interrupted", "connection"])

def test_wsarecv_text_is_none():
    text, _, _, _, _ = _run_network_error(WSARECV_ERROR)
    assert text is None

def test_wsasend_classified_as_connection_failed():
    _, _, status, _, _ = _run_network_error(WSASEND_ERROR)
    assert status == STATUS_CONNECTION_FAILED

def test_wsasend_bounded_to_exactly_one_retry():
    _, _, _, calls, sleeps = _run_network_error(WSASEND_ERROR)
    assert calls == 2 and sleeps == 1

def test_stream_reading_error_classified_connection_failed():
    _, _, status, _, _ = _run_network_error(STREAM_ERROR)
    assert status == STATUS_CONNECTION_FAILED

def test_stream_reading_error_bounded_retry():
    _, _, _, calls, _ = _run_network_error(STREAM_ERROR)
    assert calls == 2

def test_connection_aborted_classified_connection_failed():
    _, _, status, _, _ = _run_network_error(ABORTED_ERROR)
    assert status == STATUS_CONNECTION_FAILED

def test_connection_aborted_bounded_retry():
    _, _, _, calls, _ = _run_network_error(ABORTED_ERROR)
    assert calls == 2

def test_wsarecv_not_classified_as_rate_limited():
    _, _, status, _, _ = _run_network_error("wsarecv: connection aborted")
    assert status != STATUS_RATE_LIMITED

def test_wsasend_not_classified_as_auth_failed():
    _, _, status, _, _ = _run_network_error("wsasend: connection forcibly closed")
    assert status != STATUS_AUTH_FAILED

def test_network_error_recover_on_retry():
    success_resp = MagicMock()
    success_resp.text = "Gemini answer on retry"
    with patch("src.gemini_client.get_gemini_client") as mock_get_client:
        with patch("time.sleep"):
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = [Exception(WSARECV_ERROR), success_resp]
            mock_get_client.return_value = mock_client
            text, err, status = call_gemini_with_quota_protection(
                prompt="Test prompt", model_id="gemini-3.5-flash-lite", api_key=VALID_KEY
            )
    assert status == STATUS_CONNECTED, f"Expected CONNECTED after retry, got '{status}'"
    assert text == "Gemini answer on retry"
    assert err is None
    assert mock_client.models.generate_content.call_count == 2


# ---- Fix 2: get_available_gemini_models loop ----

FAKE_KEY = "AIzaSyMockKeyForUnitTestingOnly12345"


def _patch_and_call_model_listing(model_names):
    fake_models = [_fake_model(n) for n in model_names]
    with patch("src.gemini_client.get_gemini_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.models.list.return_value = fake_models
        mock_get_client.return_value = (mock_client, None)
        return get_available_gemini_models(api_key=FAKE_KEY)


def test_all_three_flash_models_collected():
    names = ["gemini-2.0-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]
    result = _patch_and_call_model_listing(names)
    assert "gemini-2.0-flash" in result, f"Missing gemini-2.0-flash in {result}"
    assert "gemini-3.5-flash" in result, f"Missing gemini-3.5-flash in {result}"
    assert "gemini-3.5-flash-lite" in result, f"Missing gemini-3.5-flash-lite in {result}"

def test_five_flash_models_all_collected():
    names = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]
    result = _patch_and_call_model_listing(names)
    assert len(result) == 5, f"Expected 5 models, got {len(result)}: {result}"

def test_preferred_lite_model_at_index_0():
    names = ["gemini-2.0-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]
    result = _patch_and_call_model_listing(names)
    assert result[0] == "gemini-3.5-flash-lite", f"Expected lite at [0], got '{result[0]}'"

def test_fallback_preferred_model_when_lite_absent():
    names = ["gemini-2.0-flash", "gemini-3.5-flash"]
    result = _patch_and_call_model_listing(names)
    assert result[0] == "gemini-3.5-flash", f"Expected flash at [0], got '{result[0]}'"

def test_non_flash_models_excluded():
    names = ["gemini-3.5-flash", "gemini-3.0-pro", "gemini-3.5-flash-tts",
             "gemini-ultra", "gemini-3.5-flash-image", "gemini-3.5-flash-native"]
    result = _patch_and_call_model_listing(names)
    for excluded in ["gemini-3.0-pro", "gemini-3.5-flash-tts", "gemini-ultra",
                     "gemini-3.5-flash-image", "gemini-3.5-flash-native"]:
        assert excluded not in result, f"'{excluded}' should be filtered out"

def test_empty_model_list_returns_supported_defaults():
    from src.config import SUPPORTED_GEMINI_MODELS
    result = _patch_and_call_model_listing([])
    assert result == SUPPORTED_GEMINI_MODELS

def test_missing_key_never_calls_sdk():
    from src.config import SUPPORTED_GEMINI_MODELS
    with patch("src.gemini_client.get_gemini_client") as mock_get_client:
        result = get_available_gemini_models(api_key="")
        mock_get_client.assert_not_called()
        assert result == SUPPORTED_GEMINI_MODELS
