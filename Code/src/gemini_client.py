"""Centralized Google Gemini Client, Rate-Limiting, Caching, and Connection Manager.
Uses the official google-genai SDK.
Handles authentication, model verification, rate limits (HTTP 429), bounded retries,
and persistent quota protection without multiple model cycling.
"""

import hashlib
import json
import re
import time
from typing import Optional, Tuple, Dict, Any, List, Union

import importlib.util

# Detect SDK availability without importing (avoids ~19s cold-start penalty)
HAS_GENAI_SDK = importlib.util.find_spec("google.genai") is not None

# Lazy-loaded SDK references (populated on first use)
_genai_module = None
_genai_types = None
_ClientError = Exception
_APIError = Exception


def _import_genai():
    """Lazily import google.genai on first use to avoid slow startup."""
    global _genai_module, _genai_types, _ClientError, _APIError
    if _genai_module is not None:
        return True
    try:
        from google import genai as _g
        from google.genai import types as _t
        from google.genai.errors import ClientError as _CE, APIError as _AE
        _genai_module = _g
        _genai_types = _t
        _ClientError = _CE
        _APIError = _AE
        return True
    except Exception:
        return False

# Standardized UI Status Labels (8 Mutually Exclusive States)
STATUS_KEY_MISSING = "Not configured"
STATUS_NOT_TESTED = "Configured, not tested"
STATUS_CONNECTED = "Connected"
STATUS_RATE_LIMITED = "Quota/rate limit exceeded"
STATUS_AUTH_FAILED = "Authentication failed"
STATUS_MODEL_UNAVAILABLE = "Model unavailable"
STATUS_SERVICE_UNAVAILABLE = "Service unavailable"
STATUS_CONNECTION_FAILED = "Connection failed"

# Global Singletons
_CLIENT_INSTANCE: Optional[Any] = None
_CLIENT_KEY: Optional[str] = None

# Rate Limit & Quota Tracking
_QUOTA_BLOCKED_UNTIL: float = 0.0
_QUOTA_STATUS_REASON: str = ""

# Bounded In-Memory Response Cache: Dict[cache_key, {"text": str, "timestamp": float}]
_RESPONSE_CACHE: Dict[str, Dict[str, Any]] = {}
MAX_CACHE_SIZE = 100

# Call Origin Tracking (Live API vs Cache vs Mock)
_LAST_CALL_IS_LIVE: bool = False
_LAST_CALL_IS_MOCK: bool = False
_LAST_CALL_IS_CACHED: bool = False


def was_last_call_live() -> bool:
    """Return True if the most recent API call was a verified live response."""
    return _LAST_CALL_IS_LIVE


def was_last_call_cached() -> bool:
    """Return True if the most recent API call was fulfilled from in-memory cache."""
    return _LAST_CALL_IS_CACHED


def was_last_call_mock() -> bool:
    """Return True if the most recent API call executed against a mock or test double."""
    return _LAST_CALL_IS_MOCK


INVALID_KEY_PLACEHOLDERS = {
    "your_api_key_here",
    "your_google_api_key_here",
    "your_gemini_api_key_here",
    "paste_your_new_key_here",
    "replace_with_a_supported_model_name",
    "your_key_here",
    "enter_your_key_here",
    "your_local_api_key",
}


def get_gemini_client(api_key: Optional[str] = None, force_new: bool = False) -> Tuple[Optional[Any], Optional[str]]:
    """Return a singleton instance of google.genai.Client or create one."""
    global _CLIENT_INSTANCE, _CLIENT_KEY
    if not HAS_GENAI_SDK:
        return None, "google-genai SDK is not installed."
    import os
    key_clean = (api_key or "").strip()
    if not key_clean:
        key_clean = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not key_clean or len(key_clean) < 15 or key_clean.lower() in INVALID_KEY_PLACEHOLDERS:
        return None, "No API key configured."

    if not force_new and _CLIENT_INSTANCE is not None and _CLIENT_KEY == key_clean:
        return _CLIENT_INSTANCE, None

    try:
        if not _import_genai():
            return None, "Failed to import google-genai SDK."
        client = _genai_module.Client(api_key=key_clean)
        _CLIENT_INSTANCE = client
        _CLIENT_KEY = key_clean
        return _CLIENT_INSTANCE, None
    except Exception as exc:
        _CLIENT_INSTANCE = None
        _CLIENT_KEY = None
        err_msg = str(exc)
        if "api_key" in err_msg.lower() or "invalid" in err_msg.lower() or "rejected" in err_msg.lower():
            return None, "The provided Google API key was rejected by the Gemini API."
        return None, f"Failed to initialize Gemini Client: {err_msg}"


def is_quota_blocked() -> Tuple[bool, str, int]:
    """Check if the API is currently under a known active quota or rate-limit cooldown.
    Returns: (is_blocked, reason_message, seconds_remaining)
    """
    now = time.time()
    if now < _QUOTA_BLOCKED_UNTIL:
        remaining = int(_QUOTA_BLOCKED_UNTIL - now)
        return True, _QUOTA_STATUS_REASON, max(1, remaining)
    return False, "", 0


def set_quota_blocked(duration_seconds: float, reason: str):
    """Mark quota/rate-limit cooldown active for the specified duration."""
    global _QUOTA_BLOCKED_UNTIL, _QUOTA_STATUS_REASON
    _QUOTA_BLOCKED_UNTIL = time.time() + max(10.0, duration_seconds)
    _QUOTA_STATUS_REASON = reason


def clear_quota_block():
    """Clear active quota block (e.g. after successful verified request)."""
    global _QUOTA_BLOCKED_UNTIL, _QUOTA_STATUS_REASON
    _QUOTA_BLOCKED_UNTIL = 0.0
    _QUOTA_STATUS_REASON = ""


def compute_evidence_cache_key(question: str, doc_scope: Optional[str], retrieved_chunks: List[Dict[str, Any]], model_id: str) -> str:
    """Generate a deterministic, tamper-resistant cache key for question and retrieved evidence passages."""
    chunk_signatures = []
    for c in retrieved_chunks:
        cid = c.get("chunk_id", "")
        meta = c.get("metadata", {})
        doc = meta.get("filename", "")
        page = meta.get("page_num", 0)
        text_snippet = c.get("text", "")[:100]
        chunk_signatures.append(f"{cid}:{doc}:{page}:{hash(text_snippet)}")

    payload = {
        "question": question.strip().lower(),
        "scope": (doc_scope or "all").strip().lower(),
        "model": model_id.strip(),
        "evidence": sorted(chunk_signatures),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_cached_answer(cache_key: str) -> Optional[str]:
    """Retrieve cached grounded response if available."""
    entry = _RESPONSE_CACHE.get(cache_key)
    if entry and "text" in entry:
        return entry["text"]
    return None


def store_cached_answer(cache_key: str, text: str):
    """Store successful grounded response in bounded cache."""
    global _RESPONSE_CACHE
    if len(_RESPONSE_CACHE) >= MAX_CACHE_SIZE:
        # Evict oldest entry
        oldest_key = min(_RESPONSE_CACHE.keys(), key=lambda k: _RESPONSE_CACHE[k].get("timestamp", 0))
        _RESPONSE_CACHE.pop(oldest_key, None)
    _RESPONSE_CACHE[cache_key] = {
        "text": text,
        "timestamp": time.time(),
    }


def parse_retry_delay_from_error(error_str: str) -> float:
    """Extract recommended retry backoff in seconds from error text or headers."""
    # Look for seconds e.g. "retryDelay': '21240s'" or "retry in 5h54m"
    s_match = re.search(r"retryDelay['\":\s]+(\d+)", error_str, re.IGNORECASE)
    if s_match:
        try:
            return float(s_match.group(1))
        except ValueError:
            pass

    # Look for human-readable "retry in Xm" or "Xs"
    m_match = re.search(r"retry in (\d+)m", error_str, re.IGNORECASE)
    if m_match:
        try:
            return float(m_match.group(1)) * 60.0
        except ValueError:
            pass

    # Default rate-limit cooldown: 60 seconds
    return 60.0


def call_gemini_with_quota_protection(
    prompt: str,
    model_id: str,
    api_key: str,
    cache_key: Optional[str] = None,
    timeout_seconds: float = 30.0,
    force_new_client: bool = False,
) -> Tuple[Optional[str], Optional[str], str]:
    """Execute Gemini generate_content call with quota tracking, timeouts, bounded retries, and caching.
    Returns: (generated_text, error_message, status_category)
    """
    global _LAST_CALL_IS_LIVE, _LAST_CALL_IS_MOCK, _LAST_CALL_IS_CACHED
    _LAST_CALL_IS_LIVE = False
    _LAST_CALL_IS_MOCK = False
    _LAST_CALL_IS_CACHED = False

    key_clean = (api_key or "").strip()
    if not key_clean or len(key_clean) < 15 or key_clean.lower() in INVALID_KEY_PLACEHOLDERS:
        return None, "No Gemini API key configured in .env or secrets.", STATUS_KEY_MISSING

    # 1. Check cache first
    if cache_key:
        cached = get_cached_answer(cache_key)
        if cached:
            _LAST_CALL_IS_CACHED = True
            return cached, None, STATUS_CONNECTED

    # 2. Check if under active quota cooldown
    blocked, block_msg, remaining = is_quota_blocked()
    if blocked:
        return None, f"Active quota limit: {block_msg} (Cooldown remaining: {remaining}s).", STATUS_RATE_LIMITED

    # 3. Get or initialize client
    client_res = get_gemini_client(key_clean, force_new=force_new_client)
    if isinstance(client_res, tuple) and len(client_res) == 2:
        client, init_err = client_res
    else:
        client, init_err = client_res, None
    if client is None:
        if not HAS_GENAI_SDK:
            return None, "google-genai SDK is not installed.", STATUS_CONNECTION_FAILED
        if init_err and ("rejected" in init_err.lower() or "invalid" in init_err.lower() or "api_key" in init_err.lower()):
            return None, init_err, STATUS_AUTH_FAILED
        return None, init_err or "Failed to initialize Gemini Client.", STATUS_AUTH_FAILED

    # 4. Attempt execution with bounded retry for transient 503 only
    max_transient_retries = 1
    for attempt in range(max_transient_retries + 1):
        try:
            # Detect mock clients or method test doubles
            is_mock_target = False
            try:
                from unittest.mock import Mock
                if (
                    isinstance(client, Mock)
                    or isinstance(getattr(client, "models", None), Mock)
                    or isinstance(getattr(getattr(client, "models", None), "generate_content", None), Mock)
                    or hasattr(client, "_mock_return_value")
                ):
                    is_mock_target = True
            except Exception:
                pass

            # Official google-genai generation call
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
            )
            if response and response.text:
                clear_quota_block()
                result_text = response.text.strip()
                if cache_key:
                    store_cached_answer(cache_key, result_text)

                # Check if response object is a mock double
                try:
                    from unittest.mock import Mock
                    if isinstance(response, Mock) or hasattr(response, "_mock_return_value"):
                        is_mock_target = True
                except Exception:
                    pass

                if is_mock_target:
                    _LAST_CALL_IS_MOCK = True
                    _LAST_CALL_IS_LIVE = False
                else:
                    _LAST_CALL_IS_LIVE = True
                    _LAST_CALL_IS_MOCK = False

                return result_text, None, STATUS_CONNECTED
            return None, "Empty response returned by Gemini model.", STATUS_CONNECTION_FAILED

        except Exception as exc:
            err_text = str(exc)
            err_lower = err_text.lower()
            status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)

            # A. Rate limit / Quota exceeded (HTTP 429) -> DO NOT RETRY!
            if status_code == 429 or "429" in err_lower or "resource_exhausted" in err_lower or "too many requests" in err_lower or "quota" in err_lower:
                delay = parse_retry_delay_from_error(err_text)
                set_quota_blocked(delay, f"Quota limit reached for model '{model_id}'")
                safe_err = f"Rate limit / daily quota exceeded for '{model_id}'. Requests paused to protect quota."
                return None, safe_err, STATUS_RATE_LIMITED

            # B. Model unavailable / Not found (HTTP 404) -> DO NOT RETRY! DO NOT SWITCH MODEL!
            if status_code == 404 or "404" in err_lower or "not_found" in err_lower or "not found" in err_lower or "no longer available" in err_lower or "not supported" in err_lower:
                safe_err = f"Configured model '{model_id}' is not available or not supported on this project. Update GEMINI_MODEL in .env."
                return None, safe_err, STATUS_MODEL_UNAVAILABLE

            # C. Authentication failure (HTTP 401 / 403) -> DO NOT RETRY!
            if status_code in (401, 403) or "401" in err_lower or "403" in err_lower or "unauthenticated" in err_lower or "permission_denied" in err_lower or "invalid api key" in err_lower or "api_key_invalid" in err_lower:
                safe_err = "Authentication failed: The provided Google API key was rejected."
                return None, safe_err, STATUS_AUTH_FAILED

            # D. Server / Temporary Availability Errors (HTTP 500, 502, 503, 504) -> Bounded single retry for 503
            if status_code in (500, 502, 503, 504) or any(s in err_lower for s in ["500", "502", "503", "504", "service unavailable", "overloaded", "backend error"]):
                if attempt < max_transient_retries and (status_code == 503 or "503" in err_lower or "unavailable" in err_lower):
                    time.sleep(2.0)
                    continue
                safe_err = f"Gemini API service temporarily unavailable (HTTP {status_code or 503})."
                return None, safe_err, STATUS_SERVICE_UNAVAILABLE

            # E. Invalid Request / Parameters (HTTP 400) -> DO NOT RETRY
            # Note: do NOT match bare "400" string – it appears in IPv6 addresses like [2001:4860:4843:400::].
            if status_code == 400 or "invalid_argument" in err_lower or "bad request" in err_lower:
                safe_err = f"Invalid request or unsupported parameters for '{model_id}'."
                return None, safe_err, STATUS_CONNECTION_FAILED

            # F. Network Timeout / Connection Error (incl. Windows wsarecv / connection aborted) -> Bounded single retry
            _network_signals = (
                "timeout", "timed out", "connection reset", "connection refused",
                "connection aborted", "wsarecv", "wsasend", "network error",
                "eof occurred", "broken pipe", "connection error", "stream reading error",
                "ssl", "handshake",
            )
            if any(sig in err_lower for sig in _network_signals):
                if attempt < max_transient_retries:
                    time.sleep(3.0)
                    continue
                safe_err = "Network connection to Gemini API was interrupted (possible firewall, antivirus, or VPN interference). Please retry."
                return None, safe_err, STATUS_CONNECTION_FAILED

            safe_err = f"API error ({type(exc).__name__}): {err_text[:120]}"
            return None, safe_err, STATUS_CONNECTION_FAILED

    return None, "Gemini API request failed after retry.", STATUS_SERVICE_UNAVAILABLE


def test_gemini_connection(api_key: Optional[Union[str, Any]] = None, model_name: Optional[str] = None) -> Tuple[str, str]:
    """Execute a manual live lightweight connection verification request.
    Only called when explicitly invoked by the user or test runner.
    Returns: (status_label, message)
    """
    from src.config import get_settings
    settings = get_settings()

    if api_key is not None:
        if isinstance(api_key, str):
            key = api_key.strip()
            target_model = (model_name or settings.gemini_model or "").strip()
        elif hasattr(api_key, "google_api_key"):
            key = (getattr(api_key, "google_api_key", "") or "").strip()
            target_model = (model_name or getattr(api_key, "gemini_model", settings.gemini_model) or "").strip()
        else:
            key = str(api_key).strip()
            target_model = (model_name or settings.gemini_model or "").strip()
    else:
        key = (settings.google_api_key or "").strip()
        target_model = (model_name or settings.gemini_model or "").strip()

    if not key or len(key) < 15 or key.lower() in INVALID_KEY_PLACEHOLDERS:
        return STATUS_KEY_MISSING, "No API key configured in .env or Streamlit secrets."

    text, err, status = call_gemini_with_quota_protection(
        prompt="Ping: respond with PONG",
        model_id=target_model,
        api_key=key,
        cache_key=None,
        force_new_client=True,
    )

    if status == STATUS_CONNECTED and text:
        return "Connected", f"Successfully connected to {target_model}."

    return status, (err or f"Connection test failed for '{target_model}'.")


def get_available_gemini_models(api_key: Optional[str] = None) -> List[str]:
    """Dynamically query available models using the official Google GenAI SDK.
    Gracefully handles permissions and errors without failing.
    """
    from src.config import get_settings, SUPPORTED_GEMINI_MODELS
    settings = get_settings()
    # Use api_key if explicitly provided (even if empty), otherwise fall back to settings.
    # An explicit empty string means "no key" — do NOT fall through to the real settings key.
    if api_key is None:
        key = (settings.google_api_key or "").strip()
    else:
        key = api_key.strip()
    if not key or len(key) < 15 or key.lower() in {"your_api_key_here", "paste_your_new_key_here"}:
        return SUPPORTED_GEMINI_MODELS
    try:
        client, _ = get_gemini_client(key)
        if client is not None:
            models = list(client.models.list())
            discovered = []
            for m in models:
                m_name = getattr(m, "name", "") or ""
                if "/" in m_name:
                    m_name = m_name.split("/")[-1]
                if "flash" in m_name and not any(tag in m_name for tag in ["tts", "image", "audio", "native", "live"]):
                    discovered.append(m_name)
            # Prioritise preferred lite model at the top of the list
            if "gemini-3.5-flash-lite" in discovered:
                discovered.remove("gemini-3.5-flash-lite")
                discovered.insert(0, "gemini-3.5-flash-lite")
            elif "gemini-3.5-flash" in discovered:
                discovered.remove("gemini-3.5-flash")
                discovered.insert(0, "gemini-3.5-flash")
            if discovered:
                return discovered
    except Exception:
        pass
    return SUPPORTED_GEMINI_MODELS
