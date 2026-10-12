"""Test that all 8 status constants required by app.py can be imported from
src.config (which re-exports them from src.gemini_client).

This is the canonical guard against the ImportError:
  "cannot import name 'STATUS_SERVICE_UNAVAILABLE' from 'src.config'"

All checks are pure Python imports — no API calls, no Streamlit runtime needed.
"""

import pytest


def test_all_status_constants_importable_from_config():
    """Every constant listed in app.py line 24-39 must be importable from src.config."""
    from src.config import (
        STATUS_KEY_MISSING,
        STATUS_NOT_TESTED,
        STATUS_CONNECTED,
        STATUS_RATE_LIMITED,
        STATUS_MODEL_UNAVAILABLE,
        STATUS_AUTH_FAILED,
        STATUS_SERVICE_UNAVAILABLE,
        STATUS_CONNECTION_FAILED,
    )
    # Verify none are None or empty
    constants = {
        "STATUS_KEY_MISSING": STATUS_KEY_MISSING,
        "STATUS_NOT_TESTED": STATUS_NOT_TESTED,
        "STATUS_CONNECTED": STATUS_CONNECTED,
        "STATUS_RATE_LIMITED": STATUS_RATE_LIMITED,
        "STATUS_MODEL_UNAVAILABLE": STATUS_MODEL_UNAVAILABLE,
        "STATUS_AUTH_FAILED": STATUS_AUTH_FAILED,
        "STATUS_SERVICE_UNAVAILABLE": STATUS_SERVICE_UNAVAILABLE,
        "STATUS_CONNECTION_FAILED": STATUS_CONNECTION_FAILED,
    }
    for name, value in constants.items():
        assert isinstance(value, str) and value.strip(), (
            f"{name} is empty or not a string: {value!r}"
        )


def test_status_constants_are_unique():
    """All 8 status strings must be mutually exclusive (no two constants share a value)."""
    from src.config import (
        STATUS_KEY_MISSING, STATUS_NOT_TESTED, STATUS_CONNECTED,
        STATUS_RATE_LIMITED, STATUS_MODEL_UNAVAILABLE, STATUS_AUTH_FAILED,
        STATUS_SERVICE_UNAVAILABLE, STATUS_CONNECTION_FAILED,
    )
    values = [
        STATUS_KEY_MISSING, STATUS_NOT_TESTED, STATUS_CONNECTED,
        STATUS_RATE_LIMITED, STATUS_MODEL_UNAVAILABLE, STATUS_AUTH_FAILED,
        STATUS_SERVICE_UNAVAILABLE, STATUS_CONNECTION_FAILED,
    ]
    assert len(values) == len(set(values)), (
        f"Duplicate status values detected: {values}"
    )


def test_status_constants_match_gemini_client_originals():
    """src.config must re-export the identical objects defined in src.gemini_client."""
    from src.config import (
        STATUS_KEY_MISSING as cfg_km,
        STATUS_NOT_TESTED as cfg_nt,
        STATUS_CONNECTED as cfg_co,
        STATUS_RATE_LIMITED as cfg_rl,
        STATUS_MODEL_UNAVAILABLE as cfg_mu,
        STATUS_AUTH_FAILED as cfg_af,
        STATUS_SERVICE_UNAVAILABLE as cfg_su,
        STATUS_CONNECTION_FAILED as cfg_cf,
    )
    from src.gemini_client import (
        STATUS_KEY_MISSING as gc_km,
        STATUS_NOT_TESTED as gc_nt,
        STATUS_CONNECTED as gc_co,
        STATUS_RATE_LIMITED as gc_rl,
        STATUS_MODEL_UNAVAILABLE as gc_mu,
        STATUS_AUTH_FAILED as gc_af,
        STATUS_SERVICE_UNAVAILABLE as gc_su,
        STATUS_CONNECTION_FAILED as gc_cf,
    )
    assert cfg_km == gc_km
    assert cfg_nt == gc_nt
    assert cfg_co == gc_co
    assert cfg_rl == gc_rl
    assert cfg_mu == gc_mu
    assert cfg_af == gc_af
    assert cfg_su == gc_su
    assert cfg_cf == gc_cf


def test_all_non_status_names_importable_from_config():
    """Non-status names in the app.py import block must also be importable."""
    from src.config import (
        Settings,
        get_settings,
        test_gemini_connection,
        SUPPORTED_GEMINI_MODELS,
        get_available_gemini_models,
        PROJECT_ROOT,
    )
    assert callable(get_settings)
    assert callable(test_gemini_connection)
    assert callable(get_available_gemini_models)
    assert isinstance(SUPPORTED_GEMINI_MODELS, list)
    assert len(SUPPORTED_GEMINI_MODELS) > 0
    assert PROJECT_ROOT.exists()


def test_app_py_syntax_valid():
    """app.py must compile without syntax errors (does not import or run Streamlit)."""
    import py_compile, os
    app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
    try:
        py_compile.compile(app_path, doraise=True)
    except py_compile.PyCompileError as exc:
        pytest.fail(f"app.py has a syntax error: {exc}")
