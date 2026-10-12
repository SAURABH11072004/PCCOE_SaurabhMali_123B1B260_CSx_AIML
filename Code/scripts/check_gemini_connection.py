#!/usr/bin/env python3
"""Gemini API Connectivity Verification Utility.
Safely tests credentials and connectivity to configured Google Gemini model.
NEVER prints or leaks the API key in terminal output or logs.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings, test_gemini_connection


def check_gemini_connection() -> int:
    settings = get_settings()
    print("=" * 65)
    print("AUTOSAR HLD ASSISTANT — GEMINI CONNECTIVITY CHECK")
    print("=" * 65)

    print(f"Target LLM Model       : {settings.gemini_model}")
    print(f"Key Configured         : {'YES' if settings.is_gemini_configured else 'NO'}")

    status, message = test_gemini_connection()
    print(f"Verification Status    : {status}")
    print(f"Details                : {message}")
    print("=" * 65)

    if status == "Connected":
        print("[RESULT] SUCCESS: Gemini API connection is operational.")
        return 0
    elif status == "Not configured":
        print("[RESULT] NOTICE: Operates in Retrieval-Only Mode (Local Vector Search).")
        print("To enable Gemini generation, set GOOGLE_API_KEY in .env or Streamlit secrets.")
        return 1
    else:
        print(f"[RESULT] FAILED: {status} — {message}")
        return 2


if __name__ == "__main__":
    sys.exit(check_gemini_connection())
