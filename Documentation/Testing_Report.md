# Automated Testing Report — AUTOSAR HLD Document Analysis Assistant

## 1. Test Suite Summary
- **Test Framework**: `pytest` 9.1.1
- **Total Test Cases**: 23
- **Passed**: 23
- **Failed**: 0
- **Pass Rate**: 100.0%
- **Execution Time**: ~29.96 seconds
- **Platform**: Windows 11 (Python 3.13.0)

---

## 2. Test Execution Details by Module

| Test Module | Test Case Name | Verified Behavior | Status |
|---|---|---|---|
| `tests/test_ingestion.py` | `test_compute_file_hash` | SHA-256 stable hashing of binary PDF content | **PASSED** |
| `tests/test_ingestion.py` | `test_normalize_text` | Non-breaking spaces, CRLF normalization, whitespace cleanup | **PASSED** |
| `tests/test_ingestion.py` | `test_extract_headings` | Pattern matching of AUTOSAR section titles with punctuation | **PASSED** |
| `tests/test_ingestion.py` | `test_ingest_valid_pdf` | Page extraction, 1-indexed numbering, metadata preservation | **PASSED** |
| `tests/test_ingestion.py` | `test_ingest_empty_bytes_raises` | Graceful error on 0-byte file upload | **PASSED** |
| `tests/test_ingestion.py` | `test_ingest_invalid_pdf_signature_raises` | Corrupted PDF header rejection | **PASSED** |
| `tests/test_ingestion.py` | `test_ocr_availability_check` | Safe host Tesseract detection without crashing | **PASSED** |
| `tests/test_chunking.py` | `test_chunking_metadata_and_stable_ids` | Hierarchical chunk splitting with stable IDs and metadata dict | **PASSED** |
| `tests/test_chunking.py` | `test_chunking_respects_sections` | Section boundary preservation during paragraph batching | **PASSED** |
| `tests/test_vector_store.py` | `test_vector_store_indexing_and_duplicate_prevention` | ChromaDB vector storage and duplicate upload suppression | **PASSED** |
| `tests/test_vector_store.py` | `test_vector_store_retrieval_and_filtering` | Cosine similarity ranking and doc_id filtering | **PASSED** |
| `tests/test_vector_store.py` | `test_vector_store_safe_deletion` | Atomic document removal without index corruption | **PASSED** |
| `tests/test_rag_pipeline.py` | `test_rag_retrieval_and_citations` | Retrieval-only fallback and citation generation | **PASSED** |
| `tests/test_rag_pipeline.py` | `test_rag_abstention_on_unsupported_query` | Grounded abstention on out-of-domain questions | **PASSED** |
| `tests/test_entity_extractor.py` | `test_entity_extraction_from_pdf` | Pydantic extraction of Components, Ports, Interfaces, Signals | **PASSED** |
| `tests/test_doc_comparison.py` | `test_revision_comparison_v1_vs_v2` | Section diffing and entity addition/deletion detection | **PASSED** |
| `tests/test_issue_detector.py` | `test_issue_detection_finds_deliberate_inconsistencies` | Detection of conflicting timing (10ms vs 50ms) and undefined interfaces | **PASSED** |
| `tests/test_export.py` | `test_export_dataframe_to_csv` | CSV string serialization | **PASSED** |
| `tests/test_export.py` | `test_export_dataframe_to_json` | Formatted JSON array serialization | **PASSED** |
| `tests/test_export.py` | `test_export_dict_to_json` | Dictionary JSON export | **PASSED** |
| `tests/test_gemini_mock.py` | `test_missing_api_key_handling` | Graceful fallback when GOOGLE_API_KEY is empty | **PASSED** |
| `tests/test_gemini_mock.py` | `test_mocked_gemini_successful_response` | Citation extraction and validation on LLM output | **PASSED** |
| `tests/test_gemini_mock.py` | `test_mocked_gemini_abstention_response` | LLM abstention handling and status flagging | **PASSED** |

---

## 3. Mock Test vs Live Test Disclosure
- **Mock Tests**: Demonstrate prompt construction, citation parsing, error trapping, and fallback modes under deterministic conditions.
- **Live Gemini Connectivity**: Independently tested via `scripts/check_gemini_connection.py`. Mock tests are never presented as proof of live API connectivity.
