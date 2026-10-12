# Model and Pipeline Configuration

## 1. Overview
This document records the exact configuration parameters, model selections, and operational constraints for the **AUTOSAR HLD Document Analysis Assistant**.

---

## 2. LLM Provider Configuration
- **Provider**: Google Gemini API
- **SDK**: Current Official Google Gen AI SDK (`google-genai` / `google.genai`)
- **Configured Model**: `gemini-1.5-flash` (Configurable via `GEMINI_MODEL` environment variable to other account-accessible models such as `gemini-2.0-flash` or `gemini-1.5-pro`)
- **Authentication**: `GOOGLE_API_KEY` stored strictly in local `.env` and loaded via `python-dotenv`.
- **Secret Safety Policy**:
  - API keys are never hardcoded in source code, prompts, UI elements, or committed files.
  - The application only reports binary key status: `Configured (Present)` or `Not Configured (Absent)`.
  - In retrieval-only mode, the system operates deterministically without external LLM calls.

---

## 3. Embedding Model Configuration
- **Embedding Provider**: HuggingFace Sentence-Transformers (Local execution)
- **Primary Model**: `all-MiniLM-L6-v2`
- **Embedding Dimensions**: 384 dimensions
- **Inference Hardware**: Local CPU (or CUDA GPU if available)
- **First-Time Download Note**: First run requires an active internet connection to cache the weights into local HuggingFace cache (`~/.cache/huggingface/hub`). Subsequent runs operate completely offline.

---

## 4. Vector Database Configuration
- **Vector Database**: ChromaDB (Embedded Persistent Vector Store)
- **Storage Location**: Local directory (`./chroma_db`)
- **Collection Name**: `autosar_hld_chunks`
- **Distance Metric**: Cosine Distance
- **Metadata Preserved in Every Chunk**:
  - `doc_id`: Unique identifier (SHA-256 hash prefix)
  - `filename`: Original PDF document name
  - `page_num`: 1-indexed document page number
  - `section`: Heading or section title
  - `chunk_index`: Integer index of chunk within the document
  - `doc_hash`: Full SHA-256 digest of source PDF
  - `extraction_method`: `pdf_text`, `ocr`, or `fallback`
  - `char_count`: Character length of the chunk

---

## 5. Chunking and Retrieval Hyperparameters
- **Chunk Size**: 500 characters / target paragraph block
- **Chunk Overlap**: 100 characters
- **Chunking Strategy**: Context-aware hierarchical boundary preservation (preserves section titles, SWC blocks, port dictionaries, and tables)
- **Retrieval Top-K**: 4 chunks
- **Abstention Policy**: If no retrieved chunk meets relevance threshold or user asks about unmentioned components, model returns explicit abstention rather than hallucinating.

---

## 6. OCR Configuration
- **Engine**: Tesseract OCR via `pytesseract` and `Pillow`
- **Activation Condition**: Only executed on pages containing fewer than 30 characters of native extractable text.
- **Handling Missing OCR**: Gracefully skipped with clear user warning if Tesseract binary is not installed on the host system.
