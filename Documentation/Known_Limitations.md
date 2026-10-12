# Known Limitations & Responsible AI Disclosure

## 1. Scope & Operational Limitations

### 1.1 Non-Certification Disclaimer
The **AUTOSAR HLD Document Analysis Assistant** is an engineering review accelerator. It **does not certify, approve, or warrant** automotive software compliance with ISO 26262, ASPICE, or official AUTOSAR specifications. All identified findings, entity definitions, and diff reports must be independently audited by certified automotive safety and systems engineers.

### 1.2 Diagram and Image Parsing
- High-Level Design documents often contain complex architectural block diagrams, UML sequence diagrams, and SysML diagrams.
- PyMuPDF and OCR extract textual elements but cannot perform full semantic graph reconstruction of visual connections without dedicated multimodal vision models.
- Architectural relationships are extracted primarily from textual descriptions, signal tables, and port mapping dictionaries.

### 1.3 OCR Support — Capabilities and Limitations

#### What OCR Can Do
- When Tesseract is installed, the application automatically detects scanned or image-only PDF pages (fewer than 30 extractable characters) and applies OCR as a fallback.
- Successfully extracted OCR text is integrated into the normal chunking, embedding, and ChromaDB indexing pipeline.
- All OCR-extracted content is traced back to the original PDF page number and carries the `extraction_method: "ocr"` metadata.

#### What OCR Cannot Do
- **Diagram semantics**: OCR extracts machine-readable text from pixel data. It does not reconstruct block diagram connections, signal-flow arrows, port-to-port links, or graphical connector relationships.
- **Table layout preservation**: Complex table structures in rasterized images may be extracted as unstructured text without preserved row/column relationships.
- **Text within graphics**: Text embedded in architectural diagrams (e.g., block names inside coloured boxes) may not be reliably identified or correctly ordered.
- **Verified truth**: OCR output is raw pixel-to-text conversion. It has not been reviewed or validated by an engineer and must not be treated as certified engineering truth.
- **ISO 26262 / AUTOSAR compliance**: Extracted content does not constitute certified AUTOSAR or ISO 26262 compliance evidence regardless of extraction method.

#### Operational Behaviour When Tesseract is Unavailable
- If Tesseract is not installed, text-based PDFs continue working completely normally.
- Scanned or low-text pages produce a `scanned_page_unextracted` status and a contextual warning in the Documents upload section — the application does not crash.
- No permanent OCR warning is shown on the Overview or Evaluation pages.

### 1.4 LLM Cloud Dependencies & Quotas
- Live natural-language grounded question answering requires access to the Google Gemini API over HTTPS.
- Rate limits, token quotas, and internet availability depend on the user's Google Cloud project billing and tier.
- In offline or unauthenticated environments, the application seamlessly defaults to **Retrieval-Only Mode**, displaying retrieved evidence chunks and exact page citations without conversational synthesis.

### 1.5 Local Embeddings First-Time Download
- The embedding model (`all-MiniLM-L6-v2`) runs 100% locally on CPU/GPU.
- The first time the model runs, weights (~90 MB) are downloaded from Hugging Face Hub. Once cached in `~/.cache/huggingface`, it operates completely offline.

---

## 2. Responsible AI & Data Privacy Guidelines
1. **Confidentiality**: Never upload confidential, proprietary, or OEM-restricted automotive specifications to third-party APIs without organizational compliance approval.
2. **Anti-Hallucination Guardrails**: The RAG prompt strictly instructs the LLM to abstain if context evidence is missing. The system programmatically intercepts out-of-domain queries.
3. **Traceability**: All factual statements are accompanied by verifiable `[Doc: ..., Page: ..., Section: ...]` citations pointing to actual retrieved passages.
