"""PDF Ingestion and Document Processing Module.
Extracts page-by-page text, headings, tables, and metadata using PyMuPDF.
Includes graceful OCR fallback detection and stable SHA-256 document hashing.
"""

import hashlib
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pytesseract
    from PIL import Image
    import io
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False


@dataclass
class ExtractedTable:
    page_number: int
    headers: List[str]
    rows: List[List[str]]
    raw_markdown: str


@dataclass
class PageData:
    page_number: int  # 1-indexed
    text: str
    headings: List[str]
    tables: List[ExtractedTable] = field(default_factory=list)
    extraction_method: str = "pdf_text"  # "pdf_text", "ocr", "empty"
    ocr_status: str = "Not needed"
    char_count: int = 0


@dataclass
class DocumentData:
    doc_id: str
    filename: str
    file_hash: str
    total_pages: int
    pages: List[PageData]
    metadata: Dict[str, Any] = field(default_factory=dict)
    ocr_enabled: bool = False
    processing_notes: List[str] = field(default_factory=list)


def check_tesseract_available() -> bool:
    """Check if Tesseract binary is available on system PATH, env var, or default Windows paths."""
    if not HAS_PYTESSERACT:
        return False
    # 1. Configurable environment variable
    tess_env = os.environ.get("TESSERACT_CMD") or os.environ.get("TESSERACT_PATH")
    if tess_env and Path(tess_env).exists():
        pytesseract.pytesseract.tesseract_cmd = str(Path(tess_env))
        return True
    # 2. System PATH
    if shutil.which("tesseract"):
        return True
    # 3. Common Windows installation paths
    common_win_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        common_win_paths.append(str(Path(local_app_data) / "Programs" / "Tesseract-OCR" / "tesseract.exe"))
    for p in common_win_paths:
        if Path(p).exists():
            pytesseract.pytesseract.tesseract_cmd = p
            return True
    return False


def compute_file_hash(file_bytes: bytes) -> str:
    """Compute stable SHA-256 digest of document bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def normalize_text(text: str) -> str:
    """Normalize whitespace and control characters without corrupting layout."""
    if not text:
        return ""
    # Replace non-breaking spaces and irregular unicode spaces
    text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    # Condense 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Condense repeated horizontal spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_headings(page_text: str) -> List[str]:
    """Identify section headings using standard AUTOSAR engineering patterns."""
    headings = []
    lines = page_text.split("\n")
    patterns = [
        r"^(?:Section\s+\d+|[0-9]+(?:\.[0-9]+)*)\s*[:\-]?\s*[A-Z][A-Za-z0-9_\s\&\(\)\/\,\-]{2,75}$",
        r"^[A-Z][A-Z0-9_\s\&\(\)\-]{3,60}:$",
        r"^(?:Software Components|Port Definitions|Interface Dictionary|Functional Flows|Signal Dictionary|Dependencies|Cyber Security|Safety Concept)",
    ]
    for line in lines:
        line_clean = line.strip()
        if not line_clean or len(line_clean) > 85:
            continue
        for pat in patterns:
            if re.match(pat, line_clean, re.IGNORECASE):
                headings.append(line_clean)
                break
    return list(dict.fromkeys(headings))  # Deduplicate preserving order


class PDFIngestionPipeline:
    def __init__(self, enable_ocr: bool = True):
        self.enable_ocr = enable_ocr
        self.tesseract_available = check_tesseract_available()

    def process_pdf(self, file_bytes: bytes, filename: str) -> DocumentData:
        """Validate and extract structured pages, headings, and metadata from PDF bytes."""
        if not file_bytes:
            raise ValueError(f"Uploaded file '{filename}' is empty (0 bytes).")
        
        if len(file_bytes) < 8 or not file_bytes.startswith(b"%PDF"):
            raise ValueError(f"Uploaded file '{filename}' is not a valid PDF document.")
        
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is required for PDF ingestion.")

        file_hash = compute_file_hash(file_bytes)
        doc_id = f"doc_{file_hash[:12]}"

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise ValueError(f"Corrupted or unreadable PDF '{filename}': {str(e)}")

        total_pages = len(doc)
        if total_pages == 0:
            raise ValueError(f"PDF '{filename}' contains 0 pages.")

        pages_data: List[PageData] = []
        processing_notes = []

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]
            page_text = page.get_text("text") or ""
            page_text_clean = normalize_text(page_text)
            
            extraction_method = "pdf_text"
            ocr_status = "Not needed"
            
            # Low text detection (< 30 characters)
            if len(page_text_clean) < 30:
                if self.enable_ocr and self.tesseract_available:
                    try:
                        pix = page.get_pixmap(dpi=150)
                        img_data = pix.tobytes("png")
                        pil_img = Image.open(io.BytesIO(img_data))
                        ocr_result = pytesseract.image_to_string(pil_img)
                        ocr_clean = normalize_text(ocr_result)
                        if len(ocr_clean) > len(page_text_clean):
                            page_text_clean = ocr_clean
                            extraction_method = "ocr"
                            ocr_status = "Successfully executed OCR"
                        else:
                            ocr_status = "OCR yielded no additional text"
                    except Exception as ocr_err:
                        ocr_status = f"OCR Error: {str(ocr_err)}"
                        processing_notes.append(f"Page {page_num} OCR failed: {str(ocr_err)}")
                elif self.enable_ocr and not self.tesseract_available:
                    ocr_status = "Skipped (Tesseract binary not installed on host system)"
                    extraction_method = "scanned_page_unextracted"
                else:
                    ocr_status = "OCR disabled by user"
                    if not page_text_clean:
                        extraction_method = "empty"

            headings = extract_headings(page_text_clean)

            # Table extraction using PyMuPDF if available
            extracted_tables = []
            try:
                tabs = page.find_tables()
                if tabs and tabs.tables:
                    for tab in tabs.tables:
                        tab_data = tab.extract()
                        if tab_data and len(tab_data) > 1:
                            headers = [str(c or "").strip() for c in tab_data[0]]
                            rows = [[str(c or "").strip() for c in r] for r in tab_data[1:]]
                            md_rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                            for r in rows:
                                md_rows.append("| " + " | ".join(r) + " |")
                            extracted_tables.append(
                                ExtractedTable(
                                    page_number=page_num,
                                    headers=headers,
                                    rows=rows,
                                    raw_markdown="\n".join(md_rows),
                                )
                            )
            except Exception:
                pass  # Fallback to plain text if table parser fails

            pages_data.append(
                PageData(
                    page_number=page_num,
                    text=page_text_clean,
                    headings=headings,
                    tables=extracted_tables,
                    extraction_method=extraction_method,
                    ocr_status=ocr_status,
                    char_count=len(page_text_clean),
                )
            )

        metadata = {
            "title": doc.metadata.get("title", "") if doc.metadata else "",
            "author": doc.metadata.get("author", "") if doc.metadata else "",
            "subject": doc.metadata.get("subject", "") if doc.metadata else "",
            "creator": doc.metadata.get("creator", "") if doc.metadata else "",
        }

        doc.close()

        return DocumentData(
            doc_id=doc_id,
            filename=filename,
            file_hash=file_hash,
            total_pages=total_pages,
            pages=pages_data,
            metadata=metadata,
            ocr_enabled=self.enable_ocr and self.tesseract_available,
            processing_notes=processing_notes,
        )
