"""Shared Data Management and Document Catalog Layer.
Provides consistent, unified access to indexed documents, page texts,
extracted entities, and deduplicated issues across all UI tabs and exports.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from src.config import PROJECT_ROOT
from src.vector_store import VectorStoreManager
from src.entity_extractor import EntityExtractor, ArchitecturalEntity
from src.issue_detector import IssueDetector, ArchitecturalIssue
from src.ingestion import DocumentData, PageData, ExtractedTable
from src.chunking import TextChunk


CATALOG_FILE = PROJECT_ROOT / "chroma_db" / "doc_catalog.json"


def classify_document(filename: str, text: str = "") -> str:
    """Classify document as Synthetic Test Data, AUTOSAR Specification, or General Document."""
    fname_lower = filename.lower()
    text_sample = text[:2000].lower() if text else ""
    if "synthetic" in fname_lower or "synthetic test data" in text_sample:
        return "Synthetic Test Data"
    if any(k in fname_lower or k in text_sample for k in ["autosar", "swc", "bsw", "mcal", "rte", "ecu"]):
        return "AUTOSAR Specification"
    return "General Document"


class DataManager:
    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.vector_store = vector_store or VectorStoreManager()
        self.entity_extractor = EntityExtractor()
        self.issue_detector = IssueDetector()
        CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._catalog: Dict[str, Dict[str, Any]] = self._load_catalog()
        self._sync_with_vector_store()

    def _load_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Load persistent catalog JSON if available."""
        if CATALOG_FILE.exists():
            try:
                with open(CATALOG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_catalog(self):
        """Save catalog dictionary to disk."""
        try:
            with open(CATALOG_FILE, "w", encoding="utf-8") as f:
                json.dump(self._catalog, f, indent=2)
        except Exception:
            pass

    def _sync_with_vector_store(self):
        """Ensure catalog matches indexed documents in ChromaDB."""
        indexed_docs = self.vector_store.get_indexed_documents()
        indexed_ids = {d["doc_id"] for d in indexed_docs}

        # Remove entries from catalog if deleted from ChromaDB
        to_delete = [did for did in self._catalog if did not in indexed_ids]
        for did in to_delete:
            del self._catalog[did]

        # Add stubs for documents present in ChromaDB but missing from catalog
        for doc in indexed_docs:
            did = doc["doc_id"]
            if did not in self._catalog:
                # Reconstruct stub from ChromaDB metadata
                chunks = self.vector_store.search("", top_k=50, doc_id_filter=did)
                full_text = " ".join([c["text"] for c in chunks])
                doc_type = classify_document(doc["filename"], full_text)
                
                # Extract entities and issues from retrieved chunks
                t_chunks = [
                    TextChunk(
                        chunk_id=c["chunk_id"],
                        doc_id=did,
                        filename=doc["filename"],
                        page_num=int(c["metadata"].get("page_num", 1)),
                        section=str(c["metadata"].get("section", "General")),
                        chunk_index=int(c["metadata"].get("chunk_index", 0)),
                        text=c["text"],
                        doc_hash=str(c["metadata"].get("doc_hash", "")),
                        extraction_method=str(c["metadata"].get("extraction_method", "pdf_text")),
                        char_count=len(c["text"]),
                    )
                    for c in chunks
                ]
                
                ents = self.entity_extractor.extract_from_chunks(t_chunks)
                
                # Mock minimal doc_data for issue detector
                max_page = max([tc.page_num for tc in t_chunks], default=1)
                p_data = [PageData(page_number=p, text="", headings=[], extraction_method="pdf_text") for p in range(1, max_page + 1)]
                for tc in t_chunks:
                    if tc.page_num - 1 < len(p_data):
                        p_data[tc.page_num - 1].text += "\n" + tc.text
                        p_data[tc.page_num - 1].headings.append(tc.section)
                d_data = DocumentData(
                    doc_id=did,
                    filename=doc["filename"],
                    file_hash=doc.get("doc_hash", ""),
                    total_pages=max_page,
                    pages=p_data,
                )
                
                issues = self.issue_detector.detect_issues(d_data, t_chunks, ents)

                self._catalog[did] = {
                    "doc_id": did,
                    "filename": doc["filename"],
                    "doc_hash": doc.get("doc_hash", ""),
                    "doc_type": doc_type,
                    "total_pages": max_page,
                    "chunk_count": len(chunks),
                    "pages": [
                        {
                            "page_number": p.page_number,
                            "text": p.text,
                            "headings": p.headings,
                            "extraction_method": p.extraction_method,
                        }
                        for p in p_data
                    ],
                    "entities": [e.model_dump() for e in ents],
                    "issues": [
                        {
                            "issue_id": i.issue_id,
                            "issue_type": i.issue_type,
                            "description": i.description,
                            "supporting_evidence": i.supporting_evidence,
                            "source_doc": i.source_doc,
                            "source_page": i.source_page,
                            "source_section": i.source_section,
                            "severity": i.severity,
                            "review_status": i.review_status,
                        }
                        for i in issues
                    ],
                }

        self._save_catalog()

    def register_document(
        self,
        doc_data: DocumentData,
        chunks: List[TextChunk],
        entities: List[ArchitecturalEntity],
        issues: List[ArchitecturalIssue],
    ):
        """Register newly ingested document into catalog."""
        doc_type = classify_document(doc_data.filename, doc_data.pages[0].text if doc_data.pages else "")
        self._catalog[doc_data.doc_id] = {
            "doc_id": doc_data.doc_id,
            "filename": doc_data.filename,
            "doc_hash": doc_data.file_hash,
            "doc_type": doc_type,
            "total_pages": doc_data.total_pages,
            "chunk_count": len(chunks),
            "pages": [
                {
                    "page_number": p.page_number,
                    "text": p.text,
                    "headings": p.headings,
                    "extraction_method": p.extraction_method,
                }
                for p in doc_data.pages
            ],
            "entities": [e.model_dump() for e in entities],
            "issues": [
                {
                    "issue_id": i.issue_id,
                    "issue_type": i.issue_type,
                    "description": i.description,
                    "supporting_evidence": i.supporting_evidence,
                    "source_doc": i.source_doc,
                    "source_page": i.source_page,
                    "source_section": i.source_section,
                    "severity": i.severity,
                    "review_status": i.review_status,
                }
                for i in issues
            ],
        }
        self._save_catalog()

    def remove_document(self, doc_id: str) -> bool:
        """Safely remove document from ChromaDB and catalog."""
        deleted = self.vector_store.delete_document(doc_id)
        if doc_id in self._catalog:
            del self._catalog[doc_id]
            self._save_catalog()
            deleted = True
        return deleted

    def get_documents_summary(self) -> List[Dict[str, Any]]:
        """Return list of all indexed documents with metadata and classifications."""
        self._sync_with_vector_store()
        docs = []
        for did, data in self._catalog.items():
            docs.append({
                "doc_id": did,
                "filename": data.get("filename", "Unknown"),
                "doc_type": data.get("doc_type", "General Document"),
                "total_pages": data.get("total_pages", 0),
                "chunk_count": data.get("chunk_count", 0),
                "doc_hash_prefix": data.get("doc_hash", "")[:12] + "...",
            })
        return docs

    def get_document_details(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Return full document data including page texts and chunk metadata."""
        return self._catalog.get(doc_id)

    def get_entities(self, doc_id_filter: Optional[str] = None) -> List[ArchitecturalEntity]:
        """Return extracted architectural entities matching optional document filter."""
        entities: List[ArchitecturalEntity] = []
        for did, data in self._catalog.items():
            if doc_id_filter and did != doc_id_filter:
                continue
            for ent_dict in data.get("entities", []):
                try:
                    entities.append(ArchitecturalEntity(**ent_dict))
                except Exception:
                    pass
        return entities

    def get_issues(self, doc_id_filter: Optional[str] = None) -> List[ArchitecturalIssue]:
        """Return deduplicated architectural issues matching optional document filter."""
        issues: List[ArchitecturalIssue] = []
        for did, data in self._catalog.items():
            if doc_id_filter and did != doc_id_filter:
                continue
            for iss_dict in data.get("issues", []):
                try:
                    issues.append(ArchitecturalIssue(**iss_dict))
                except Exception:
                    pass
        return issues

    def ensure_baseline_document(
        self,
        chunker: Optional[Any] = None,
        baseline_pdf_path: Optional[Path] = None,
        enable_ocr: bool = True,
        force: bool = False,
    ) -> bool:
        """Ensure baseline synthetic AUTOSAR HLD document is indexed."""
        return ensure_baseline_document(
            vstore=self.vector_store,
            data_mgr=self,
            chunker=chunker,
            baseline_path=baseline_pdf_path,
            enable_ocr=enable_ocr,
            force=force,
        )


def ensure_baseline_document(
    vstore: VectorStoreManager,
    data_mgr: DataManager,
    chunker: Optional[Any] = None,
    baseline_path: Optional[Path] = None,
    enable_ocr: bool = True,
    force: bool = False,
) -> bool:
    """Ensure baseline synthetic AUTOSAR HLD specification is available in the library.

    - Checks if Input_Data/synthetic_autosar_hld.pdf exists on disk.
    - If the library already contains documents and force=False, returns True immediately.
    - If not indexed in ChromaDB, runs through the standard ingestion pipeline,
      chunks the document, extracts entities/issues, and registers it cleanly.
    - Prevents duplicate chunks using SHA-256 hash checking.
    - Never fabricates or silently generates files if the baseline PDF is missing.

    Returns:
        bool: True if baseline document is present and indexed, False if missing on disk or failed.
    """
    target_path = baseline_path or (PROJECT_ROOT / "Input_Data" / "synthetic_autosar_hld.pdf")
    if not target_path.exists():
        return False

    # If library already has the document and force is False, return immediately
    existing_docs = data_mgr.get_documents_summary()
    if not force and any(d.get("filename") == target_path.name for d in existing_docs):
        return True

    try:
        from src.ingestion import PDFIngestionPipeline
        from src.chunking import ContextAwareChunker

        file_bytes = target_path.read_bytes()
        ingestor = PDFIngestionPipeline(enable_ocr=enable_ocr)
        doc_data = ingestor.process_pdf(file_bytes, target_path.name)

        active_chunker = chunker or ContextAwareChunker()
        chunks = active_chunker.chunk_document(doc_data)

        # Index chunks only if not already present in vector store
        if not vstore.is_document_indexed(doc_data.file_hash):
            vstore.index_chunks(chunks)

        # Extract entities and issues
        entities = data_mgr.entity_extractor.extract_from_chunks(chunks)
        issues = data_mgr.issue_detector.detect_issues(doc_data, chunks, entities)

        # Register document into shared catalog
        data_mgr.register_document(doc_data, chunks, entities, issues)
        return True
    except Exception:
        return False

