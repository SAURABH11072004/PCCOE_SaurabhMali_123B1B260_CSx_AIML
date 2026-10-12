"""Context-Aware Chunking Pipeline.
Splits document text while preserving section boundaries, headings,
tables, and AUTOSAR architectural definitions.
"""

import re
from dataclasses import dataclass
from typing import List, Optional
from src.ingestion import DocumentData, PageData


@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    filename: str
    page_num: int
    section: str
    chunk_index: int
    text: str
    doc_hash: str
    extraction_method: str
    char_count: int

    def to_metadata(self) -> dict:
        """Convert to ChromaDB-compatible primitive metadata dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page_num": int(self.page_num),
            "section": str(self.section),
            "chunk_index": int(self.chunk_index),
            "doc_hash": str(self.doc_hash),
            "extraction_method": str(self.extraction_method),
            "char_count": int(self.char_count),
        }


class ContextAwareChunker:
    def __init__(self, target_chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc_data: DocumentData) -> List[TextChunk]:
        """Split document pages into context-aware chunks preserving metadata."""
        all_chunks: List[TextChunk] = []
        global_chunk_idx = 0

        for page in doc_data.pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            # Determine dominant section for page
            active_section = page.headings[0] if page.headings else f"Page {page.page_number}"
            
            # First, append markdown tables as discrete, atomic chunks if available
            for tbl in page.tables:
                if tbl.raw_markdown and len(tbl.raw_markdown) > 20:
                    tbl_chunk_id = f"{doc_data.doc_id}_p{page.page_number}_c{global_chunk_idx}"
                    all_chunks.append(
                        TextChunk(
                            chunk_id=tbl_chunk_id,
                            doc_id=doc_data.doc_id,
                            filename=doc_data.filename,
                            page_num=page.page_number,
                            section=f"{active_section} [Table]",
                            chunk_index=global_chunk_idx,
                            text=tbl.raw_markdown,
                            doc_hash=doc_data.file_hash,
                            extraction_method=page.extraction_method,
                            char_count=len(tbl.raw_markdown),
                        )
                    )
                    global_chunk_idx += 1

            # Split remaining page text by paragraph blocks
            paragraphs = re.split(r"\n\s*\n+", page_text)
            current_buffer = []
            current_length = 0
            current_section = active_section

            for para in paragraphs:
                para_clean = para.strip()
                if not para_clean:
                    continue

                # Check if paragraph is a new section header
                for h in page.headings:
                    if h.lower() in para_clean.lower() and len(para_clean) < 120:
                        current_section = h
                        break

                para_len = len(para_clean)

                # If adding paragraph exceeds target chunk size and buffer is not empty
                if current_length + para_len > self.chunk_size and current_buffer:
                    chunk_text = "\n\n".join(current_buffer)
                    chunk_id = f"{doc_data.doc_id}_p{page.page_number}_c{global_chunk_idx}"
                    all_chunks.append(
                        TextChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_data.doc_id,
                            filename=doc_data.filename,
                            page_num=page.page_number,
                            section=current_section,
                            chunk_index=global_chunk_idx,
                            text=chunk_text,
                            doc_hash=doc_data.file_hash,
                            extraction_method=page.extraction_method,
                            char_count=len(chunk_text),
                        )
                    )
                    global_chunk_idx += 1

                    # Retain overlap from end of previous buffer if feasible
                    if self.chunk_overlap > 0 and current_buffer:
                        last_p = current_buffer[-1]
                        if len(last_p) <= self.chunk_overlap:
                            current_buffer = [last_p, para_clean]
                            current_length = len(last_p) + para_len
                        else:
                            current_buffer = [para_clean]
                            current_length = para_len
                    else:
                        current_buffer = [para_clean]
                        current_length = para_len
                else:
                    current_buffer.append(para_clean)
                    current_length += para_len

            # Flush remaining buffer
            if current_buffer:
                chunk_text = "\n\n".join(current_buffer)
                chunk_id = f"{doc_data.doc_id}_p{page.page_number}_c{global_chunk_idx}"
                all_chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_data.doc_id,
                        filename=doc_data.filename,
                        page_num=page.page_number,
                        section=current_section,
                        chunk_index=global_chunk_idx,
                        text=chunk_text,
                        doc_hash=doc_data.file_hash,
                        extraction_method=page.extraction_method,
                        char_count=len(chunk_text),
                    )
                )
                global_chunk_idx += 1

        return all_chunks
