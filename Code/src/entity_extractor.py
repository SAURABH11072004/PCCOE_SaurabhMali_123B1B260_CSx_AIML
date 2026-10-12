"""Architectural Entity Extraction Module.
Extracts AUTOSAR SWCs, Interfaces, Ports, Signals, Functional Flows,
and Dependencies with source grounding and Pydantic validation.
"""

import json
import re
from typing import List, Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field

from src.chunking import TextChunk
from src.config import Settings, get_settings


class ArchitecturalEntity(BaseModel):
    name: str = Field(..., description="Unique name of the architectural entity")
    type: str = Field(..., description="Component, Interface, Port, Signal, Functional Flow, Dependency, Requirement")
    description: str = Field(default="", description="Engineering explanation or definition")
    doc_id: str = Field(default="", description="Source document identifier")
    source_filename: str = Field(default="", description="Source document filename")
    source_page: int = Field(default=1, description="1-indexed source document page number")
    source_section: str = Field(default="", description="Source section or heading")
    supporting_excerpt: str = Field(default="", description="Direct verbatim excerpt from the source text")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    review_status: str = Field(default="Candidate - Pending Review")


class EntityExtractor:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def _extract_heuristic_entities(self, chunk: TextChunk) -> List[ArchitecturalEntity]:
        """Extract AUTOSAR entities via domain-specific pattern matching."""
        entities: List[ArchitecturalEntity] = []
        text = chunk.text

        # 1. Software Components (e.g., ThrottleControl_SWC, BrakeManager_SWC)
        comp_matches = re.finditer(r"\b([A-Z][A-Za-z0-9_]+(?:_SWC|_Component|Manager_SWC|Control_SWC))\b", text)
        for m in comp_matches:
            name = m.group(1)
            # Find surrounding sentence
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 100)
            excerpt = text[start:end].strip()
            entities.append(
                ArchitecturalEntity(
                    name=name,
                    type="Component",
                    description=f"AUTOSAR Software Component identified in {chunk.section}",
                    doc_id=chunk.doc_id,
                    source_filename=chunk.filename,
                    source_page=chunk.page_num,
                    source_section=chunk.section,
                    supporting_excerpt=excerpt,
                    confidence=0.95,
                )
            )

        # 2. Interfaces (e.g., If_ThrottlePosition, If_BrakeTorque)
        iface_matches = re.finditer(r"\b(If_[A-Za-z0-9_]+)\b", text)
        for m in iface_matches:
            name = m.group(1)
            start = max(0, m.start() - 50)
            end = min(len(text), m.end() + 80)
            excerpt = text[start:end].strip()
            entities.append(
                ArchitecturalEntity(
                    name=name,
                    type="Interface",
                    description=f"AUTOSAR Software Interface identified in {chunk.section}",
                    doc_id=chunk.doc_id,
                    source_filename=chunk.filename,
                    source_page=chunk.page_num,
                    source_section=chunk.section,
                    supporting_excerpt=excerpt,
                    confidence=0.95,
                )
            )

        # 3. Ports (e.g., PpThrottlePos, RpPedalSensor, PpBrakeTorqueRequest)
        port_matches = re.finditer(r"\b((?:Pp|Rp)[A-Z][A-Za-z0-9_]+)\b", text)
        for m in port_matches:
            name = m.group(1)
            port_type = "Provided Port (P-Port)" if name.startswith("Pp") else "Required Port (R-Port)"
            start = max(0, m.start() - 50)
            end = min(len(text), m.end() + 80)
            excerpt = text[start:end].strip()
            entities.append(
                ArchitecturalEntity(
                    name=name,
                    type="Port",
                    description=f"AUTOSAR {port_type} identified in {chunk.section}",
                    doc_id=chunk.doc_id,
                    source_filename=chunk.filename,
                    source_page=chunk.page_num,
                    source_section=chunk.section,
                    supporting_excerpt=excerpt,
                    confidence=0.92,
                )
            )

        # 4. Signals (e.g., Sig_ThrottleAngle, Sig_RequestedTorque, Sig_StateOfCharge)
        sig_matches = re.finditer(r"\b(Sig_[A-Za-z0-9_]+)\b", text)
        for m in sig_matches:
            name = m.group(1)
            start = max(0, m.start() - 50)
            end = min(len(text), m.end() + 100)
            excerpt = text[start:end].strip()
            entities.append(
                ArchitecturalEntity(
                    name=name,
                    type="Signal",
                    description=f"AUTOSAR Signal / Data Element identified in {chunk.section}",
                    doc_id=chunk.doc_id,
                    source_filename=chunk.filename,
                    source_page=chunk.page_num,
                    source_section=chunk.section,
                    supporting_excerpt=excerpt,
                    confidence=0.93,
                )
            )

        # 5. Functional Flows (e.g., Flow 1: Driver Acceleration Request Flow)
        flow_matches = re.finditer(r"(?:Flow\s*\d+|Functional\s*Flow)[:\-]?\s*([A-Za-z0-9_\s]{5,60})", text, re.IGNORECASE)
        for m in flow_matches:
            name = m.group(0).strip()
            start = max(0, m.start() - 20)
            end = min(len(text), m.end() + 120)
            excerpt = text[start:end].strip()
            entities.append(
                ArchitecturalEntity(
                    name=name,
                    type="Functional Flow",
                    description="End-to-end AUTOSAR functional execution chain",
                    doc_id=chunk.doc_id,
                    source_filename=chunk.filename,
                    source_page=chunk.page_num,
                    source_section=chunk.section,
                    supporting_excerpt=excerpt,
                    confidence=0.88,
                )
            )

        # 6. Requirements / Safety Norms (e.g., REQ-001, ASIL-D, ISO 26262)
        req_matches = re.finditer(r"\b(REQ-[A-Z0-9_\-]+|ASIL-[A-D]|ISO\s*26262)\b", text)
        for m in req_matches:
            name = m.group(1)
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 80)
            excerpt = text[start:end].strip()
            entities.append(
                ArchitecturalEntity(
                    name=name,
                    type="Requirement",
                    description="Safety or requirement specification reference",
                    doc_id=chunk.doc_id,
                    source_filename=chunk.filename,
                    source_page=chunk.page_num,
                    source_section=chunk.section,
                    supporting_excerpt=excerpt,
                    confidence=0.90,
                )
            )

        return entities

    def extract_from_chunks(self, chunks: List[TextChunk]) -> List[ArchitecturalEntity]:
        """Extract and deduplicate entities from all chunks in a document."""
        seen = set()
        unique_entities: List[ArchitecturalEntity] = []

        for chunk in chunks:
            extracted = self._extract_heuristic_entities(chunk)
            for ent in extracted:
                key = (ent.name.lower(), ent.type.lower(), ent.doc_id, ent.source_page)
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(ent)

        return unique_entities

    def entities_to_dataframe(self, entities: List[ArchitecturalEntity]) -> pd.DataFrame:
        """Convert extracted entity list to pandas DataFrame."""
        if not entities:
            return pd.DataFrame(
                columns=[
                    "Name",
                    "Type",
                    "Description",
                    "Source Document",
                    "Page",
                    "Section",
                    "Confidence",
                    "Review Status",
                    "Supporting Excerpt",
                ]
            )
        data = [
            {
                "Name": e.name,
                "Type": e.type,
                "Description": e.description,
                "Source Document": e.source_filename,
                "Page": e.source_page,
                "Section": e.source_section,
                "Confidence": f"{e.confidence * 100:.0f}%",
                "Review Status": e.review_status,
                "Supporting Excerpt": e.supporting_excerpt,
            }
            for e in entities
        ]
        return pd.DataFrame(data)
