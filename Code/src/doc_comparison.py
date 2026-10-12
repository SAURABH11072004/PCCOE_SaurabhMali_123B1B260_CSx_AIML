"""Document Revision Comparison Module.
Performs semantic and structural diffing between two AUTOSAR HLD revisions,
identifying section additions/removals, entity changes, and potential architectural impacts.
"""

import difflib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd

from src.ingestion import DocumentData
from src.entity_extractor import EntityExtractor, ArchitecturalEntity


@dataclass
class SectionDiff:
    section_title: str
    change_type: str  # "ADDED", "REMOVED", "MODIFIED", "UNCHANGED"
    doc_a_page: Optional[int] = None
    doc_b_page: Optional[int] = None
    similarity_ratio: float = 1.0
    diff_summary: str = ""


@dataclass
class EntityDiff:
    entity_name: str
    entity_type: str
    change_type: str  # "ADDED", "REMOVED", "MODIFIED"
    doc_a_ref: Optional[str] = None
    doc_b_ref: Optional[str] = None
    impact_note: str = ""


@dataclass
class RevisionComparisonReport:
    doc_a_name: str
    doc_b_name: str
    added_sections_count: int
    removed_sections_count: int
    modified_sections_count: int
    section_diffs: List[SectionDiff]
    entity_diffs: List[EntityDiff]
    architectural_impacts: List[str]


class DocumentComparator:
    def __init__(self):
        self.entity_extractor = EntityExtractor()

    def compare_documents(
        self,
        doc_a: DocumentData,
        doc_b: DocumentData,
        entities_a: Optional[List[ArchitecturalEntity]] = None,
        entities_b: Optional[List[ArchitecturalEntity]] = None,
    ) -> RevisionComparisonReport:
        """Compare two documents and generate a detailed structural and architectural diff."""
        # 1. Map sections by page
        sections_a: Dict[str, Dict[str, Any]] = {}
        for p in doc_a.pages:
            h_list = p.headings if p.headings else [f"Page {p.page_number}"]
            for sec_name in h_list:
                if sec_name not in sections_a:
                    sections_a[sec_name] = {"page": p.page_number, "text": p.text}
                else:
                    sections_a[sec_name]["text"] += "\n" + p.text

        sections_b: Dict[str, Dict[str, Any]] = {}
        for p in doc_b.pages:
            h_list = p.headings if p.headings else [f"Page {p.page_number}"]
            for sec_name in h_list:
                if sec_name not in sections_b:
                    sections_b[sec_name] = {"page": p.page_number, "text": p.text}
                else:
                    sections_b[sec_name]["text"] += "\n" + p.text

        all_section_titles = sorted(list(set(list(sections_a.keys()) + list(sections_b.keys()))))
        section_diffs: List[SectionDiff] = []
        added_sec_count = 0
        removed_sec_count = 0
        mod_sec_count = 0

        for title in all_section_titles:
            in_a = title in sections_a
            in_b = title in sections_b

            if in_b and not in_a:
                added_sec_count += 1
                section_diffs.append(
                    SectionDiff(
                        section_title=title,
                        change_type="ADDED",
                        doc_a_page=None,
                        doc_b_page=sections_b[title]["page"],
                        similarity_ratio=0.0,
                        diff_summary=f"Section introduced in revision {doc_b.filename}",
                    )
                )
            elif in_a and not in_b:
                # Check if it was slightly renamed
                closest = difflib.get_close_matches(title, sections_b.keys(), n=1, cutoff=0.7)
                if closest:
                    c_title = closest[0]
                    ratio = difflib.SequenceMatcher(
                        None, sections_a[title]["text"], sections_b[c_title]["text"]
                    ).ratio()
                    mod_sec_count += 1
                    section_diffs.append(
                        SectionDiff(
                            section_title=f"{title} -> {c_title} (Renamed/Modified)",
                            change_type="MODIFIED",
                            doc_a_page=sections_a[title]["page"],
                            doc_b_page=sections_b[c_title]["page"],
                            similarity_ratio=round(ratio, 3),
                            diff_summary=f"Section modified and retitled ({ratio*100:.1f}% text similarity)",
                        )
                    )
                else:
                    removed_sec_count += 1
                    section_diffs.append(
                        SectionDiff(
                            section_title=title,
                            change_type="REMOVED",
                            doc_a_page=sections_a[title]["page"],
                            doc_b_page=None,
                            similarity_ratio=0.0,
                            diff_summary=f"Section removed from revision {doc_b.filename}",
                        )
                    )
            else:
                # Both present - compute textual similarity
                text_a = sections_a[title]["text"]
                text_b = sections_b[title]["text"]
                ratio = difflib.SequenceMatcher(None, text_a, text_b).ratio()

                if ratio < 0.98:
                    mod_sec_count += 1
                    section_diffs.append(
                        SectionDiff(
                            section_title=title,
                            change_type="MODIFIED",
                            doc_a_page=sections_a[title]["page"],
                            doc_b_page=sections_b[title]["page"],
                            similarity_ratio=round(ratio, 3),
                            diff_summary=f"Content updated ({ratio*100:.1f}% text match)",
                        )
                    )
                else:
                    section_diffs.append(
                        SectionDiff(
                            section_title=title,
                            change_type="UNCHANGED",
                            doc_a_page=sections_a[title]["page"],
                            doc_b_page=sections_b[title]["page"],
                            similarity_ratio=1.0,
                            diff_summary="Identical content across revisions",
                        )
                    )

        # 2. Entity Diffs
        ent_map_a = {e.name.lower(): e for e in (entities_a or [])}
        ent_map_b = {e.name.lower(): e for e in (entities_b or [])}
        all_ent_names = sorted(list(set(list(ent_map_a.keys()) + list(ent_map_b.keys()))))

        entity_diffs: List[EntityDiff] = []
        impacts: List[str] = []

        for ename in all_ent_names:
            ea = ent_map_a.get(ename)
            eb = ent_map_b.get(ename)

            if eb and not ea:
                entity_diffs.append(
                    EntityDiff(
                        entity_name=eb.name,
                        entity_type=eb.type,
                        change_type="ADDED",
                        doc_a_ref="Not present",
                        doc_b_ref=f"p.{eb.source_page} ({eb.source_section})",
                        impact_note=f"Newly introduced {eb.type}. Requires interface allocation and scheduling review.",
                    )
                )
                impacts.append(f"New {eb.type} '{eb.name}' added in {doc_b.filename} (Page {eb.source_page}).")
            elif ea and not eb:
                entity_diffs.append(
                    EntityDiff(
                        entity_name=ea.name,
                        entity_type=ea.type,
                        change_type="REMOVED",
                        doc_a_ref=f"p.{ea.source_page} ({ea.source_section})",
                        doc_b_ref="Not present",
                        impact_note=f"Removed {ea.type}. Potential breaking change for dependent components.",
                    )
                )
                impacts.append(f"Deprecated/Removed {ea.type} '{ea.name}' from {doc_a.filename}. Dependent components require verification.")
            elif ea and eb:
                # Compare descriptions and excerpts
                if ea.supporting_excerpt.strip() != eb.supporting_excerpt.strip():
                    entity_diffs.append(
                        EntityDiff(
                            entity_name=eb.name,
                            entity_type=eb.type,
                            change_type="MODIFIED",
                            doc_a_ref=f"p.{ea.source_page}",
                            doc_b_ref=f"p.{eb.source_page}",
                            impact_note="Definition or attributes updated across revisions.",
                        )
                    )
                    impacts.append(f"Updated definition for {eb.type} '{eb.name}'.")

        return RevisionComparisonReport(
            doc_a_name=doc_a.filename,
            doc_b_name=doc_b.filename,
            added_sections_count=added_sec_count,
            removed_sections_count=removed_sec_count,
            modified_sections_count=mod_sec_count,
            section_diffs=section_diffs,
            entity_diffs=entity_diffs,
            architectural_impacts=impacts,
        )

    def diff_to_dataframe(self, section_diffs: List[SectionDiff]) -> pd.DataFrame:
        """Convert section diff list to DataFrame."""
        return pd.DataFrame([
            {
                "Section": s.section_title,
                "Change Type": s.change_type,
                "Doc A Page": s.doc_a_page if s.doc_a_page else "-",
                "Doc B Page": s.doc_b_page if s.doc_b_page else "-",
                "Similarity": f"{s.similarity_ratio * 100:.1f}%",
                "Summary": s.diff_summary,
            }
            for s in section_diffs
        ])
