"""Potential Issues and Inconsistency Detection Module.
Detects conflicting signal definitions, undefined interface references,
undeclared components, and broken cross-references across AUTOSAR HLD sections.
Deduplicates repetitive findings and applies documented severity rules.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd

from src.ingestion import DocumentData
from src.chunking import TextChunk
from src.entity_extractor import ArchitecturalEntity


@dataclass
class ArchitecturalIssue:
    issue_id: str
    issue_type: str  # "Conflicting Definition", "Undefined Interface", "Undeclared Component", "Broken Cross-Reference"
    description: str
    supporting_evidence: str
    source_doc: str
    source_page: int
    source_section: str
    severity: str  # "High", "Medium", "Low"
    review_status: str = "Candidate Finding — Requires Human Engineering Review"


class IssueDetector:
    """Automated consistency verification engine for AUTOSAR HLD documentation."""

    def detect_issues(
        self,
        doc_data: DocumentData,
        chunks: List[TextChunk],
        entities: List[ArchitecturalEntity],
    ) -> List[ArchitecturalIssue]:
        """Analyze document text and extracted entities to identify deduplicated architectural inconsistencies."""
        issues: List[ArchitecturalIssue] = []
        issue_counter = 1

        declared_swcs = {e.name.lower(): e for e in entities if e.type == "Component"}
        declared_ifaces = {e.name.lower(): e for e in entities if e.type == "Interface"}
        declared_signals = {e.name.lower(): e for e in entities if e.type == "Signal"}

        # 1. Conflicting Signal Definitions / Update Periods
        # Group by signal name to ensure zero repetition
        signal_timing_mentions: Dict[str, List[Dict[str, Any]]] = {}
        for c in chunks:
            for sig in declared_signals:
                if sig in c.text.lower():
                    time_matches = re.findall(r"(\b\d+\s*ms\b)", c.text, re.IGNORECASE)
                    if time_matches:
                        for tm in time_matches:
                            tm_clean = tm.lower().replace(" ", "")
                            if sig not in signal_timing_mentions:
                                signal_timing_mentions[sig] = []
                            signal_timing_mentions[sig].append({
                                "timing": tm_clean,
                                "chunk": c,
                            })

        for sig, mentions in sorted(signal_timing_mentions.items()):
            distinct_timings = sorted(list({m["timing"] for m in mentions}))
            if len(distinct_timings) > 1:
                # Find distinct location mentions
                first_mention = next(m for m in mentions if m["timing"] == distinct_timings[0])
                second_mention = next(m for m in mentions if m["timing"] == distinct_timings[1])
                
                # Check if signal is in ASIL component -> High severity
                is_asil = any(
                    "asil-d" in m["chunk"].text.lower() or "asil-c" in m["chunk"].text.lower()
                    for m in mentions
                )
                severity = "High" if is_asil else "Medium"

                issues.append(
                    ArchitecturalIssue(
                        issue_id=f"ISSUE-{issue_counter:03d}",
                        issue_type="Conflicting Definition",
                        description=(
                            f"Signal '{sig}' is specified with conflicting cycle times "
                            f"({', '.join(distinct_timings)}) across different architectural sections."
                        ),
                        supporting_evidence=(
                            f"Section '{first_mention['chunk'].section}' (Page {first_mention['chunk'].page_num}) specifies '{first_mention['timing']}', "
                            f"while Section '{second_mention['chunk'].section}' (Page {second_mention['chunk'].page_num}) specifies '{second_mention['timing']}'."
                        ),
                        source_doc=doc_data.filename,
                        source_page=second_mention["chunk"].page_num,
                        source_section=second_mention["chunk"].section,
                        severity=severity,
                    )
                )
                issue_counter += 1

        # 2. Undefined Interface References (Deduplicated per interface name)
        seen_undefined_interfaces = set()
        for c in chunks:
            iface_refs = re.findall(r"\b(If_[A-Za-z0-9_]+)\b", c.text)
            for iref in iface_refs:
                iref_lower = iref.lower()
                if iref_lower not in declared_ifaces and iref_lower not in seen_undefined_interfaces:
                    if f"interface {iref_lower}" not in c.text.lower() and f"{iref_lower}:" not in c.text.lower():
                        seen_undefined_interfaces.add(iref_lower)
                        issues.append(
                            ArchitecturalIssue(
                                issue_id=f"ISSUE-{issue_counter:03d}",
                                issue_type="Undefined Interface",
                                description=(
                                    f"Port references interface '{iref}', but no formal interface definition "
                                    f"was found in the Interface Dictionary."
                                ),
                                supporting_evidence=f"Referenced in Section '{c.section}' (Page {c.page_num}): '{c.text[:130]}...'",
                                source_doc=doc_data.filename,
                                source_page=c.page_num,
                                source_section=c.section,
                                severity="Medium",
                            )
                        )
                        issue_counter += 1

        # 3. Undeclared Components in Functional Flows (Deduplicated per component name)
        seen_undeclared_swcs = set()
        for c in chunks:
            if "flow" in c.section.lower() or "functional flow" in c.text.lower():
                comp_refs = re.findall(r"\b([A-Z][A-Za-z0-9_]+(?:_SWC|_Component))\b", c.text)
                for cref in comp_refs:
                    cref_lower = cref.lower()
                    if cref_lower not in declared_swcs and cref_lower not in seen_undeclared_swcs:
                        seen_undeclared_swcs.add(cref_lower)
                        issues.append(
                            ArchitecturalIssue(
                                issue_id=f"ISSUE-{issue_counter:03d}",
                                issue_type="Undeclared Component",
                                description=(
                                    f"Component '{cref}' is actively referenced in functional flow '{c.section}', "
                                    f"but was not declared in the Software Components catalog."
                                ),
                                supporting_evidence=f"Mentioned in flow excerpt (Page {c.page_num}): '{c.text[:130]}...'",
                                source_doc=doc_data.filename,
                                source_page=c.page_num,
                                source_section=c.section,
                                severity="Medium",
                            )
                        )
                        issue_counter += 1

        # 4. Broken Cross-References (Deduplicated per referenced section)
        existing_sections = {h.lower() for p in doc_data.pages for h in p.headings}
        seen_broken_refs = set()
        for c in chunks:
            see_matches = re.finditer(r"(?:see|refer to)\s+(Section\s+[0-9]+(?:\.[0-9]+)?)", c.text, re.IGNORECASE)
            for sm in see_matches:
                ref_sec = sm.group(1).strip()
                ref_sec_lower = ref_sec.lower()
                if ref_sec_lower not in seen_broken_refs:
                    found = any(ref_sec_lower in es for es in existing_sections)
                    if not found:
                        seen_broken_refs.add(ref_sec_lower)
                        issues.append(
                            ArchitecturalIssue(
                                issue_id=f"ISSUE-{issue_counter:03d}",
                                issue_type="Broken Cross-Reference",
                                description=f"Document cites '{ref_sec}', but no corresponding section heading exists in the specification.",
                                supporting_evidence=f"Citing text in Section '{c.section}' (Page {c.page_num}): '{sm.group(0)}'",
                                source_doc=doc_data.filename,
                                source_page=c.page_num,
                                source_section=c.section,
                                severity="Low",
                            )
                        )
                        issue_counter += 1

        return issues

    def issues_to_dataframe(self, issues: List[ArchitecturalIssue]) -> pd.DataFrame:
        """Convert detected issues list to DataFrame."""
        if not issues:
            return pd.DataFrame(
                columns=[
                    "Issue ID",
                    "Type",
                    "Severity",
                    "Description",
                    "Page",
                    "Section",
                    "Supporting Evidence",
                    "Review Status",
                ]
            )
        return pd.DataFrame([
            {
                "Issue ID": i.issue_id,
                "Type": i.issue_type,
                "Severity": i.severity,
                "Description": i.description,
                "Page": i.source_page,
                "Section": i.source_section,
                "Supporting Evidence": i.supporting_evidence,
                "Review Status": i.review_status,
            }
            for i in issues
        ])
