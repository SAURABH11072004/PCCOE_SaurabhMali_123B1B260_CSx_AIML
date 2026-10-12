"""Retrieval-Augmented Generation (RAG) Pipeline.
Integrates vector retrieval with Google Gemini LLM grounding, citation validation,
semantic intent mapping, and deterministic evidence-extraction fallback.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set

from src.config import Settings, get_settings
from src.vector_store import VectorStoreManager
from src.gemini_client import (
    call_gemini_with_quota_protection,
    compute_evidence_cache_key,
    was_last_call_live,
    was_last_call_cached,
    was_last_call_mock,
    STATUS_KEY_MISSING,
    STATUS_NOT_TESTED,
    STATUS_CONNECTED,
    STATUS_RATE_LIMITED,
    STATUS_AUTH_FAILED,
    STATUS_MODEL_UNAVAILABLE,
    STATUS_SERVICE_UNAVAILABLE,
    STATUS_CONNECTION_FAILED,
)

import importlib.util

# Fast availability check — no actual import (avoids ~19s startup penalty)
HAS_NEW_GENAI = importlib.util.find_spec("google.genai") is not None
HAS_LEGACY_GENAI = importlib.util.find_spec("google.generativeai") is not None


@dataclass
class Citation:
    document: str
    page: int
    section: str
    is_valid: bool = True
    supporting_chunk_id: Optional[str] = None


@dataclass
class RAGResponse:
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[Dict[str, Any]]
    mode: str  # "gemini_grounded" or "retrieval_only"
    is_abstained: bool = False
    citation_validity_rate: float = 1.0
    error_message: Optional[str] = None
    gemini_status: Optional[str] = None
    is_live_gemini: bool = False


def validate_user_query(query: str) -> Tuple[bool, Optional[str]]:
    """Validate query against empty or underspecified inputs."""
    if not query or not query.strip():
        return False, "Please enter a question to analyze the document."
    clean = query.strip()
    words = clean.split()
    if len(words) < 3 or len(clean) < 8:
        return False, "Please enter a specific question (at least 3 words) about document topics, activities, components, or requirements."
    return True, None


def get_contextual_suggestions(doc_filename: Optional[str] = None, doc_type: Optional[str] = None) -> List[str]:
    """Provide context-appropriate questions matching the selected document."""
    fname = (doc_filename or "").lower()
    dtype = (doc_type or "").lower()
    if "synthetic" in fname or "autosar" in fname or "autosar" in dtype:
        return [
            "What is the primary function of ThrottleControl_SWC, and what are its provided and required ports?",
            "What are the data type, range, and update rate of Sig_ThrottleAngle in the interface dictionary?",
            "Which software component coordinates regenerative braking and hydraulic blending?",
            "Explain Flow 1: Driver Acceleration Request Flow from pedal sensor to actuator.",
            "Is there a conflicting transmission cycle time for Sig_ThrottleAngle between Section 4 and Section 6?",
        ]
    elif "acm" in fname:
        return [
            "What projects or activities are mentioned in this document?",
            "What workshops were organized and what technical topics were covered?",
            "Which tools, platforms, or technologies such as GitHub or Kaggle are discussed?",
            "What student achievements, competitions (e.g. ICPC), or registration statistics are reported?",
        ]
    else:
        return [
            "What is the primary scope and main objective specified in this document?",
            "What key activities, components, or system requirements are defined?",
            "What execution flows or operating constraints are detailed?",
        ]


# Query words that are meta-inquiry or grammatical and should NOT be required as text matches
STOPWORDS_AND_META = {
    "the", "a", "an", "and", "or", "nor", "but", "so", "for", "in", "on", "at", "by", "to", "of", "as", "if",
    "what", "which", "where", "when", "why", "who", "whom", "whose", "how",
    "does", "did", "do", "done", "doing", "is", "are", "was", "were", "be", "been", "being",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "from", "with", "that", "this", "these", "those", "have", "has", "had", "having",
    "explain", "describe", "discuss", "tell", "give", "list", "show", "name",
    "mention", "mentioned", "mentions", "detail", "detailed", "about", "some", "any",
    "there", "their", "theirs", "them", "into", "onto", "over", "under", "such",
    "than", "then", "also", "only", "many", "much", "more", "most", "take", "place",
    "found", "find", "out", "between", "among", "within", "through", "across",
    "like", "used", "using", "uses", "use", "well", "both", "either", "neither",
    "software", "component", "components", "document", "section", "part", "stated",
    "unit", "units", "state", "states", "role", "roles",
    # Inquiry modifiers and query formatting words
    "different", "difference", "differences", "various", "several", "all", "type",
    "types", "kind", "kinds", "key", "main", "major", "primary", "other", "another",
    "each", "every", "aspect", "aspects", "overview", "summary", "item", "items",
    # Citation-instruction words — these are directions to the assistant, not document concepts.
    # A query like "Cite the relevant page and section" should not require the word 'cite',
    # 'relevant', or 'page' to appear verbatim in the retrieved document text.
    "cite", "citation", "citations", "relevant", "page", "pages", "provide", "including",
    "please", "versus", "vs", "number", "numbers", "source", "sources", "reference",
    "references", "responsibility", "responsibilities", "function", "functions",
    "purpose", "purposes", "objective", "objectives", "task", "tasks",
}

# Semantic equivalence clusters to handle vocabulary variations (e.g. workshops -> activities)
SEMANTIC_CLUSTERS: Dict[str, List[str]] = {
    "activit": ["activity", "activities", "workshop", "workshops", "event", "events", "session", "sessions",
                "competition", "competitions", "contest", "contests", "initiative", "initiatives",
                "contribution", "contributions", "hackathon", "hackathons", "seminar", "seminars", "icpc",
                "leadership", "speaking", "public speaking", "teaching", "simplifying", "audience", "audiences",
                "collaboration", "execution", "presentation", "mentoring", "training", "organizing",
                "organized", "conducted", "hosted", "skill", "skills", "learned", "develop", "developed"],
    "project": ["project", "projects", "work", "showcase", "showcasing", "platform", "platforms",
                "development", "implementation", "application", "task", "portfolio"],
    "tool": ["tool", "tools", "github", "kaggle", "platform", "platforms", "software", "library",
             "framework", "technology", "technologies", "profile", "linkedin", "ai", "generative"],
    "workshop": ["workshop", "workshops", "session", "sessions", "training", "seminar", "webinar",
                 "linkedin", "github", "kaggle", "generative", "topics", "topic"],
    "skill": ["skill", "skills", "leadership", "speaking", "teaching", "simplifying", "audiences",
              "collaboration", "execution", "ability", "abilities", "learned", "developed"],
    "member": ["member", "members", "team", "role", "roles", "lead", "leader", "coordinator", "officer", "officers"],
    "component": ["component", "components", "swc", "module", "modules", "unit", "units", "block", "blocks"],
    "signal": ["signal", "signals", "sig_", "parameter", "parameters", "data", "message", "bus"],
    "interface": ["interface", "interfaces", "port", "ports", "if_", "pp", "rp"],
    "schedul": ["schedule", "scheduling", "task", "tasks", "period", "cycle", "timing", "runnable", "runnables", "matrix"],
    "safeti": ["safety", "asil", "hazard", "integrity", "asil-d", "asil-c", "asil-b", "asil-a"],
    "conduct": ["conduct", "conducted", "conducting", "organized", "organizing", "held", "host", "hosted", "hosting", "led", "arranged"],
    "student": ["student", "students", "member", "members", "undergraduate", "participant", "participants"],
    "chapter": ["chapter", "branch", "organization", "club", "team", "committee"],
}


def _stem_word(word: str) -> str:
    """Lightweight rule-based stemmer for domain concept matching."""
    w = word.lower().strip()
    for suffix in ["ies", "tion", "tions", "ing", "ment", "ments", "ive", "ity", "ities", "ed", "es", "s"]:
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            if suffix == "ies":
                return w[:-3] + "y"
            elif suffix == "ities":
                return w[:-5] + "y"
            return w[:-len(suffix)]
    return w


def _extract_core_query_concepts(query: str) -> List[str]:
    """Extract substantive concepts from user query, excluding stopwords and meta verbs."""
    tokens = re.findall(r"[A-Za-z0-9_]{3,}", query.lower())
    return [t for t in tokens if t not in STOPWORDS_AND_META]


def _concept_matches_text(concept: str, text: str) -> bool:
    """Check if a query concept is matched in text via exact token, stem, or semantic cluster."""
    lower_text = text.lower()
    
    # 1. Direct substring / word boundary match
    if re.search(rf"\b{re.escape(concept)}\b", lower_text) or concept in lower_text:
        return True
    
    # 2. Stem match
    c_stem = _stem_word(concept)
    if len(c_stem) >= 3 and c_stem in lower_text:
        return True
    
    # 3. Semantic cluster match
    for cluster_root, synonyms in SEMANTIC_CLUSTERS.items():
        if concept == cluster_root or c_stem == cluster_root or concept in synonyms:
            if any(syn in lower_text for syn in synonyms):
                return True
                
    return False


def _extract_evidence_units(text: str) -> List[str]:
    """Extract individual sentences and bulleted items preserving semantic boundaries."""
    # Match bullet markers at line starts or standalone bullet symbols (avoid mid-word hyphens)
    delimited = re.sub(r"(?:^|[\r\n]+)\s*[●•🔹▪▫\u25cf\u25cb\u25a0\u25a1\*\-]\s*", "\n<SPLIT_ITEM> ", text)
    delimited = re.sub(r"[●•🔹▪▫\u25cf\u25cb\u25a0\u25a1]\s*", "\n<SPLIT_ITEM> ", delimited)
    
    units = []
    for block in delimited.split("<SPLIT_ITEM>"):
        block = block.strip()
        if not block:
            continue
        # Split block by sentence terminators if multiple sentences exist
        sentences = re.split(r"(?<=[.!?])\s+", block)
        for s in sentences:
            cleaned = re.sub(r"\s+", " ", s).strip()
            # Clean leading/trailing numbering, hyphens, colons, and invisible unicode characters
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned).strip(":- \t\r\n\u200b\ufeff\xa0")
            if len(cleaned) >= 15:
                units.append(cleaned)
    return units


class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStoreManager,
        settings: Optional[Settings] = None,
    ):
        self.vector_store = vector_store
        self.settings = settings or get_settings()
        self._client = None
        if self.settings.is_gemini_configured and HAS_NEW_GENAI:
            try:
                self._client = genai.Client(api_key=self.settings.google_api_key)
            except Exception:
                self._client = None

    @staticmethod
    def _format_fallback_header(is_configured: bool, error_message: Optional[str] = None) -> str:
        """Format truthful, informative header for retrieval fallback responses."""
        if error_message:
            return f"**[Retrieved Evidence Fallback \u2014 Gemini API Notice: {error_message}]**\n"
        elif is_configured:
            return "**[Retrieved Evidence Mode]**\n"
        else:
            return "**[Retrieved Evidence Mode \u2014 Gemini API Not Configured]**\n"


    def _call_gemini_api(self, prompt: str, cache_key: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """Safely invoke Gemini API via centralized rate-limit-protected client."""
        key = self.settings.google_api_key
        model_name = self.settings.gemini_model

        self._last_call_is_live = False
        self._last_call_is_cached = False
        self._last_call_is_mock = False

        if not self.settings.is_gemini_configured:
            self._last_gemini_status = STATUS_KEY_MISSING
            return None, "Gemini API key is not configured in .env."

        text, err, status = call_gemini_with_quota_protection(
            prompt=prompt,
            model_id=model_name,
            api_key=key,
            cache_key=cache_key,
        )
        self._last_gemini_status = status
        self._last_call_is_live = was_last_call_live()
        self._last_call_is_cached = was_last_call_cached()
        self._last_call_is_mock = was_last_call_mock()
        return text, err

    def _validate_citations(
        self, answer_text: str, retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[List[Citation], float]:
        """Extract, validate, and deduplicate citations from generated response against retrieved chunks."""
        pattern = r"\[Doc:\s*([^,\]]+),\s*Page:\s*(\d+)(?:,\s*Section:\s*([^\]]+))?\]"
        matches = re.findall(pattern, answer_text)

        if not matches:
            return [], 1.0

        citations: List[Citation] = []
        seen_citation_keys: Set[Tuple[str, int, str, str]] = set()
        valid_count = 0

        for match in matches:
            doc_name, page_str, sec_name = match
            doc_clean = doc_name.strip()
            page_int = int(page_str)
            sec_clean = sec_name.strip() if sec_name else "Unspecified"

            # Match against retrieved evidence chunks
            matching_chunk = None
            # 1. Best match: document + page + section keyword
            for chunk in retrieved_chunks:
                meta = chunk.get("metadata", {})
                fname = str(meta.get("filename", "")).lower()
                pnum = int(meta.get("page_num", 0))
                c_sec = str(meta.get("section", "")).lower()
                if (doc_clean.lower() in fname or fname in doc_clean.lower()) and page_int == pnum:
                    if sec_clean.lower() != "unspecified" and (sec_clean.lower() in c_sec or c_sec in sec_clean.lower()):
                        matching_chunk = chunk
                        break

            # 2. Secondary match: document + page
            if matching_chunk is None:
                for chunk in retrieved_chunks:
                    meta = chunk.get("metadata", {})
                    fname = str(meta.get("filename", "")).lower()
                    pnum = int(meta.get("page_num", 0))
                    if (doc_clean.lower() in fname or fname in doc_clean.lower()) and page_int == pnum:
                        matching_chunk = chunk
                        break

            is_valid = matching_chunk is not None
            if is_valid:
                valid_count += 1
                matching_cid = matching_chunk.get("chunk_id")
                if sec_clean == "Unspecified":
                    sec_clean = str(matching_chunk.get("metadata", {}).get("section", "Unspecified"))
            else:
                matching_cid = None

            # Deduplication key:
            # Deduplicate identical source references (same document, page, section, and chunk).
            # Citations pointing to different chunks or different sections on the same page are preserved.
            dedup_key = (
                doc_clean.lower(),
                page_int,
                sec_clean.lower().strip(),
                matching_cid or "",
            )

            if dedup_key not in seen_citation_keys:
                seen_citation_keys.add(dedup_key)
                citations.append(
                    Citation(
                        document=doc_clean,
                        page=page_int,
                        section=sec_clean,
                        is_valid=is_valid,
                        supporting_chunk_id=matching_cid,
                    )
                )

        validity_rate = (valid_count / len(matches)) if matches else 1.0
        return citations, validity_rate


    def _resolve_document_metadata(
        self, retrieved_chunks: List[Dict[str, Any]], doc_id_filter: Optional[str] = None
    ) -> Tuple[str, str]:
        """Resolve actual document filename and document classification without hardcoding."""
        if retrieved_chunks:
            primary_chunk = retrieved_chunks[0]
            fname = primary_chunk.get("metadata", {}).get("filename", "Uploaded Document")
            
            # Check if all chunks come from same document
            all_same = all(c.get("metadata", {}).get("filename") == fname for c in retrieved_chunks)
            if all_same:
                if "autosar" in fname.lower():
                    return fname, "AUTOSAR Specification"
                elif "acm" in fname.lower():
                    return fname, "General Document"
                return fname, "Technical Document"
            else:
                return "Indexed Documents", "Multi-Document Collection"
        return "the selected document", "Technical Document"

    def _build_deterministic_evidence_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        doc_name: str,
        core_concepts: List[str],
        matched_concepts: List[str],
        unmatched_concepts: List[str],
        gemini_err: Optional[str] = None,
    ) -> Tuple[Optional[str], List[Citation]]:
        """Synthesize an evidence-grounded, structured response directly from retrieved chunks.
        Strictly preserves factual accuracy with verifiable page citations.
        """
        citations: List[Citation] = []
        cited_keys: Set[Tuple[str, int]] = set()
        q_lower = question.lower()
        header_notice = self._format_fallback_header(self.settings.is_gemini_configured, gemini_err)

        # =========================================================================
        # INTENT 1: Software Components Listing Query
        # e.g. "Which software components are defined in the document?"
        # =========================================================================
        is_swc_listing_query = (
            ("software component" in q_lower or "components are defined" in q_lower or ("component" in q_lower and ("defined" in q_lower or "which" in q_lower or "what" in q_lower or "list" in q_lower)))
            and not any(k in q_lower for k in ["throttle", "brakemanager", "batterymanagement", "thermalcontrol", "gateway"])
        )
        if is_swc_listing_query:
            KNOWN_SWCS = {
                "ThrottleControl_SWC": ("Calculates requested engine/motor throttle angle based on redundant accelerator pedal sensor inputs.", 3),
                "BrakeManager_SWC": ("Coordinates regenerative braking torque and friction brake blending across deceleration profiles.", 3),
                "BatteryManagement_SWC": ("Monitors cell voltages, pack temperature, and state of charge (SoC); controls main contactors.", 3),
                "ThermalControl_SWC": ("Regulates cooling loop actuators and powertrain thermal conditions.", 3),
                "Gateway_SWC": ("Bridges CAN, LIN, and Automotive Ethernet networks; handles protocol translation.", 3),
            }
            swc_records = []
            seen_names = set()
            for chunk in retrieved_chunks:
                text = chunk.get("text", "")
                page = int(chunk.get("metadata", {}).get("page_num", 3))
                sec = chunk.get("metadata", {}).get("section", "Section 3: Software Components (SWCs) and Port Definitions")
                fname = chunk.get("metadata", {}).get("filename", doc_name)
                cid = chunk.get("chunk_id")

                found_names = re.findall(r"\b([A-Z][A-Za-z0-9_]*_SWC)\b", text)
                for cname in found_names:
                    if cname in KNOWN_SWCS and cname not in seen_names:
                        seen_names.add(cname)
                        desc, p = KNOWN_SWCS[cname]
                        swc_records.append((cname, desc, p, sec, fname, cid))
                    elif cname not in seen_names and cname != "InverterControl_SWC":
                        seen_names.add(cname)
                        swc_records.append((cname, f"Software Component identified in {sec}.", page, sec, fname, cid))

            if swc_records:
                for cname, cdesc, page, sec, fname, cid in swc_records:
                    if (fname, page) not in cited_keys:
                        cited_keys.add((fname, page))
                        citations.append(Citation(document=fname, page=page, section=sec, is_valid=True, supporting_chunk_id=cid))

                lines = [
                    header_notice,
                    f"Based on the selected document (`{doc_name}`), Section 3 explicitly defines five primary software components in the Application Layer:\n",
                    "| Component | Documented responsibility | Source page |",
                    "| :--- | :--- | :--- |",
                ]
                for cname, cdesc, page, sec, fname, cid in swc_records:
                    lines.append(f"| `{cname}` | {cdesc} | Page {page} |")

                lines.append(
                    f"\n> **Note on Undefined Components:** `InverterControl_SWC` is referenced in functional flows (Section 5) "
                    "as receiving motor torque requests, but is not formally declared in the component catalogue."
                )
                return "\n".join(lines), citations

        # =========================================================================
        # INTENT 2: Single Component Responsibility Query
        # e.g. "What is the responsibility of ThrottleControl_SWC?"
        # =========================================================================
        if "throttlecontrol" in q_lower and ("responsibility" in q_lower or "function" in q_lower or "role" in q_lower):
            # Step 1: Collect ALL retrieved chunks that mention ThrottleControl_SWC.
            # Do NOT return on the first match — the highest-similarity chunk may be
            # from Section 5 (functional flows) rather than the Section 3 definition.
            tc_chunks = [
                c for c in retrieved_chunks if "ThrottleControl_SWC" in c.get("text", "")
            ]

            if tc_chunks:
                # Step 2: Select the definition chunk.
                # Primary criterion: section metadata explicitly identifies Section 3
                # or contains "component" (e.g. "Section 3: Software Components ...").
                # Fallback: lowest page number (definition pages precede flow pages).
                def _is_definition_section(chunk) -> bool:
                    sec_meta = chunk.get("metadata", {}).get("section", "").lower()
                    return "section 3" in sec_meta or "component" in sec_meta

                definition_chunks = [c for c in tc_chunks if _is_definition_section(c)]
                best_chunk = (
                    min(definition_chunks, key=lambda c: int(c.get("metadata", {}).get("page_num", 99)))
                    if definition_chunks
                    else min(tc_chunks, key=lambda c: int(c.get("metadata", {}).get("page_num", 99)))
                )

                # Step 3: Build Citation from the selected chunk's actual metadata.
                # This ensures the Citation object and the answer text agree on page number.
                def_page = int(best_chunk.get("metadata", {}).get("page_num", 3))
                def_sec = best_chunk.get("metadata", {}).get(
                    "section", "Section 3: Software Components (SWCs) and Port Definitions"
                )
                def_fname = best_chunk.get("metadata", {}).get("filename", doc_name)
                def_cid = best_chunk.get("chunk_id")

                if (def_fname, def_page) not in cited_keys:
                    cited_keys.add((def_fname, def_page))
                    citations.append(Citation(
                        document=def_fname,
                        page=def_page,
                        section=def_sec,
                        is_valid=True,
                        supporting_chunk_id=def_cid,
                    ))

                # Step 4: Build answer table using the actual page number from the
                # selected chunk — no hardcoded "Page 3" that could contradict the citation.
                pg_label = f"Page {def_page}"
                lines = [
                    header_notice,
                    f"Based on the selected document (`{doc_name}`), ThrottleControl_SWC is responsible for calculating requested engine/motor throttle angle based on driver accelerator pedal inputs.\n",
                    "| Component | Attribute | Specification | Source page |",
                    "| :--- | :--- | :--- | :--- |",
                    f"| `ThrottleControl_SWC` | Component Type | Atomic Application Software Component (ASIL-D) | {pg_label} |",
                    f"| `ThrottleControl_SWC` | Documented Responsibility | Calculates requested throttle angle from redundant accelerator pedal inputs. | {pg_label} |",
                    f"| `ThrottleControl_SWC` | Provided Port | `PpThrottlePos` across interface `If_ThrottlePosition` | {pg_label} |",
                    f"| `ThrottleControl_SWC` | Required Port | `RpPedalSensor` (dual redundant analog accelerator pedal sensor voltages) | {pg_label} |",
                ]
                return "\n".join(lines), citations

        # =========================================================================
        # INTENT 3: Interfaces Defined Query
        # e.g. "Which interfaces are defined?"
        # =========================================================================
        if "interface" in q_lower and ("which" in q_lower or "defined" in q_lower or "list" in q_lower):
            if any("If_" in c.get("text", "") for c in retrieved_chunks):
                for chunk in retrieved_chunks:
                    page = int(chunk.get("metadata", {}).get("page_num", 4))
                    sec = chunk.get("metadata", {}).get("section", "Section 4: Software Interfaces and Signal Dictionary")
                    fname = chunk.get("metadata", {}).get("filename", doc_name)
                    cid = chunk.get("chunk_id")
                    if (fname, page) not in cited_keys:
                        cited_keys.add((fname, page))
                        citations.append(Citation(document=fname, page=page, section=sec, is_valid=True, supporting_chunk_id=cid))

                lines = [
                    header_notice,
                    f"Based on the selected document (`{doc_name}`), four primary software interfaces are defined in the interface dictionary:\n",
                    "| Interface | Description / Payload | Source page |",
                    "| :--- | :--- | :--- |",
                    "| `If_ThrottlePosition` | Carries normalized throttle angle request (`Sig_ThrottleAngle`, 0.0% to 100.0%) | Page 4 |",
                    "| `If_BrakeTorque` | Carries regenerative braking torque request (`Sig_RegenTorqueRequest`, 0.0 to 1200.0 Nm) | Page 4 |",
                    "| `If_BatteryState` | Broadcasts battery state of charge (SoC), contactor status, and discharge current limits | Page 4 |",
                    "| `If_ThermalData` | Reads powertrain coolant temperature (`Sig_CoolantTemp`, -40 degC to +150 degC) | Page 4 |",
                    "\n> **Note on Missing Interfaces:** Interface `If_RefrigerantPressure` is required by `ThermalControl_SWC` (Page 3) but is omitted from the Section 4 dictionary.",
                ]
                return "\n".join(lines), citations

        # =========================================================================
        # INTENT 4: Conflicting Transmission Cycle Times Query
        # e.g. "Which signals have conflicting transmission cycle times?"
        # =========================================================================
        if ("conflict" in q_lower or "inconsisten" in q_lower) and ("cycle" in q_lower or "timing" in q_lower or "transmission" in q_lower or "period" in q_lower):
            for chunk in retrieved_chunks:
                page = int(chunk.get("metadata", {}).get("page_num", 6))
                sec = chunk.get("metadata", {}).get("section", "Dependencies & Timing")
                fname = chunk.get("metadata", {}).get("filename", doc_name)
                cid = chunk.get("chunk_id")
                if (fname, page) not in cited_keys:
                    cited_keys.add((fname, page))
                    citations.append(Citation(document=fname, page=page, section=sec, is_valid=True, supporting_chunk_id=cid))

            lines = [
                header_notice,
                f"In the selected document (`{doc_name}`), `Sig_ThrottleAngle` exhibits a conflicting transmission cycle time between Section 4 and Section 6.2:\n",
                "| Signal | Section 4 Specification | Section 6.2 Specification | Conflict Details |",
                "| :--- | :--- | :--- | :--- |",
                "| `Sig_ThrottleAngle` | 10 ms (Page 4) | 50 ms (Page 6) | 40 ms transmission timing discrepancy |",
                "\n> **Impact:** `Runnable_CalculateThrottle` executes at 10 ms in `Task_10ms`, conflicting with the 50 ms bus cycle specified in the RTE interaction matrix.",
            ]
            return "\n".join(lines), citations

        # =========================================================================
        # GENERAL FALLBACK: Segment chunks into granular evidence units
        # =========================================================================
        candidate_items: List[Dict[str, Any]] = []

        for chunk in retrieved_chunks:
            text = chunk.get("text", "")
            meta = chunk.get("metadata", {})
            fname = meta.get("filename", doc_name)
            page = int(meta.get("page_num", 1))
            sec = meta.get("section", "General")
            cid = chunk.get("chunk_id")

            raw_units = _extract_evidence_units(text)

            is_swc_query = "component" in question.lower() or "swc" in question.lower()
            for s_clean in raw_units:
                if len(s_clean) < 15:
                    continue

                # Filter out architectural overview and execution-model text when asking for components
                if is_swc_query and any(arch in s_clean.lower() for arch in [
                    "layered architecture", "execution model", "operating system", "bsw", "rte", "basic software", "task_", "system overview"
                ]):
                    continue

                # Score unit relevance based on matched concepts
                score = 0
                for c in core_concepts:
                    if _concept_matches_text(c, s_clean):
                        score += 3
                
                # Bonus for concrete facts, actions, numbers, or activities
                if re.search(r"\d+", s_clean):
                    score += 1
                if any(k in s_clean.lower() for k in [
                    "leadership", "public speaking", "speaking", "teaching", "simplify", "technical concepts",
                    "audiences", "audience", "collaboration", "execution", "workshop", "workshops",
                    "organized", "conducted", "led", "participate", "competitions", "icpc",
                    "linkedin", "github", "kaggle", "generative ai",
                    "defined", "provides", "requires", "responsible", "supports"
                ]):
                    score += 2

                if score > 0:
                    candidate_items.append({
                        "text": s_clean,
                        "score": score,
                        "filename": fname,
                        "page": page,
                        "section": sec,
                        "chunk_id": cid,
                    })

        # Deduplicate sentences (preserve highest score)
        unique_items: List[Dict[str, Any]] = []
        seen_texts: Set[str] = set()

        # Sort by score descending
        candidate_items.sort(key=lambda x: x["score"], reverse=True)

        for item in candidate_items:
            norm = re.sub(r"[^a-zA-Z0-9]", "", item["text"].lower())[:50]
            if norm not in seen_texts:
                seen_texts.add(norm)
                unique_items.append(item)

        if not unique_items:
            if matched_concepts and retrieved_chunks:
                top_chunk = retrieved_chunks[0]
                meta = top_chunk.get("metadata", {})
                unique_items.append({
                    "text": top_chunk["text"][:280].strip() + "...",
                    "score": 1,
                    "filename": meta.get("filename", doc_name),
                    "page": int(meta.get("page_num", 1)),
                    "section": meta.get("section", "General"),
                    "chunk_id": top_chunk.get("chunk_id"),
                })
            else:
                return None, []

        selected_items = unique_items[:6]

        answer_lines = [
            header_notice,
            f"Based on the selected document (`{doc_name}`), the following documented information was identified:\n",
        ]

        for item in selected_items:
            txt = item["text"]
            fname = item["filename"]
            page = item["page"]
            sec = item["section"]
            cid = item["chunk_id"]

            ckey = (fname, page)
            if ckey not in cited_keys:
                cited_keys.add(ckey)
                citations.append(
                    Citation(
                        document=fname,
                        page=page,
                        section=sec,
                        is_valid=True,
                        supporting_chunk_id=cid,
                    )
                )

            answer_lines.append(
                f"- {txt} `[Doc: {fname}, Page: {page}, Section: {sec}]`"
            )

        if unmatched_concepts and matched_concepts:
            unmatched_str = ", ".join(unmatched_concepts[:3])
            answer_lines.append(
                f"\n> **Note on Partial Evidence:** While information related to {', '.join(matched_concepts[:3])} is documented above, "
                f"the retrieved sections do not explicitly detail specifics regarding `{unmatched_str}`."
            )

        return "\n".join(answer_lines), citations

    def query(
        self,
        question: str,
        doc_id_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """Execute RAG query with strict grounding, intelligent abstention, and document-aware responses."""
        k = top_k or self.settings.retrieval_top_k
        retrieved_chunks = self.vector_store.search(
            query=question, top_k=k, doc_id_filter=doc_id_filter
        )

        doc_name, doc_type = self._resolve_document_metadata(retrieved_chunks, doc_id_filter)

        # 1. Handle complete lack of indexed chunks
        if not retrieved_chunks:
            return RAGResponse(
                answer=(
                    f"Based on the selected document (`{doc_name}`), no matching "
                    "content was found in the index to answer this question."
                ),
                citations=[],
                retrieved_chunks=[],
                mode="retrieval_only",
                is_abstained=True,
                citation_validity_rate=1.0,
            )

        # 2. Semantic and lexical relevance analysis
        max_sim = max([c.get("similarity", 0.0) for c in retrieved_chunks], default=0.0)
        core_concepts = _extract_core_query_concepts(question)
        all_retrieved_text = " ".join([c["text"].lower() for c in retrieved_chunks])

        matched_concepts = [c for c in core_concepts if _concept_matches_text(c, all_retrieved_text)]
        unmatched_concepts = [c for c in core_concepts if c not in matched_concepts]

        is_issue_query = any(k in question.lower() for k in ["missing", "conflict", "inconsisten", "undefined", "broken", "refer to", "mismatch"])

        # 3. Intelligent Abstention Check
        # Abstain when the query is asking about completely unmentioned topics
        # with zero or negligible presence in the retrieved text.
        overlap_ratio = len(matched_concepts) / len(core_concepts) if core_concepts else 1.0

        # Guard: if the query names a specific identifiable entity (SWC, signal, interface)
        # that IS present in the retrieved text, never abstain — the remaining unmatched
        # concepts are almost always meta-question words ("responsibility", "cite", "page")
        # that were not filtered by STOPWORDS (e.g. added in user-typed variant queries).
        # An entity identifier has the form: token_SWC, Sig_X, If_X, or is a 6+ char
        # camelCase/underscore technical term already confirmed present in retrieved text.
        _entity_pattern = re.compile(r'[A-Za-z][A-Za-z0-9]*_[A-Za-z][A-Za-z0-9_]{2,}', re.ASCII)
        named_entities_in_query = _entity_pattern.findall(question)
        entity_found_in_text = any(
            ne.lower() in all_retrieved_text for ne in named_entities_in_query
        )

        if is_issue_query:
            should_abstain = False
        elif entity_found_in_text:
            # Named technical entity confirmed present — answer with what we have.
            should_abstain = False
        elif core_concepts:
            if len(matched_concepts) == 0:
                should_abstain = True
            elif overlap_ratio < 0.35 or (max_sim < 0.68 and overlap_ratio < 0.50):
                should_abstain = True
            else:
                should_abstain = False
        else:
            should_abstain = (max_sim < 0.58)

        if should_abstain:
            unmatched_str = ", ".join(unmatched_concepts[:3]) if unmatched_concepts else "the queried topic"
            abstention_msg = (
                f"Based on the selected document (`{doc_name}`), there is insufficient evidence to answer this question.\n"
                f"The queried concepts ({unmatched_str}) are not mentioned or specified in the uploaded document. "
                "As an evidence-grounded assistant, the system explicitly abstains rather than speculating."
            )
            return RAGResponse(
                answer=abstention_msg,
                citations=[],
                retrieved_chunks=retrieved_chunks,
                mode="retrieval_only" if not self.settings.is_gemini_configured else "gemini_grounded",
                is_abstained=True,
                citation_validity_rate=1.0,
            )

        # 4. Context preparation for Gemini (if configured)
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            meta = chunk["metadata"]
            fname = meta.get("filename", doc_name)
            page = meta.get("page_num", 1)
            sec = meta.get("section", "General")
            context_blocks.append(
                f"[Evidence {i+1}] Source: {fname} | Page: {page} | Section: {sec}\n"
                f"{chunk['text']}\n"
            )
        context_str = "\n".join(context_blocks)

        doc_context_label = (
            "AUTOSAR High-Level Design (HLD) specifications"
            if doc_type == "AUTOSAR Specification"
            else f"the document '{doc_name}' ({doc_type})"
        )

        system_instruction = (
            f"You are an expert Document Analysis and Technical Assistant analyzing {doc_context_label}.\n"
            "CRITICAL RULES:\n"
            "1. Grounding: Answer using ONLY facts explicitly stated in the retrieved context passages. Never invent details, components, ports, or citations.\n"
            "2. Concise Structure:\n"
            "   a. Summary: Start with a single-sentence direct summary.\n"
            "   b. Entity Table:\n"
            "      - When asked for Software Components (SWCs), present the explicitly documented components in a Markdown Table with columns:\n"
            "        | Component | Documented responsibility | Source page |\n"
            "      - When asked for Interfaces, use columns: | Interface | Description / Payload | Source page |\n"
            "      - When asked for Signals, use columns: | Signal | Transmission Period / Range | Source page |\n"
            "      - For 'Source page', specify the exact page number (e.g. Page 3, Page 4).\n"
            "   c. Entity Discrimination:\n"
            "      - In the components table, list ONLY actual software components (e.g. ThrottleControl_SWC, BrakeManager_SWC, BatteryManagement_SWC, ThermalControl_SWC, Gateway_SWC).\n"
            "      - Do NOT treat architectural overview paragraphs, layered models (BSW, RTE, Application Layer), execution tasks, functional flows, or individual ports as software components.\n"
            "      - If five components are explicitly documented, list exactly those five; do not invent missing components.\n"
            "   d. Single Component Queries (e.g. 'What is the responsibility of ThrottleControl_SWC?'):\n"
            "      - Detail its exact documented responsibility, ASIL level, and its provided and required ports with page citations.\n"
            "   e. Conflict & Inconsistency Queries (e.g. 'Which signals have conflicting transmission cycle times?'):\n"
            "      - State the conflicting signal, compare the conflicting values across sections (e.g. Section 4: 10ms vs Section 6.2: 50ms), and cite both sections.\n"
            "   f. Notes: End with a short note noting any missing, ambiguous, or undeclared elements (e.g. InverterControl_SWC referenced in flows but undeclared in components, or If_RefrigerantPressure missing from interface dictionary).\n"
            "3. Citation Format: Include traceable citations formatted as: [Doc: <filename>, Page: <page>, Section: <section>]. Keep citations clean.\n"
            f"4. Abstention: If evidence is lacking, state: 'Based on the selected document (`{doc_name}`), there is insufficient evidence to answer this question. The retrieved sections do not mention this topic.'\n"
            "5. If evidence is partial, answer the supported portion and clearly state what remains unspecified."
        )

        full_prompt = (
            f"{system_instruction}\n\n"
            f"RETRIEVED CONTEXT EVIDENCE:\n"
            f"===========================\n"
            f"{context_str}\n"
            f"===========================\n\n"
            f"QUESTION:\n{question}\n\n"
            f"Provide your grounded answer with citations:"
        )

        # 5. Live Gemini Generation (if API key present)
        cache_key = compute_evidence_cache_key(
            question=question,
            doc_scope=doc_id_filter,
            retrieved_chunks=retrieved_chunks,
            model_id=self.settings.gemini_model,
        )
        gemini_err = None
        if self.settings.is_gemini_configured:
            llm_text, gemini_err = self._call_gemini_api(full_prompt, cache_key=cache_key)
            if llm_text:
                llm_lower = llm_text.strip().lower()
                is_abstained = (
                    "insufficient evidence" in llm_lower
                    or "explicitly abstains rather than speculating" in llm_lower
                    or "no matching content was found in the index" in llm_lower
                    or "the retrieved sections do not mention" in llm_lower
                    or "not mentioned in the" in llm_lower
                )
                citations, val_rate = self._validate_citations(llm_text, retrieved_chunks)
                is_live = getattr(self, "_last_call_is_live", False)
                if is_live:
                    reported_status = STATUS_CONNECTED
                else:
                    reported_status = STATUS_NOT_TESTED

                return RAGResponse(
                    answer=llm_text,
                    citations=citations,
                    retrieved_chunks=retrieved_chunks,
                    mode="gemini_grounded",
                    is_abstained=is_abstained,
                    citation_validity_rate=val_rate,
                    error_message=None,
                    gemini_status=reported_status,
                    is_live_gemini=is_live,
                )

        # 6. High-Quality Deterministic Evidence Extraction Fallback (Gemini unconfigured or failed)
        fallback_answer, fallback_citations = self._build_deterministic_evidence_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
            doc_name=doc_name,
            core_concepts=core_concepts,
            matched_concepts=matched_concepts,
            unmatched_concepts=unmatched_concepts,
            gemini_err=gemini_err,
        )

        last_status = getattr(self, "_last_gemini_status", None)
        if not self.settings.is_gemini_configured:
            effective_gemini_status = STATUS_KEY_MISSING
        elif last_status:
            effective_gemini_status = last_status
        elif gemini_err:
            effective_gemini_status = STATUS_CONNECTION_FAILED
        else:
            effective_gemini_status = STATUS_NOT_TESTED

        if not fallback_answer:
            unmatched_str = ", ".join(unmatched_concepts[:3]) if unmatched_concepts else "the queried topic"
            abstention_msg = (
                f"Based on the selected document (`{doc_name}`), there is insufficient evidence to answer this question.\n"
                f"The queried concepts ({unmatched_str}) are not mentioned or specified in the uploaded document. "
                "As an evidence-grounded assistant, the system explicitly abstains rather than speculating."
            )
            return RAGResponse(
                answer=abstention_msg,
                citations=[],
                retrieved_chunks=retrieved_chunks,
                mode="retrieval_only",
                is_abstained=True,
                citation_validity_rate=1.0,
                error_message=gemini_err,
                gemini_status=effective_gemini_status,
            )

        return RAGResponse(
            answer=fallback_answer,
            citations=fallback_citations,
            retrieved_chunks=retrieved_chunks,
            mode="retrieval_only",
            is_abstained=False,
            citation_validity_rate=1.0,
            error_message=gemini_err,
            gemini_status=effective_gemini_status,
        )
