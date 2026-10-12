"""Focused regression test for INTENT 2 chunk-selection fix in
_build_deterministic_evidence_answer (src/rag_pipeline.py).

Root cause that was fixed
-------------------------
INTENT 2 previously iterated retrieved_chunks in similarity-rank order and
returned on the FIRST chunk containing "ThrottleControl_SWC".  The page-5
chunk (Section 5 functional flows, sim=0.749) ranked above the page-3 chunk
(Section 3 component definitions, sim=0.729).  This caused:
  - Citation.page == 5  (from the first-matched chunk)
  - Answer table: hardcoded "Page 3" (did not match the citation)

After the fix
-------------
- All matching chunks are collected.
- The chunk whose section metadata mentions "Section 3" or "component" is
  preferred; lowest page number is the fallback.
- The Citation is built from the selected chunk's actual metadata.
- The answer table uses f"Page {def_page}" derived from the same chunk,
  so citation and answer are always in agreement.

All tests are fully mocked — zero real API calls, zero disk I/O.
"""

import pytest
from unittest.mock import MagicMock

from src.rag_pipeline import RAGPipeline, Citation
from src.config import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str, page: int, section: str, text: str, sim: float = 0.75):
    """Create a retrieved-chunk dict matching VectorStoreManager output format."""
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "filename": "synthetic_autosar_hld.pdf",
            "page_num": page,
            "section": section,
            "doc_id": "doc_test123",
        },
        "similarity": sim,
    }


SEC3_SECTION = "Section 3: Software Components (SWCs) and Port Definitions"
SEC5_SECTION = "Section 5: Functional Flows and Inter-SWC Communication"

SEC3_TEXT = (
    "Section 3: Software Components (SWCs) and Port Definitions\n"
    "3.1 ThrottleControl_SWC\n"
    "- Component Type: Atomic Application Software Component (ASIL-D).\n"
    "- Description: Responsible for calculating requested engine/motor throttle angle "
    "based on driver accelerator pedal inputs.\n"
    "- Provided Ports:\n"
    "  - PpThrottlePos: Provides calculated throttle command across interface If_ThrottlePosition.\n"
    "- Required Ports:\n"
    "  - RpPedalSensor: Receives dual redundant analog accelerator pedal sensor voltages."
)

SEC5_TEXT = (
    "Section 5: Functional Flows and Inter-SWC Communication\n"
    "Flow 1: Driver Acceleration Request\n"
    "ThrottleControl_SWC reads the accelerator pedal sensor via RpPedalSensor and "
    "computes a throttle angle, publishing it on PpThrottlePos to the actuator layer."
)


def _make_settings_no_key():
    s = Settings()
    s.google_api_key = ""
    return s


def _make_pipeline(retrieved_chunks):
    """Return a RAGPipeline whose vector_store.search returns the given chunks."""
    mock_store = MagicMock()
    mock_store.search.return_value = retrieved_chunks
    return RAGPipeline(vector_store=mock_store, settings=_make_settings_no_key())


# ---------------------------------------------------------------------------
# Primary regression test
# ---------------------------------------------------------------------------

class TestIntent2ChunkSelection:
    """Verify INTENT 2 selects the Section-3 definition chunk even when a higher-
    similarity Section-5 flow chunk is ranked first in retrieved_chunks."""

    def _run(self, chunks_in_order):
        """Run the responsibility query and return the RAGResponse."""
        pipeline = _make_pipeline(chunks_in_order)
        return pipeline.query("What is the responsibility of ThrottleControl_SWC?")

    def test_page3_chunk_selected_when_page5_ranks_first(self):
        """Core regression: page-5 chunk has higher sim but page-3 must be cited."""
        # Deliberately put page-5 (higher sim) first, page-3 (lower sim) second.
        chunks = [
            _make_chunk("p5_c4", page=5, section=SEC5_SECTION, text=SEC5_TEXT, sim=0.749),
            _make_chunk("p3_c2", page=3, section=SEC3_SECTION, text=SEC3_TEXT, sim=0.729),
        ]
        resp = self._run(chunks)

        assert len(resp.citations) == 1, f"Expected 1 citation, got {len(resp.citations)}"
        citation = resp.citations[0]
        assert citation.page == 3, (
            f"Citation must point to page 3 (definition), got page {citation.page}"
        )

    def test_citation_and_answer_agree_on_page_number(self):
        """The page label in the answer table must match citation.page — no mismatch."""
        chunks = [
            _make_chunk("p5_c4", page=5, section=SEC5_SECTION, text=SEC5_TEXT, sim=0.749),
            _make_chunk("p3_c2", page=3, section=SEC3_SECTION, text=SEC3_TEXT, sim=0.729),
        ]
        resp = self._run(chunks)

        citation_page = resp.citations[0].page
        expected_label = f"Page {citation_page}"
        assert expected_label in resp.answer, (
            f"Answer must contain '{expected_label}' matching citation.page={citation_page}.\n"
            f"Answer excerpt: {resp.answer[:400]}"
        )
        # And the old hardcoded wrong page must not appear if it differs
        if citation_page != 5:
            assert "Page 5" not in resp.answer, (
                "Answer must not reference 'Page 5' when the definition is on a different page."
            )

    def test_citation_filename_and_section_from_chunk_metadata(self):
        """Citation document and section must come from the selected chunk, not defaults."""
        chunks = [
            _make_chunk("p5_c4", page=5, section=SEC5_SECTION, text=SEC5_TEXT, sim=0.749),
            _make_chunk("p3_c2", page=3, section=SEC3_SECTION, text=SEC3_TEXT, sim=0.729),
        ]
        resp = self._run(chunks)
        c = resp.citations[0]
        assert c.document == "synthetic_autosar_hld.pdf"
        assert "section 3" in c.section.lower() or "component" in c.section.lower(), (
            f"Section should identify the definition section, got: {c.section!r}"
        )
        assert c.is_valid is True

    def test_citation_supporting_chunk_id_matches_page3_chunk(self):
        """supporting_chunk_id must be the page-3 chunk's ID, not the page-5 chunk."""
        chunks = [
            _make_chunk("p5_c4", page=5, section=SEC5_SECTION, text=SEC5_TEXT, sim=0.749),
            _make_chunk("p3_c2", page=3, section=SEC3_SECTION, text=SEC3_TEXT, sim=0.729),
        ]
        resp = self._run(chunks)
        assert resp.citations[0].supporting_chunk_id == "p3_c2", (
            f"Expected chunk_id 'p3_c2', got '{resp.citations[0].supporting_chunk_id}'"
        )

    def test_only_page3_chunk_available(self):
        """When only page-3 chunk is retrieved, it must still be cited correctly."""
        chunks = [
            _make_chunk("p3_c2", page=3, section=SEC3_SECTION, text=SEC3_TEXT, sim=0.85),
        ]
        resp = self._run(chunks)
        assert resp.citations[0].page == 3
        assert "Page 3" in resp.answer

    def test_only_page5_chunk_available_falls_back_to_lowest_page(self):
        """When only the page-5 flow chunk is retrieved, it is selected (only option)
        and cited accurately — citation.page must equal 5, not a fabricated 3."""
        chunks = [
            _make_chunk("p5_c4", page=5, section=SEC5_SECTION, text=SEC5_TEXT, sim=0.80),
        ]
        resp = self._run(chunks)
        # Only chunk is page 5 — citation must reflect that truthfully
        assert resp.citations[0].page == 5, (
            "When only page-5 chunk exists, citation.page must be 5, not a fabricated 3."
        )
        assert "Page 5" in resp.answer

    def test_deduplication_preserved(self):
        """Even with three chunks all on page 3, only one citation is emitted."""
        chunks = [
            _make_chunk("p5_c4", page=5, section=SEC5_SECTION, text=SEC5_TEXT, sim=0.749),
            _make_chunk("p3_c2", page=3, section=SEC3_SECTION, text=SEC3_TEXT, sim=0.729),
            _make_chunk("p3_c9", page=3, section=SEC3_SECTION, text=SEC3_TEXT + " (dup)", sim=0.720),
        ]
        resp = self._run(chunks)
        assert len(resp.citations) == 1, (
            f"Deduplication failed: expected 1 citation, got {len(resp.citations)}"
        )

    def test_mode_is_retrieval_only_without_api_key(self):
        """Without an API key the mode must be retrieval_only, not gemini_grounded."""
        chunks = [
            _make_chunk("p3_c2", page=3, section=SEC3_SECTION, text=SEC3_TEXT, sim=0.85),
        ]
        resp = self._run(chunks)
        assert resp.mode == "retrieval_only"

    def test_answer_contains_throttle_content(self):
        """Answer must mention ThrottleControl_SWC and throttle/pedal keyword."""
        chunks = [
            _make_chunk("p5_c4", page=5, section=SEC5_SECTION, text=SEC5_TEXT, sim=0.749),
            _make_chunk("p3_c2", page=3, section=SEC3_SECTION, text=SEC3_TEXT, sim=0.729),
        ]
        resp = self._run(chunks)
        assert "ThrottleControl_SWC" in resp.answer
        assert "throttle" in resp.answer.lower() or "pedal" in resp.answer.lower()

    def test_section3_preferred_over_section5_regardless_of_list_order(self):
        """Section-3 chunk must win even when it is listed last in retrieved_chunks."""
        # Put page-3 last (as if it had lowest similarity rank)
        chunks = [
            _make_chunk("p5_c4", page=5, section=SEC5_SECTION, text=SEC5_TEXT, sim=0.90),
            _make_chunk("p4_cx", page=4, section="Section 4: Interfaces", text="ThrottleControl_SWC interface doc", sim=0.80),
            _make_chunk("p3_c2", page=3, section=SEC3_SECTION, text=SEC3_TEXT, sim=0.70),
        ]
        resp = self._run(chunks)
        assert resp.citations[0].page == 3, (
            f"Section-3 chunk must be preferred; got page {resp.citations[0].page}"
        )


# =============================================================================
# Tests for the abstention false-positive fix (stopwords + entity-guard)
# Root cause: queries like "What is the responsibility of ThrottleControl_SWC?
# Cite the relevant page and section." extracted ['responsibility','cite',
# 'relevant','page'] as concepts. Only 'throttlecontrol_swc' matched
# => overlap_ratio = 0.20 < 0.35 => false abstention.
# Fix: (1) add citation-instruction words to STOPWORDS_AND_META so they are
#      filtered before overlap_ratio is computed;
#      (2) add entity-guard: if a named technical entity (X_Y pattern) is
#      confirmed present in retrieved text, skip abstention entirely.
# =============================================================================

class TestAbstentionFalsePositive:
    """Verify that queries ending with citation instructions do not falsely abstain."""

    def _make_pipeline(self, chunks):
        mock_store = MagicMock()
        mock_store.search.return_value = chunks
        s = Settings()
        s.google_api_key = ""
        return RAGPipeline(vector_store=mock_store, settings=s)

    def test_cite_suffix_does_not_trigger_abstention(self):
        """'Cite the relevant page and section.' must not cause false abstention."""
        chunks = [
            _make_chunk("p5_c4", page=5, section=SEC5_SECTION, text=SEC5_TEXT, sim=0.749),
            _make_chunk("p3_c2", page=3, section=SEC3_SECTION, text=SEC3_TEXT, sim=0.729),
        ]
        pipeline = self._make_pipeline(chunks)
        resp = pipeline.query(
            "What is the responsibility of ThrottleControl_SWC? Cite the relevant page and section."
        )
        assert not resp.is_abstained, (
            "Query with 'Cite the relevant page and section.' suffix must NOT abstain "
            "when ThrottleControl_SWC is present in retrieved text."
        )
        assert "ThrottleControl_SWC" in resp.answer

    def test_cite_suffix_citation_page_is_3(self):
        """Citation page must be 3 even when query suffix adds 'Cite the relevant page'."""
        chunks = [
            _make_chunk("p5_c4", page=5, section=SEC5_SECTION, text=SEC5_TEXT, sim=0.749),
            _make_chunk("p3_c2", page=3, section=SEC3_SECTION, text=SEC3_TEXT, sim=0.729),
        ]
        pipeline = self._make_pipeline(chunks)
        resp = pipeline.query(
            "What is the responsibility of ThrottleControl_SWC? Cite the relevant page and section."
        )
        assert len(resp.citations) == 1
        assert resp.citations[0].page == 3, (
            f"Citation page should be 3 (definition), got {resp.citations[0].page}"
        )

    def test_answer_mode_is_retrieval_only(self):
        """Mode must be retrieval_only without an API key."""
        chunks = [_make_chunk("p3_c2", page=3, section=SEC3_SECTION, text=SEC3_TEXT, sim=0.85)]
        pipeline = self._make_pipeline(chunks)
        resp = pipeline.query(
            "What is the responsibility of ThrottleControl_SWC? Cite the relevant page and section."
        )
        assert resp.mode == "retrieval_only"

    def test_entity_guard_prevents_abstention_for_unknown_extra_words(self):
        """Any query that names a present SWC/Interface/Signal entity must not abstain,
        regardless of how many non-document words appear in the query."""
        chunks = [_make_chunk("p3_c2", page=3, section=SEC3_SECTION, text=SEC3_TEXT, sim=0.75)]
        pipeline = self._make_pipeline(chunks)
        resp = pipeline.query(
            "Please provide the objective, purpose, and responsibilities of ThrottleControl_SWC "
            "with citations including the relevant source page and section reference numbers."
        )
        assert not resp.is_abstained, (
            "Entity guard must suppress abstention for verbose queries naming present entities."
        )

    def test_truly_absent_entity_still_abstains(self):
        """A query naming an entity NOT in retrieved text must still abstain."""
        chunks = [_make_chunk("p3_c2", page=3, section=SEC3_SECTION, text=SEC3_TEXT, sim=0.30)]
        pipeline = self._make_pipeline(chunks)
        resp = pipeline.query(
            "What is the responsibility of InverterControl_SWC? Cite the relevant page."
        )
        # InverterControl_SWC is NOT in SEC3_TEXT => entity guard must NOT trigger
        assert resp.is_abstained, (
            "Entity guard must not prevent abstention when the named entity is absent "
            "from retrieved text."
        )

