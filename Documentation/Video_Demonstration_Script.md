# Video Demonstration Script

**Project:** AUTOSAR HLD Document Analysis Assistant
**Duration Target:** 5–8 minutes

---

## Scene 1 — Application Launch (0:00–0:45)

1. Open a terminal and activate the virtual environment.
2. Run streamlit run app.py.
3. Show the browser opening at http://localhost:8501.
4. Walk through the sidebar navigation: Overview, Documents, HLD Assistant, Settings & Evaluation.

## Scene 2 — Overview Page (0:45–1:30)

1. Highlight the system status panel (Gemini API badge, Vector Store indicator).
2. Show the document count and chunk count in the header.
3. Read out the project description and feature cards.

## Scene 3 — Documents Page (1:30–3:00)

1. Show the pre-loaded synthetic_autosar_hld.pdf in the document library.
2. Demonstrate uploading synthetic_autosar_hld_v2.pdf via the file uploader.
3. Confirm the ingestion progress and the updated chunk count.
4. Show document metadata (pages, doc type, chunk count, doc hash).
5. Demonstrate the confirmation popover for document removal (without actually removing).

## Scene 4 — HLD Assistant (3:00–5:30)

1. Select the synthetic HLD document from the scope selector.
2. Type the example query: *"What software components interact with the ECU Abstraction Layer?"*
3. Show the RAG response, inline citations, and evidence cards.
4. Run a second query: *"List all identified architectural consistency issues."*
5. Show the Issues tab populated with detected problems.
6. Show the Entities tab with extracted SWCs, buses, and ECUs.

## Scene 5 — Settings & Evaluation (5:30–7:30)

1. Open Settings & Evaluation.
2. Show the benchmark summary cards (Precision, Recall, F1, Latency).
3. Expand the **Detailed Evaluation Results** section — walk through 3–4 example questions.
4. Expand **Metric Definitions & Thresholds** and explain the scoring.
5. Expand **Comprehensive Evaluation Report** and read the key findings.

## Scene 6 — Wrap-up (7:30–8:00)

1. Return to Overview and summarise the demonstrated capabilities.
2. State the evaluation results (e.g., Precision 0.82, Recall 0.78).
3. End recording.
