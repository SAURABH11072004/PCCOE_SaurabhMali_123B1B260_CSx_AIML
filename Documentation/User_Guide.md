# User Guide — AUTOSAR HLD Document Analysis Assistant

## 1. Introduction
The **AUTOSAR HLD Document Analysis Assistant** is designed for automotive systems engineers, safety managers, software architects, and QA engineers reviewing AUTOSAR High-Level Design (HLD) specifications.

---

## 2. Navigating the Interface

### Page 1: Home / System Overview
- Provides a real-time health summary: count of indexed documents, total chunks, Gemini API status, and OCR availability.
- Gives an overview of supported engineering workflows.

### Page 2: Document Upload & Library
- **Upload HLD PDF**: Drag and drop any AUTOSAR HLD PDF document.
- **Enable OCR**: Check the box if the PDF contains scanned pages or rasterized diagrams with text.
- **Library Management**: View currently indexed documents, check their SHA-256 fingerprint, or safely delete documents without index corruption.

### Page 3: Document Processing Status
- Inspect raw text extracted page-by-page.
- View structured tables extracted from the specification.
- Review chunk breakdown with associated section headings and character counts.

### Page 4: Ask the HLD Assistant
- **Document Filtering**: Select "All Documents" or restrict questions to a specific release.
- **Sample Queries**: Select pre-populated evaluation questions to test retrieval, signal lookup, and functional flows.
- **Inspect Citations**: Review exact source citations in `[Doc: ..., Page: ..., Section: ...]` format.
- **Evidence Snippets**: Expand the retrieved evidence drawer to view the exact text excerpts supporting the answer.
- **Abstention Behavior**: If you ask an out-of-scope question (e.g., radar frequencies on a powertrain ECU), the assistant clearly abstains rather than inventing specs.

### Page 5: Architecture Knowledge Explorer
- View extracted AUTOSAR entities:
  - **Components**: `ThrottleControl_SWC`, `BrakeManager_SWC`, `BatteryManagement_SWC`
  - **Ports**: `PpThrottlePos`, `RpPedalSensor`, `PpBrakeTorqueRequest`
  - **Interfaces**: `If_ThrottlePosition`, `If_BrakeTorque`
  - **Signals**: `Sig_ThrottleAngle`, `Sig_RequestedTorque`
- Filter by entity type or document.
- Export entities as CSV or JSON for integration with enterprise ALM/PLM tools.

### Page 6: Revision Comparison
- Select baseline (e.g. V1.0) and revised (e.g. V2.0) HLD documents.
- Review structural changes: added sections (e.g. Cybersecurity HSM integration), modified timing, and deleted legacy interfaces.
- Inspect architectural impact notes highlighting downstream components affected.

### Page 7: Potential Issues & Inconsistencies
- Review automated rule-based defect detection:
  - **Conflicting Definitions**: Discrepancies in signal cycle times across sections.
  - **Undefined Interfaces**: R-Ports that point to interfaces missing from the interface catalog.
  - **Undeclared Components**: Components referenced in functional flows but omitted from the component list.
  - **Broken Cross-References**: References to missing sections.
- Export issue findings to CSV or JSON for review meetings.

### Page 8: Evaluation Dashboard
- View empirical evaluation metrics calculated on the 20-question automotive evaluation benchmark:
  - Retrieval Recall@k
  - Citation Validity Rate
  - Citation Coverage
  - Abstention Accuracy
  - Factual Match Rate
- Re-run the evaluation benchmark with a single click.

### Page 9: Export & Reports
- Centralized hub for downloading entities, issues, diffs, and evaluation reports.

### Page 10: Configuration & Help
- Live Gemini API ping test.
- Windows OCR configuration guide.
- Confidentiality and data handling reminders.
