# AUTOSAR HLD Assistant — Evaluation Report

## 1. Executive Summary
This evaluation report documents the empirical performance benchmark conducted on the **AUTOSAR HLD Document Analysis Assistant**.
The benchmark evaluated the end-to-end pipeline covering document parsing, local vector retrieval, grounded question answering, citation verification, and out-of-domain query abstention.

- **Benchmark Date**: 2026-10-11T06:50:53Z
- **Evaluated Document**: `synthetic_autosar_hld.pdf` (Pages: 6, Chunks: 6)
- **Total Test Questions**: 20
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Execution Mode**: `Offline Retrieval Evaluation`
- **Overall Status**: **PASS — All Targets Met**

---

## 2. Quantitative Performance Metrics

| Metric | Measured Value | Target Standard | Status | Formula / Description |
|---|---|---|---|---|
| **Retrieval Recall@k** | **100.0%** | >= 85.0% | PASS | Proportion of factual queries where target page/section was in top-4 chunks |
| **Citation Validity Rate** | **100.0%** | 100.0% | PASS | Valid emitted citations / Total emitted citations |
| **Citation Coverage** | **100.0%** | >= 90.0% | PASS | Factual questions with verified citations / Total factual questions |
| **Abstention Accuracy** | **100.0%** | >= 80.0% | PASS | Correct abstentions on unmentioned/negative queries |
| **Factual Match Rate** | **89.1%** | >= 85.0% | PASS | Ground-truth engineering facts retrieved and presented |
| **Document Ingestion Time** | **0.60 s** | < 10.0 s | PASS | Wall-clock time for PDF parsing, chunking, and embedding |
| **Average Query Latency** | **1.403 s** | < 2.0 s | PASS | Mean retrieval and citation generation response time |

---

## 3. Detailed Query Evaluation Breakdown

| ID | Query Type | Question | Expected Page | Hits@k | Abstained | Latency |
|---|---|---|---|---|---|---|
| `Q01` | factual_lookup | What is the primary function of ThrottleContr... | Page 3 | YES | False | 4.095s |
| `Q02` | signal_lookup | What are the data type, range, and update rat... | Page 4 | YES | False | 1.606s |
| `Q03` | signal_lookup | What signals are contained within the If_Batt... | Page 4 | YES | False | 1.789s |
| `Q04` | component_lookup | Which software component controls vehicle dec... | Page 3 | YES | False | 1.602s |
| `Q05` | functional_flow | Explain Flow 1: Driver Acceleration Request F... | Page 5 | YES | False | 1.690s |
| `Q06` | functional_flow | What happens in Flow 2 if the battery State o... | Page 5 | YES | False | 1.379s |
| `Q07` | functional_flow | How does ThermalControl_SWC trigger derating ... | Page 5 | YES | False | 1.519s |
| `Q08` | inconsistency_detection | Is there a conflict regarding the transmissio... | Page 6 | YES | False | 1.578s |
| `Q09` | issue_lookup | Does ThermalControl_SWC reference any interfa... | Page 3 | YES | False | 1.512s |
| `Q10` | architecture_overview | What AUTOSAR Classic release and operating sy... | Page 1 | YES | False | 1.763s |
| `Q11` | safety_lookup | What ASIL safety integrity levels are allocat... | Page 3 | YES | False | 1.668s |
| `Q12` | task_scheduling | Which tasks in the scheduling matrix execute ... | Page 6 | YES | False | 1.417s |
| `Q13` | negative_abstention | What is the radar sensor frequency and antenn... | N/A (Negative) | YES | True | 0.071s |
| `Q14` | negative_abstention | What is the chemical composition of the catho... | N/A (Negative) | YES | True | 0.054s |
| `Q15` | negative_abstention | Which software component manages the infotain... | N/A (Negative) | YES | True | 0.038s |
| `Q16` | component_lookup | What are the ports and role of Gateway_SWC in... | Page 3 | YES | False | 1.665s |
| `Q17` | signal_lookup | What is the update rate and range of Sig_Pump... | Page 4 | YES | False | 1.250s |
| `Q18` | cross_reference | Does Section 6.3 refer to any external sectio... | Page 6 | YES | False | 1.239s |
| `Q19` | negative_abstention | What is the lidar point cloud resolution and ... | N/A (Negative) | YES | True | 0.208s |
| `Q20` | flow_inconsistency | Does Flow 2 reference any component that is m... | Page 5 | YES | False | 1.912s |

---

## 4. Methodology and Limitations
1. **Reproducibility**: The evaluation script is fully runnable using `python -m evaluation.run_evaluation`.
2. **Ground Truth Verification**: Ground-truth facts and expected sections were authored concurrently with the synthetic AUTOSAR HLD document.
3. **Operational Grounding**: When running with live Gemini credentials, responses are grounded strictly via the system prompt. In offline/retrieval-only mode, evidence snippets with exact page citations are presented directly to the user.
4. **Safety Disclaimer**: Automated architectural analysis assists human engineering review and does not constitute certified AUTOSAR or ISO 26262 compliance approval.
