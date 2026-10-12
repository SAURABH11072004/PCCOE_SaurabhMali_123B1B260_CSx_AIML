# Evaluation Methodology

## 1. Overview
The evaluation framework objectively assesses the performance of the AUTOSAR HLD Document Analysis Assistant across four critical evaluation dimensions:
1. **Retrieval Performance (Recall@k, MRR)**
2. **Citation Validity & Grounding Rate**
3. **Answer Correctness & Fact Verification**
4. **Abstention on Out-of-Domain / Insufficient Evidence Queries**

---

## 2. Metrics & Mathematical Definitions

### 2.1 Retrieval Recall@k
Measures whether the ground-truth target chunk or section is present in the top-k retrieved chunks:
$$\text{Recall@k} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\text{TargetChunk}(q) \in \text{RetrievedTopK}(q))$$

### 2.2 Citation Validity Rate
Measures the proportion of citations emitted by the system that correspond to genuine indexed chunks and valid document pages:
$$\text{Citation Validity} = \frac{\text{Number of Valid Verified Citations}}{\text{Total Emitted Citations}}$$

### 2.3 Answer Fact F1 / Keyword Overlap
For factual queries, checks whether mandatory expected architectural facts (e.g., component names, signal types, update periods) are present in the answer.

### 2.4 Abstention Accuracy
Measures whether the system correctly abstains on negative queries (questions regarding non-existent components or unmentioned requirements):
$$\text{Abstention Accuracy} = \frac{\text{Correct Abstentions on Negative Queries}}{\text{Total Negative Queries}}$$

---

## 3. Evaluation Dataset Structure
Stored in `Input_Data/evaluation_dataset.json` and `Evaluation_Results/evaluation_questions.json`.
Contains 20 curated questions covering:
- Direct factual lookups (Component, Port, Interface)
- Signal specifications (Data types, update cycles)
- Functional flows across components
- Cross-section inconsistency detection
- Unanswerable negative queries
