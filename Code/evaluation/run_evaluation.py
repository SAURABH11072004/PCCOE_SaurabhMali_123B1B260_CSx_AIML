#!/usr/bin/env python3
"""Automated Evaluation Runner for AUTOSAR HLD Document Analysis Assistant.
Executes reproducible evaluation benchmark across 20 curated architectural test questions.
Generates real, measured evaluation artifacts in Evaluation_Results directory.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.ingestion import PDFIngestionPipeline
from src.chunking import ContextAwareChunker
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline
from evaluation.metrics import (
    calculate_retrieval_recall,
    calculate_citation_metrics,
    calculate_abstention_accuracy,
    calculate_factual_correctness,
)


def run_evaluation_benchmark():
    settings = get_settings()
    print("=" * 70)
    print("AUTOSAR HLD DOCUMENT ANALYSIS ASSISTANT — EVALUATION BENCHMARK")
    print("=" * 70)

    # 1. Verify input files
    pdf_path = settings.input_data_dir / "synthetic_autosar_hld.pdf"
    dataset_path = settings.input_data_dir / "evaluation_dataset.json"

    if not pdf_path.exists() or not dataset_path.exists():
        print("[ERROR] Test dataset or PDF missing. Generating synthetic files first...")
        from scripts.generate_synthetic_hld import main as gen_main
        gen_main()

    with open(dataset_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} evaluation questions from {dataset_path.name}")

    # 2. Ingest and index the evaluation document
    print(f"\n[1/4] Ingesting and Indexing Document: {pdf_path.name}...")
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    ingestion_start = time.time()
    ingestor = PDFIngestionPipeline(enable_ocr=False)
    doc_data = ingestor.process_pdf(pdf_bytes, pdf_path.name)
    chunker = ContextAwareChunker(target_chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    chunks = chunker.chunk_document(doc_data)

    embedding_mgr = EmbeddingManager(model_name=settings.embedding_model)
    # Use distinct test collection to avoid contaminating user library
    vector_store = VectorStoreManager(
        persist_directory=str(PROJECT_ROOT / "chroma_eval_db"),
        embedding_manager=embedding_mgr,
        collection_name="eval_autosar_chunks",
    )
    vector_store.index_chunks(chunks)
    ingestion_time = round(time.time() - ingestion_start, 3)

    print(f"  - Pages Processed  : {doc_data.total_pages}")
    print(f"  - Chunks Generated : {len(chunks)}")
    print(f"  - Ingestion Time   : {ingestion_time}s")
    print(f"  - Embedding Mode   : {'SentenceTransformers' if embedding_mgr.is_using_sentence_transformers else 'Deterministic Fallback'}")

    # 3. Execute RAG benchmark
    print("\n[2/4] Executing RAG Queries across Benchmark Questions...")
    rag_pipe = RAGPipeline(vector_store=vector_store, settings=settings)

    retrieval_records = []
    answer_records = []
    citation_records = []

    total_query_time = 0.0

    for i, q in enumerate(questions):
        qid = q["id"]
        qtext = q["question"]
        t0 = time.time()

        response = rag_pipe.query(question=qtext, doc_id_filter=doc_data.doc_id, top_k=settings.retrieval_top_k)
        q_elapsed = time.time() - t0
        total_query_time += q_elapsed

        # Record retrieval
        retrieval_records.append({
            "question_id": qid,
            "question": qtext,
            "target_sections": q.get("target_sections", []),
            "expected_page": q.get("expected_page"),
            "retrieved_chunks": response.retrieved_chunks,
        })

        # Record answer
        answer_records.append({
            "question_id": qid,
            "question": qtext,
            "answer": response.answer,
            "mode": response.mode,
            "is_live_gemini": getattr(response, "is_live_gemini", False),
            "is_abstained": response.is_abstained,
            "should_abstain": q.get("should_abstain", False),
            "latency_seconds": round(q_elapsed, 4),
            "retrieved_chunks": response.retrieved_chunks,
        })

        # Record citations
        cits_dict = [
            {
                "document": c.document,
                "page": c.page,
                "section": c.section,
                "is_valid": c.is_valid,
                "supporting_chunk_id": c.supporting_chunk_id,
            }
            for c in response.citations
        ]
        citation_records.append({
            "question_id": qid,
            "question": qtext,
            "should_abstain": q.get("should_abstain", False),
            "citations": cits_dict,
            "validity_rate": response.citation_validity_rate,
        })

        print(f"  [Q{i+1:02d}/{len(questions):02d}] {qid}: Latency {q_elapsed:.3f}s | Mode: {response.mode} | Abstained: {response.is_abstained}")

    avg_latency = round(total_query_time / len(questions), 4)

    # 4. Calculate metrics
    print("\n[3/4] Calculating Evaluation Metrics...")
    recall_res = calculate_retrieval_recall(questions, retrieval_records)
    citation_res = calculate_citation_metrics(citation_records)
    abstention_res = calculate_abstention_accuracy(questions, answer_records)
    fact_res = calculate_factual_correctness(questions, answer_records)

    has_live = any(a.get("is_live_gemini", False) for a in answer_records)
    all_live = all(a.get("is_live_gemini", False) for a in answer_records) if answer_records else False
    if all_live:
        exec_mode = "live_generation"
        exec_mode_label = f"Live Answer Generation ({settings.gemini_model})"
    elif has_live:
        exec_mode = "mixed_mode"
        exec_mode_label = "Mixed Mode (Live Generation & Retrieval Fallback)"
    else:
        exec_mode = "retrieval_only"
        exec_mode_label = "Offline Retrieval Evaluation"

    final_metrics = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_name": "AUTOSAR HLD Synthetic Evaluation Dataset",
        "total_questions": len(questions),
        "document_processing_time_seconds": ingestion_time,
        "average_query_latency_seconds": avg_latency,
        "embedding_model": settings.embedding_model,
        "gemini_model": settings.gemini_model,
        "execution_mode": exec_mode,
        "is_live_generation": all_live,
        "metrics": {
            **recall_res,
            **citation_res,
            **abstention_res,
            **fact_res,
        }
    }

    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRIC SUMMARY")
    print("=" * 50)
    print(f"  - Execution Mode          : {exec_mode_label}")
    print(f"  - Retrieval Recall@k      : {recall_res['retrieval_recall_at_k'] * 100:.1f}% ({recall_res['hits_at_k']}/{recall_res['factual_queries_evaluated']})")
    print(f"  - Citation Validity Rate  : {citation_res['citation_validity_rate'] * 100:.1f}% ({citation_res['valid_citations']}/{citation_res['total_citations_emitted']})")
    print(f"  - Citation Coverage Rate  : {citation_res['citation_coverage_rate'] * 100:.1f}%")
    print(f"  - Abstention Accuracy     : {abstention_res['abstention_accuracy'] * 100:.1f}% ({abstention_res['correct_abstentions']}/{abstention_res['negative_queries_count']})")
    print(f"  - Factual Match Rate      : {fact_res['factual_match_rate'] * 100:.1f}% ({fact_res['matched_facts']}/{fact_res['total_expected_facts']})")
    print(f"  - Avg Query Latency       : {avg_latency:.3f} s")
    print("=" * 50)

    # 5. Persist Evaluation Results
    print("\n[4/4] Writing Evaluation Artifacts to Evaluation_Results/...")
    out_dir = settings.evaluation_results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "evaluation_questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)

    with open(out_dir / "retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_records, f, indent=2)

    with open(out_dir / "answer_results.json", "w", encoding="utf-8") as f:
        json.dump(answer_records, f, indent=2)

    with open(out_dir / "citation_validation.json", "w", encoding="utf-8") as f:
        json.dump(citation_records, f, indent=2)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    # Evaluate metric statuses against target standards
    recall_status = "PASS" if recall_res['retrieval_recall_at_k'] >= 0.85 else "FAIL"
    citation_val_status = "PASS" if citation_res['citation_validity_rate'] >= 1.0 else "FAIL"
    citation_cov_status = "PASS" if citation_res['citation_coverage_rate'] >= 0.90 else "FAIL"
    abstention_status = "PASS" if abstention_res['abstention_accuracy'] >= 0.80 else "FAIL"
    fact_status = "PASS" if fact_res['factual_match_rate'] >= 0.85 else "FAIL"
    ingestion_status = "PASS" if ingestion_time < 10.0 else "FAIL"
    latency_status = "PASS" if avg_latency < 2.0 else "FAIL (Above Target)"

    overall_passed = all(
        s == "PASS"
        for s in [
            recall_status,
            citation_val_status,
            citation_cov_status,
            abstention_status,
            fact_status,
            ingestion_status,
            "PASS" if avg_latency < 2.0 else "FAIL",
        ]
    )
    overall_status_str = "PASS — All Targets Met" if overall_passed else "FAIL — One or More Targets Not Met"

    # Generate Markdown Report
    report_md = f"""# AUTOSAR HLD Assistant — Evaluation Report

## 1. Executive Summary
This evaluation report documents the empirical performance benchmark conducted on the **AUTOSAR HLD Document Analysis Assistant**.
The benchmark evaluated the end-to-end pipeline covering document parsing, local vector retrieval, grounded question answering, citation verification, and out-of-domain query abstention.

- **Benchmark Date**: {final_metrics['timestamp']}
- **Evaluated Document**: `{doc_data.filename}` (Pages: {doc_data.total_pages}, Chunks: {len(chunks)})
- **Total Test Questions**: {len(questions)}
- **Embedding Model**: `{settings.embedding_model}`
- **Execution Mode**: `{exec_mode_label}`
- **Overall Status**: **{overall_status_str}**

---

## 2. Quantitative Performance Metrics

| Metric | Measured Value | Target Standard | Status | Formula / Description |
|---|---|---|---|---|
| **Retrieval Recall@k** | **{recall_res['retrieval_recall_at_k'] * 100:.1f}%** | >= 85.0% | {recall_status} | Proportion of factual queries where target page/section was in top-{settings.retrieval_top_k} chunks |
| **Citation Validity Rate** | **{citation_res['citation_validity_rate'] * 100:.1f}%** | 100.0% | {citation_val_status} | Valid emitted citations / Total emitted citations |
| **Citation Coverage** | **{citation_res['citation_coverage_rate'] * 100:.1f}%** | >= 90.0% | {citation_cov_status} | Factual questions with verified citations / Total factual questions |
| **Abstention Accuracy** | **{abstention_res['abstention_accuracy'] * 100:.1f}%** | >= 80.0% | {abstention_status} | Correct abstentions on unmentioned/negative queries |
| **Factual Match Rate** | **{fact_res['factual_match_rate'] * 100:.1f}%** | >= 85.0% | {fact_status} | Ground-truth engineering facts retrieved and presented |
| **Document Ingestion Time** | **{ingestion_time:.2f} s** | < 10.0 s | {ingestion_status} | Wall-clock time for PDF parsing, chunking, and embedding |
| **Average Query Latency** | **{avg_latency:.3f} s** | < 2.0 s | {latency_status} | Mean retrieval and citation generation response time |

---

## 3. Detailed Query Evaluation Breakdown

| ID | Query Type | Question | Expected Page | Hits@k | Abstained | Latency |
|---|---|---|---|---|---|---|
"""
    for a in answer_records:
        qid = a["question_id"]
        q_obj = next(q for q in questions if q["id"] == qid)
        exp_p = q_obj.get("expected_page")
        exp_str = f"Page {exp_p}" if exp_p else "N/A (Negative)"
        hits_str = "YES" if (q_obj.get("should_abstain") or any(
            c.get("metadata", {}).get("page_num") == exp_p for c in a["retrieved_chunks"]
        )) else "NO"
        report_md += f"| `{qid}` | {q_obj['type']} | {a['question'][:45]}... | {exp_str} | {hits_str} | {a['is_abstained']} | {a['latency_seconds']:.3f}s |\n"

    report_md += """
---

## 4. Methodology and Limitations
1. **Reproducibility**: The evaluation script is fully runnable using `python -m evaluation.run_evaluation`.
2. **Ground Truth Verification**: Ground-truth facts and expected sections were authored concurrently with the synthetic AUTOSAR HLD document.
3. **Operational Grounding**: When running with live Gemini credentials, responses are grounded strictly via the system prompt. In offline/retrieval-only mode, evidence snippets with exact page citations are presented directly to the user.
4. **Safety Disclaimer**: Automated architectural analysis assists human engineering review and does not constitute certified AUTOSAR or ISO 26262 compliance approval.
"""

    with open(out_dir / "Evaluation_Report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Evaluation report generated successfully at: {out_dir / 'Evaluation_Report.md'}")
    return final_metrics


if __name__ == "__main__":
    run_evaluation_benchmark()
