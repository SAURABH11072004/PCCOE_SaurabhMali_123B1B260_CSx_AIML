"""Evaluation Metrics Calculation Module.
Computes mathematically grounded performance metrics:
- Retrieval Recall@k
- Citation Validity Rate & Citation Coverage
- Factual Match Rate against ground-truth facts
- Abstention Accuracy on out-of-domain / unmentioned topics
"""

from typing import List, Dict, Any


def calculate_retrieval_recall(
    eval_questions: List[Dict[str, Any]],
    retrieval_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate Recall@k for queries having ground-truth page/section targets."""
    factual_q_count = 0
    hits = 0

    retrieval_map = {r["question_id"]: r["retrieved_chunks"] for r in retrieval_results}

    for q in eval_questions:
        if q.get("should_abstain", False):
            continue  # Negative questions do not have target chunks in document
        
        factual_q_count += 1
        expected_page = q.get("expected_page")
        target_sections = [s.lower() for s in q.get("target_sections", [])]
        retrieved = retrieval_map.get(q["id"], [])

        # Hit if any retrieved chunk matches expected page or target section
        is_hit = False
        for chunk in retrieved:
            meta = chunk.get("metadata", {})
            chunk_page = meta.get("page_num")
            chunk_sec = str(meta.get("section", "")).lower()

            if expected_page is not None and chunk_page == expected_page:
                is_hit = True
                break
            if any(ts in chunk_sec for ts in target_sections):
                is_hit = True
                break

        if is_hit:
            hits += 1

    recall = (hits / factual_q_count) if factual_q_count > 0 else 0.0
    return {
        "factual_queries_evaluated": factual_q_count,
        "hits_at_k": hits,
        "retrieval_recall_at_k": round(recall, 4),
    }


def calculate_citation_metrics(
    citation_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate citation validity and citation coverage rates."""
    total_citations = 0
    valid_citations = 0
    factual_answers_with_citations = 0
    total_factual_answers = 0

    for res in citation_results:
        if not res.get("should_abstain", False):
            total_factual_answers += 1
            cits = res.get("citations", [])
            if cits:
                factual_answers_with_citations += 1
            for c in cits:
                total_citations += 1
                if c.get("is_valid", False):
                    valid_citations += 1

    validity_rate = (valid_citations / total_citations) if total_citations > 0 else 1.0
    coverage_rate = (
        (factual_answers_with_citations / total_factual_answers)
        if total_factual_answers > 0
        else 0.0
    )

    return {
        "total_citations_emitted": total_citations,
        "valid_citations": valid_citations,
        "citation_validity_rate": round(validity_rate, 4),
        "citation_coverage_rate": round(coverage_rate, 4),
    }


def calculate_abstention_accuracy(
    eval_questions: List[Dict[str, Any]],
    answer_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate precision of abstention on out-of-scope / unanswerable queries."""
    answer_map = {a["question_id"]: a for a in answer_results}
    negative_q_count = 0
    correct_abstentions = 0

    for q in eval_questions:
        if q.get("should_abstain", False):
            negative_q_count += 1
            ans = answer_map.get(q["id"], {})
            if ans.get("is_abstained", False):
                correct_abstentions += 1

    acc = (correct_abstentions / negative_q_count) if negative_q_count > 0 else 1.0
    return {
        "negative_queries_count": negative_q_count,
        "correct_abstentions": correct_abstentions,
        "abstention_accuracy": round(acc, 4),
    }


def calculate_factual_correctness(
    eval_questions: List[Dict[str, Any]],
    answer_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate fact overlap score for factual queries against ground-truth facts."""
    answer_map = {a["question_id"]: a for a in answer_results}
    factual_q_count = 0
    total_facts = 0
    matched_facts = 0

    for q in eval_questions:
        if q.get("should_abstain", False):
            continue
        expected_facts = q.get("expected_facts", [])
        if not expected_facts:
            continue

        factual_q_count += 1
        ans_data = answer_map.get(q["id"], {})
        answer_text = ans_data.get("answer", "").lower()

        # Check in answer text or retrieved evidence if in retrieval-only mode
        retrieved_text = " ".join(
            [c.get("text", "") for c in ans_data.get("retrieved_chunks", [])]
        ).lower()
        search_corpus = f"{answer_text} {retrieved_text}"

        for fact in expected_facts:
            total_facts += 1
            if fact.lower() in search_corpus:
                matched_facts += 1

    fact_score = (matched_facts / total_facts) if total_facts > 0 else 0.0
    return {
        "evaluated_queries": factual_q_count,
        "total_expected_facts": total_facts,
        "matched_facts": matched_facts,
        "factual_match_rate": round(fact_score, 4),
    }
