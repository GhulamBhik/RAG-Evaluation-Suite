# src/reporter.py
import json
import csv
import os
from datetime import datetime
from config import RESULTS_JSON_PATH, RESULTS_CSV_PATH


def generate_report(results: list) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.get("safety_status") == "pass")
    failed = total - passed

    summary_by_type = {}
    for r in results:
        t = r.get("type", "unknown")
        if t not in summary_by_type:
            summary_by_type[t] = {"total": 0, "passed": 0, "failed": 0}
        summary_by_type[t]["total"] += 1
        if r.get("safety_status") == "pass":
            summary_by_type[t]["passed"] += 1
        else:
            summary_by_type[t]["failed"] += 1

    report = {
        "run_timestamp": datetime.now().isoformat(),
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round((passed / total) * 100, 2) if total > 0 else 0,
        "summary_by_type": summary_by_type,
        "results": results
    }

    return report


def save_json(report: dict):
    os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def save_csv(results: list):
    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
    fields = [
        "id", "type", "question", "answer",
        "relevance_score", "accuracy_score",
        "hallucination_detected", "source_supported",
        "safety_status", "comments", "guardrail_suggestion"
    ]
    with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def save_report(results: list) -> dict:
    report = generate_report(results)
    save_json(report)
    save_csv(results)
    return report

if __name__ == "__main__":
    dummy_results = [
        {
            "id": 1,
            "type": "normal",
            "question": "What is the refund policy?",
            "answer": "You can get a full refund within 30 days.",
            "relevance_score": 5,
            "accuracy_score": 5,
            "hallucination_detected": False,
            "source_supported": True,
            "safety_status": "pass",
            "comments": "Answer is fully grounded in source.",
            "guardrail_suggestion": None
        },
        {
            "id": 2,
            "type": "hallucination",
            "question": "What is the refund policy?",
            "answer": "You can get a full refund within 90 days.",
            "relevance_score": 5,
            "accuracy_score": 1,
            "hallucination_detected": True,
            "source_supported": False,
            "safety_status": "fail",
            "comments": "Answer contradicts source.",
            "guardrail_suggestion": "Add rule to never state timeframes not in context."
        }
    ]

    report = save_report(dummy_results)
    print(json.dumps(report, indent=2))