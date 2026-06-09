# run_benchmark.py
import json
import requests
import csv
import os
from datetime import datetime

API_URL = "http://localhost:8000/evaluate/batch"
TEST_CASES_PATH = "data/sample_inputs/testcases.json"
RESULTS_JSON_PATH = "data/sample_outputs/results.json"
RESULTS_CSV_PATH = "data/sample_outputs/results.csv"


def load_test_cases() -> list:
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]


def format_cases_for_api(test_cases: list) -> dict:
    return {
        "cases": [
            {
                "id": str(case["id"]),
                "question": case["input"]["question"],
                "context": case["input"]["context"],
                "answer": case["input"]["answer"]
            }
            for case in test_cases
        ]
    }


def enrich_results(test_cases: list, results: list) -> list:
    id_to_case = {str(case["id"]): case for case in test_cases}
    enriched = []
    for r in results:
        case = id_to_case.get(str(r["id"]), {})
        expected = case.get("expected_output", {}).get("safety_status", None)
        actual = r["evaluation"]["safety_status"]
        r["type"] = case.get("type", "unknown")
        r["expected_safety_status"] = expected
        r["evaluator_correct"] = (actual == expected)
        enriched.append(r)
    return enriched


def get_category_summary(results: list) -> dict:
    summary = {}
    for r in results:
        category = r.get("type", "unknown")
        if category not in summary:
            summary[category] = {"total": 0, "correct": 0, "incorrect": 0}
        summary[category]["total"] += 1
        if r["evaluator_correct"]:
            summary[category]["correct"] += 1
        else:
            summary[category]["incorrect"] += 1
    return summary


def build_report(results: list, category_summary: dict) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r["evaluator_correct"])
    incorrect = total - correct
    return {
        "run_timestamp": datetime.now().isoformat(),
        "total_cases": total,
        "correct_predictions": correct,
        "incorrect_predictions": incorrect,
        "evaluator_accuracy": round((correct / total) * 100, 2) if total > 0 else 0,
        "summary_by_category": category_summary,
        "results": results
    }


def print_summary(report: dict):
    print("\n=== BENCHMARK RESULTS ===")
    print(f"Total cases          : {report['total_cases']}")
    print(f"Correct predictions  : {report['correct_predictions']}")
    print(f"Incorrect predictions: {report['incorrect_predictions']}")
    print(f"Evaluator accuracy   : {report['evaluator_accuracy']}%")

    print("\n=== RESULTS BY CATEGORY ===")
    for category, stats in report["summary_by_category"].items():
        print(f"  {category:<20} total: {stats['total']}  correct: {stats['correct']}  incorrect: {stats['incorrect']}")

    print("\n=== PER CASE RESULTS ===")
    for r in report["results"]:
        marker = "✅" if r["evaluator_correct"] else "❌"
        actual = r["evaluation"]["safety_status"].upper()
        expected = r["expected_safety_status"].upper() if r["expected_safety_status"] else "N/A"
        print(f"  {marker} ID {r['id']:<4} expected: {expected:<6} got: {actual:<6} — {r['evaluation']['comments']}")


def save_results(report: dict):
    os.makedirs("data/sample_outputs", exist_ok=True)

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    fields = [
        "id", "type", "question", "answer",
        "expected_safety_status", "evaluator_correct",
        "relevance_score", "accuracy_score",
        "hallucination_detected", "source_supported",
        "safety_status", "comments", "guardrail_suggestion"
    ]

    with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in report["results"]:
            row = {
                "id": r["id"],
                "type": r["type"],
                "question": r["question"],
                "answer": r["answer"],
                "expected_safety_status": r["expected_safety_status"],
                "evaluator_correct": r["evaluator_correct"]
            }
            row.update(r["evaluation"])
            writer.writerow(row)

    print(f"\nResults saved to {RESULTS_JSON_PATH} and {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    print("Loading test cases...")
    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} test cases")

    payload = format_cases_for_api(test_cases)

    print("Calling API...")
    response = requests.post(API_URL, json=payload)

    if response.status_code != 200:
        print(f"API error: {response.status_code} - {response.text}")
        exit(1)

    results = enrich_results(test_cases, response.json()["results"])
    category_summary = get_category_summary(results)
    report = build_report(results, category_summary)

    print_summary(report)
    save_results(report)