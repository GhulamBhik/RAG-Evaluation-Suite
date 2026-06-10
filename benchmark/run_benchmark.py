from pathlib import Path
import json
import requests

API_URL = "http://localhost:8000"  # change if needed


def load_test_cases():
    path = Path("data/sample_inputs/testcases.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["test_cases"]


def format_cases_for_api(test_cases):
    return {
        "cases": [
            {
                "question": tc["input"]["question"],
                "context": tc["input"]["context"],
                "answer": tc["input"]["answer"]
            }
            for tc in test_cases
        ]
    }

def enrich_results(test_cases, results):
    enriched = []

    for tc, res in zip(test_cases, results):
        expected_safety = tc.get("expected_output", {}).get("safety_status")
        actual_safety = res.get("evaluation", {}).get("safety_status")

        enriched.append({
            "id": tc["id"],
            "type": tc["type"],
            "input": tc["input"],
            "expected_output": tc.get("expected_output"),
            "actual_output": res.get("evaluation"),
            "status": res.get("status"),
            "passed": expected_safety == actual_safety,
            "expected_safety_status": expected_safety,
            "actual_safety_status": actual_safety
        })

    return enriched


def get_category_summary(enriched):
    summary = {}

    for item in enriched:
        t = item["type"]
        summary.setdefault(t, {"total": 0, "pass": 0})

        summary[t]["total"] += 1

        if item["passed"]:
            summary[t]["pass"] += 1

    return summary

def build_report(enriched, summary):
    return {
        "summary": summary,
        "results": enriched
    }


def save_results(report):
    path = Path("data/sample_outputs/final_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def print_summary(report):
    print("\n=== BENCHMARK SUMMARY ===")
    for k, v in report["summary"].items():
        print(f"{k}: {v['pass']}/{v['total']}")