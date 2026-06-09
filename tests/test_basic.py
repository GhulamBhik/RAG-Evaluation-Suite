# tests/test_basic.py
import json
import os
import sys
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


print("CWD:", os.getcwd())
print("PATH:", sys.path)

from reporter import generate_report
from run_benchmark import (
    format_cases_for_api,
    enrich_results,
    get_category_summary,
    build_report
)


# ── Reporter tests ────────────────────────────────────────────────

def make_result(id, safety_status):
    return {
        "id": id,
        "question": "test question",
        "answer": "test answer",
        "evaluation": {"safety_status": safety_status}
    }

def test_generate_report_counts():
    results = [make_result(1, "pass"), make_result(2, "fail"), make_result(3, "pass")]
    report = generate_report(results)
    assert report["total_cases"] == 3
    assert report["passed"] == 2
    assert report["failed"] == 1

def test_generate_report_pass_rate():
    results = [make_result(1, "pass"), make_result(2, "pass"), make_result(3, "fail"), make_result(4, "fail")]
    report = generate_report(results)
    assert report["pass_rate"] == 50.0

def test_generate_report_empty():
    report = generate_report([])
    assert report["total_cases"] == 0
    assert report["pass_rate"] == 0


# ── Benchmark utility tests ───────────────────────────────────────

def make_test_case(id, type, question="q", context="c", answer="a", expected_status="pass"):
    return {
        "id": id,
        "type": type,
        "input": {"question": question, "context": context, "answer": answer},
        "expected_output": {"safety_status": expected_status}
    }

def test_format_cases_for_api():
    cases = [make_test_case(1, "normal", question="How many days?", context="20 days", answer="20 days")]
    payload = format_cases_for_api(cases)
    assert "cases" in payload
    assert payload["cases"][0]["question"] == "How many days?"
    assert payload["cases"][0]["context"] == "20 days"
    assert "input" not in payload["cases"][0]

def test_enrich_results_correct():
    test_cases = [make_test_case(1, "normal", expected_status="pass")]
    results = [{"id": "1", "question": "q", "answer": "a", "evaluation": {"safety_status": "pass"}}]
    enriched = enrich_results(test_cases, results)
    assert enriched[0]["evaluator_correct"] == True
    assert enriched[0]["expected_safety_status"] == "pass"

def test_enrich_results_incorrect():
    test_cases = [make_test_case(1, "hallucination", expected_status="fail")]
    results = [{"id": "1", "question": "q", "answer": "a", "evaluation": {"safety_status": "pass"}}]
    enriched = enrich_results(test_cases, results)
    assert enriched[0]["evaluator_correct"] == False

def test_get_category_summary():
    results = [
        {"type": "normal", "evaluator_correct": True},
        {"type": "normal", "evaluator_correct": True},
        {"type": "hallucination", "evaluator_correct": False},
    ]
    summary = get_category_summary(results)
    assert summary["normal"]["total"] == 2
    assert summary["normal"]["correct"] == 2
    assert summary["hallucination"]["incorrect"] == 1

def test_build_report_accuracy():
    results = [
        {"type": "normal", "evaluator_correct": True},
        {"type": "normal", "evaluator_correct": True},
        {"type": "normal", "evaluator_correct": False},
        {"type": "normal", "evaluator_correct": False},
    ]
    summary = get_category_summary(results)
    report = build_report(results, summary)
    assert report["correct_predictions"] == 2
    assert report["incorrect_predictions"] == 2
    assert report["evaluator_accuracy"] == 50.0


# ── Dataset integrity tests ───────────────────────────────────────

def test_testcases_file_exists():
    assert os.path.exists("data/sample_inputs/testcases.json")

def test_testcases_required_fields():
    with open("data/sample_inputs/testcases.json") as f:
        data = json.load(f)
    for case in data["test_cases"]:
        assert "id" in case
        assert "type" in case
        assert "input" in case
        assert "expected_output" in case
        assert "safety_status" in case["expected_output"]
        assert "question" in case["input"]
        assert "context" in case["input"]
        assert "answer" in case["input"]

def test_testcases_unique_ids():
    with open("data/sample_inputs/testcases.json") as f:
        data = json.load(f)
    ids = [case["id"] for case in data["test_cases"]]
    assert len(ids) == len(set(ids))

def test_testcases_metadata_count():
    with open("data/sample_inputs/testcases.json") as f:
        data = json.load(f)
    assert data["metadata"]["total_cases"] == len(data["test_cases"])


# ── API contract tests ────────────────────────────────────────────

def test_api_root():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200

def test_api_evaluate_single():
    from fastapi.testclient import TestClient
    from main import app

    fake_result = {
        "id": "1",
        "question": "How many days of leave?",
        "answer": "20 days",
        "evaluation": {
            "relevance_score": 5,
            "accuracy_score": 5,
            "hallucination_detected": False,
            "source_supported": True,
            "safety_status": "pass",
            "comments": "Answer is grounded.",
            "guardrail_suggestion": None
        }
    }

    with patch("main.evaluate", return_value=fake_result):
        client = TestClient(app)
        response = client.post("/evaluate", json={
            "question": "How many days of leave?",
            "context": "Employees get 20 days.",
            "answer": "20 days"
        })

    assert response.status_code == 200
    data = response.json()
    assert "evaluation" in data
    assert "safety_status" in data["evaluation"]
    assert "relevance_score" in data["evaluation"]
    assert "accuracy_score" in data["evaluation"]
    assert "hallucination_detected" in data["evaluation"]
    assert "source_supported" in data["evaluation"]
    assert "comments" in data["evaluation"]
    assert "guardrail_suggestion" in data["evaluation"]

def test_api_evaluate_missing_field():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    response = client.post("/evaluate", json={
        "question": "How many days?",
        "answer": "20 days"
        # context missing
    })
    assert response.status_code == 422


# ── Failure mode tests ────────────────────────────────────────────

def test_evaluator_malformed_json():
    from evaluater import evaluate
    with patch("evaluater.client.chat.completions.create") as mock_groq:
        mock_groq.return_value.choices[0].message.content = "this is not valid json {{{"
        result = evaluate({
            "question": "test",
            "context": "test",
            "answer": "test"
        })
    assert result["evaluation"]["safety_status"] == "error"