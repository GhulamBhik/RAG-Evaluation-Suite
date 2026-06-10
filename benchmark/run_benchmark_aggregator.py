import json
import os
import requests

from run_benchmark import (
    API_URL,
    load_test_cases,
    enrich_results,
    get_category_summary,
    build_report,
    save_results,
    print_summary
)

PENDING_PATH = "data/sample_outputs/pending_results.json"


def main():
    if not os.path.exists(PENDING_PATH):
        print("Pending file not found")
        return

    with open(PENDING_PATH, "r", encoding="utf-8") as f:
        pending = json.load(f)
    ids = [result["id"] for result in pending.get("results", [])]

    if not ids:
        print("No run IDs found")
        return

    print(f"Fetching results for {len(ids)} cases...")

    resp = requests.post(
        f"{API_URL}/result/batch",
        json={"ids": ids}
    )

    if resp.status_code != 200:
        print("API Error:", resp.text)
        return

    results = resp.json().get("results", [])

    test_cases = load_test_cases()

    enriched = enrich_results(test_cases, results)
    summary = get_category_summary(enriched)

    report = build_report(enriched, summary)

    save_results(report)
    print_summary(report)


if __name__ == "__main__":
    main()