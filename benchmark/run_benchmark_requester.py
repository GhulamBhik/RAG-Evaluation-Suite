import json
import os
import requests

from run_benchmark import API_URL, load_test_cases, format_cases_for_api

PENDING_PATH = "data/sample_outputs/pending_results.json"


def main():
    os.makedirs("data/sample_outputs", exist_ok=True)

    print("Loading test cases...")
    test_cases = load_test_cases()

    payload = format_cases_for_api(test_cases)

    print(f"Submitting {len(payload['cases'])} cases...")

    resp = requests.post(f"{API_URL}/evaluate/batch", json=payload)

    if resp.status_code != 200:
        print("API Error:", resp.text)
        return

    data = resp.json()

    with open(PENDING_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Saved pending run_ids →", PENDING_PATH)


if __name__ == "__main__":
    main()