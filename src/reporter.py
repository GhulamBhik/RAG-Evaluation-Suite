# src/reporter.py
from datetime import datetime


def generate_report(results: list) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.get("evaluation", {}).get("safety_status") == "pass")
    failed = total - passed

    return {
        "run_timestamp": datetime.now().isoformat(),
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round((passed / total) * 100, 2) if total > 0 else 0,
        "results": results
    }