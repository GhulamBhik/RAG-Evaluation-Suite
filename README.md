RAG Evaluation Suite
====================

Lightweight toolkit to evaluate Retrieval-Augmented Generation (RAG) answers for hallucination, source support and safety with a human-in-the-loop review dashboard.

Quick links
- Code: [src/main.py](src/main.py)
- Human review UI: [review_dashboard.py](review_dashboard.py)
- Benchmarks: [benchmark/run_benchmark_requester.py](benchmark/run_benchmark_requester.py), [benchmark/run_benchmark_aggregator.py](benchmark/run_benchmark_aggregator.py)
- Tests: [tests/test_basic.py](tests/test_basic.py)

Requirements
- Python 3.11
- Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell
pip install -r requirements.txt
```

Initialize database

```bash
python src/database.py
```

Run the API

```bash
uvicorn src.main:app --reload
```

Run the human review dashboard

```bash
streamlit run review_dashboard.py
```

Benchmark workflow (two-step)

1) Send test cases and save pending IDs:

```bash
python benchmark/run_benchmark_requester.py
```

2) After human review completes in the dashboard, fetch final results and save simple JSON/CSV:

```bash
python benchmark/run_benchmark_aggregator.py --pending data/sample_outputs/pending_results.json
```

Run tests

```bash
pytest
```

Notes
- `/evaluate` and `/evaluate/batch` create evaluations with `status: pending`.
- Streamlit UI shows pending reviews and updates entries to `reviewed` when marked.
- `/result/batch` returns per-id results and will include `not_found` entries for missing IDs so aggregators can save partial results.
