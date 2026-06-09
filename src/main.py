# src/main.py
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from evaluater import evaluate
from reporter import generate_report

app = FastAPI(
    title="RAG Safety Evaluator",
    description="Evaluates whether a RAG assistant gives grounded, safe, and source-supported answers.",
    version="1.0.0"
)


class EvaluateRequest(BaseModel):
    id: Optional[str] = None
    question: str
    context: str
    answer: str


class BatchRequest(BaseModel):
    cases: List[EvaluateRequest]


@app.get("/")
def root():
    return {"message": "RAG Safety Evaluator is running. Visit /docs for API documentation."}


@app.post("/evaluate")
def evaluate_one(request: EvaluateRequest):
    try:
        result = evaluate(request.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/batch")
def evaluate_batch(request: BatchRequest):
    try:
        results = []
        for case in request.cases:
            result = evaluate(case.model_dump())
            results.append(result)
        report = generate_report(results)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)