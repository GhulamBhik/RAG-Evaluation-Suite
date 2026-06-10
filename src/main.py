import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from database import (
    init_db,
    create_evaluation,
    get_evaluation,
    delete_evaluation
)

from evaluater import evaluate
from database import (
    init_db,
    create_evaluation
)

app = FastAPI(
    title="RAG Safety Evaluator",
    description="Evaluates whether a RAG assistant gives grounded, safe, and source-supported answers with Human-in-the-Loop review.",
    version="2.0.0"
)

# Initialize database on startup
init_db()


# -----------------------------
# Request Schemas
# -----------------------------

class ResultRequest(BaseModel):
    id: str


class ResultBatchRequest(BaseModel):
    ids: List[str]

class EvaluateRequest(BaseModel):
    id: Optional[str] = None
    question: str
    context: str
    answer: str


class BatchRequest(BaseModel):
    cases: List[EvaluateRequest]


# -----------------------------
# Routes
# -----------------------------

@app.get("/")
def root():
    return {
        "message": "RAG Safety Evaluator is running."
    }


@app.post("/evaluate")
def evaluate_one(request: EvaluateRequest):
    try:
        result = evaluate(request.model_dump())

        case_id = str(uuid.uuid4())

        create_evaluation(
            case_id=case_id,
            question=result["question"],
            context=result["context"],
            answer=result["answer"],
            evaluation=result["evaluation"]
        )

        return {
            "id": case_id,
            "status": "pending_review",
            "ai_preliminary_evaluation": result["evaluation"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/batch")
def evaluate_batch(request: BatchRequest):
    try:
        results = []

        for case in request.cases:

            result = evaluate(case.model_dump())

            case_id = str(uuid.uuid4())

            create_evaluation(
                case_id=case_id,
                question=result["question"],
                context=result["context"],
                answer=result["answer"],
                evaluation=result["evaluation"]
            )

            results.append({
                "id": case_id,
                "status": "pending_review",
                "ai_preliminary_evaluation": result["evaluation"]
            })

        return {
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/result")
def get_result(request: ResultRequest):

    record = get_evaluation(request.id)

    if not record:
        raise HTTPException(
            status_code=404,
            detail="Evaluation not found."
        )

    response = {
        "id": record["id"],
        "status": record["status"],
        "evaluation": record["evaluation"]
    }

    if record["status"] == "reviewed":
        delete_evaluation(record["id"])

    return response


@app.post("/result/batch")
def get_result_batch(request: ResultBatchRequest):

    results = []

    for evaluation_id in request.ids:

        record = get_evaluation(evaluation_id)

        if not record:
            results.append({
                "id": evaluation_id,
                "status": "not_found",
                "evaluation": None,
                "detail": "Evaluation not found."
            })
            continue

        response = {
            "id": record["id"],
            "status": record["status"],
            "evaluation": record["evaluation"]
        }

        if record["status"] == "reviewed":
            delete_evaluation(record["id"])

        results.append(response)

    return {
        "results": results
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

