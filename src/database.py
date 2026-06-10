import sqlite3
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "evaluations.db"
def get_connection():
    """
    Returns a SQLite connection.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Creates the evaluations table if it doesn't exist.
    """
    conn = get_connection()

    conn.execute("""
    CREATE TABLE IF NOT EXISTS evaluations (
        id TEXT PRIMARY KEY,

        question TEXT NOT NULL,
        context TEXT NOT NULL,
        answer TEXT NOT NULL,

        evaluation_json TEXT NOT NULL,

        status TEXT NOT NULL,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def create_evaluation(
    case_id: str,
    question: str,
    context: str,
    answer: str,
    evaluation: dict
):
    """
    Stores a new AI evaluation.
    Status starts as 'pending'.
    """

    conn = get_connection()

    conn.execute(
        """
        INSERT INTO evaluations (
            id,
            question,
            context,
            answer,
            evaluation_json,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            case_id,
            question,
            context,
            answer,
            json.dumps(evaluation),
            "pending"
        )
    )

    conn.commit()
    conn.close()


def get_evaluation(case_id: str):
    """
    Fetches a record by ID.
    Returns None if not found.
    """

    conn = get_connection()

    row = conn.execute(
        """
        SELECT *
        FROM evaluations
        WHERE id = ?
        """,
        (case_id,)
    ).fetchone()

    conn.close()

    if not row:
        return None

    return {
        "id": row["id"],
        "question": row["question"],
        "context": row["context"],
        "answer": row["answer"],
        "evaluation": json.loads(row["evaluation_json"]),
        "status": row["status"]
    }


def update_review(case_id: str, reviewed_evaluation: dict):
    """
    Overwrites AI evaluation with human-reviewed version
    and marks the case as reviewed.
    """

    conn = get_connection()

    conn.execute(
        """
        UPDATE evaluations
        SET
            evaluation_json = ?,
            status = 'reviewed'
        WHERE id = ?
        """,
        (
            json.dumps(reviewed_evaluation),
            case_id
        )
    )

    conn.commit()
    conn.close()


def get_pending_reviews():
    """
    Returns all pending reviews for Streamlit.
    """
    conn = get_connection()

    rows = conn.execute(
        """
        SELECT *
        FROM evaluations
        WHERE status = "pending"
        ORDER BY created_at ASC
        """
    ).fetchall()

    conn.close()

    results = []

    for row in rows:
        results.append({
            "id": row["id"],
            "question": row["question"],
            "context": row["context"],
            "answer": row["answer"],
            "evaluation": json.loads(row["evaluation_json"]),
            "status": row["status"]
        })

    return results


def delete_evaluation(case_id: str):
    """
    Deletes a completed evaluation after user retrieval.
    """

    conn = get_connection()

    conn.execute(
        """
        DELETE FROM evaluations
        WHERE id = ?
        """,
        (case_id,)
    )

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized.")