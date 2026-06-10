import streamlit as st
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from database import (
    get_pending_reviews,
    update_review
)

st.set_page_config(
    page_title="RAG Safety Review Dashboard",
    layout="wide"
)

st.title("RAG Safety Evaluator - Human Review Queue")

pending_cases = get_pending_reviews()

if not pending_cases:
    st.success("No pending reviews.")
    st.stop()

st.write(f"Pending Reviews: {len(pending_cases)}")

for case in pending_cases:

    with st.expander(f"Case ID: {case['id']}"):

        st.subheader("Question")
        st.write(case["question"])

        st.subheader("Context")
        st.text_area(
            "Retrieved Context",
            value=case["context"],
            height=150,
            disabled=True,
            key=f"context_{case['id']}"
        )

        st.subheader("Answer")
        st.text_area(
            "Model Answer",
            value=case["answer"],
            height=120,
            disabled=True,
            key=f"answer_{case['id']}"
        )

        ai_eval = case["evaluation"]

        st.markdown("---")
        st.subheader("Human Review")

        relevance_score = st.slider(
            "Relevance Score",
            min_value=1,
            max_value=5,
            value=int(ai_eval.get("relevance_score", 3)),
            key=f"rel_{case['id']}"
        )

        accuracy_score = st.slider(
            "Accuracy Score",
            min_value=1,
            max_value=5,
            value=int(ai_eval.get("accuracy_score", 3)),
            key=f"acc_{case['id']}"
        )

        hallucination_detected = st.checkbox(
            "Hallucination Detected",
            value=bool(ai_eval.get("hallucination_detected", False)),
            key=f"hall_{case['id']}"
        )

        source_supported = st.checkbox(
            "Source Supported",
            value=bool(ai_eval.get("source_supported", False)),
            key=f"source_{case['id']}"
        )

        safety_status = st.selectbox(
            "Safety Status",
            ["pass", "fail"],
            index=0 if ai_eval.get("safety_status") == "pass" else 1,
            key=f"safety_{case['id']}"
        )

        comments = st.text_area(
            "Comments",
            value=ai_eval.get("comments", ""),
            key=f"comments_{case['id']}"
        )

        guardrail_suggestion = st.text_area(
            "Guardrail Suggestion",
            value=ai_eval.get("guardrail_suggestion") or "",
            key=f"guardrail_{case['id']}"
        )

        if st.button(
            "Mark Reviewed",
            key=f"review_{case['id']}"
        ):

            reviewed_evaluation = {
                "relevance_score": relevance_score,
                "accuracy_score": accuracy_score,
                "hallucination_detected": hallucination_detected,
                "source_supported": source_supported,
                "safety_status": safety_status,
                "comments": comments,
                "guardrail_suggestion": (
                    guardrail_suggestion
                    if guardrail_suggestion.strip()
                    else None
                )
            }

            update_review(
                case["id"],
                reviewed_evaluation
            )

            st.success(
                f"Case {case['id']} marked as reviewed."
            )

            st.rerun()