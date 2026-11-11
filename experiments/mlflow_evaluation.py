"""Prepare and export synthetic reference cases for the local MLflow fit check."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities import (
    AssessmentSourceType,
    Feedback,
    MlflowExperimentLocation,
    Trace,
)

EXPERIMENT_NAME = "ChatCV - Readable Review"
CASE_TAG = "chatcv.fit_check.case_id"
PYTHON_EVIDENCE = "Page 1: Developed Python tools for data processing."
PREPARED_CASES = (
    {
        "case_id": "demo-1",
        "answerability": "full",
        "question": "Which language did the candidate use for data processing?",
        "expected_facts": ["The candidate used Python."],
        "expected_evidence": [PYTHON_EVIDENCE],
        "answer": "The candidate used Python.",
        "evidence": [PYTHON_EVIDENCE],
    },
    {
        "case_id": "demo-2",
        "answerability": "partial",
        "question": "Which language did the candidate use, and for how many years?",
        "expected_facts": ["The candidate used Python."],
        "expected_evidence": [PYTHON_EVIDENCE],
        "answer": "The candidate used Python. The resume does not specify how many years.",
        "evidence": [PYTHON_EVIDENCE],
    },
    {
        "case_id": "demo-3",
        "answerability": "none",
        "question": "Which AWS certification does the candidate hold?",
        "expected_facts": [],
        "expected_evidence": [],
        "answer": "The resume does not say whether the candidate holds an AWS certification.",
        "evidence": [],
    },
)


def seed_fit_check(client: MlflowClient, experiment_id: str) -> None:
    """Record one prepared trace per answerability class, preserving earlier reviews."""
    existing = {
        trace.info.tags.get(CASE_TAG)
        for trace in client.search_traces(
            locations=[experiment_id], include_spans=False
        )
    }
    for case in PREPARED_CASES:
        case_id = str(case["case_id"])
        if case_id in existing:
            continue
        facts = (
            "\n\n".join(
                f"F{index}. {fact}"
                for index, fact in enumerate(case["expected_facts"], start=1)
            )
            or "None."
        )
        sources = (
            "\n".join(f"- {passage}" for passage in case["evidence"])
            or "No passages retrieved."
        )
        with mlflow.start_span(
            name=f"{case_id} ({case['answerability']})",
            trace_destination=MlflowExperimentLocation(experiment_id=experiment_id),
        ) as span:
            span.set_inputs(
                {
                    "case_id": case_id,
                    "answerability": case["answerability"],
                    "question": case["question"],
                    "expected_facts": case["expected_facts"],
                    "expected_evidence": case["expected_evidence"],
                }
            )
            span.set_outputs({"answer": case["answer"], "evidence": case["evidence"]})
            mlflow.update_current_trace(
                tags={CASE_TAG: case_id, "chatcv.result_kind": "prepared"},
                request_preview=f"**Question**\n\n{case['question']}\n\n**Expected facts**\n\n{facts}",
                response_preview=f"**Answer**\n\n{case['answer']}\n\n**Retrieved sources**\n\n{sources}",
            )
    mlflow.flush_trace_async_logging()


def reviewed_case_record(trace: Trace) -> dict[str, Any]:
    """Read a reference question, its result, and its current human feedback."""
    root = next((span for span in trace.data.spans if span.parent_id is None), None)
    if root is None:
        raise ValueError(f"Trace {trace.trace_id} has no root span")

    inputs = root.inputs
    outputs = root.outputs
    reviews = [
        {
            "name": feedback.name,
            "value": feedback.value,
            "rationale": feedback.rationale,
        }
        for feedback in trace.search_assessments(type="feedback")
        if isinstance(feedback, Feedback)
        and feedback.source.source_type == AssessmentSourceType.HUMAN
    ]
    record = {
        "case_id": inputs["case_id"],
        "answerability": inputs.get("answerability"),
        "question": inputs["question"],
        "expected_facts": inputs["expected_facts"],
        "expected_evidence": inputs.get("expected_evidence"),
        "answer": outputs["answer"],
        "evidence": outputs["evidence"],
        "reviews": reviews,
    }
    for key in ("configuration", "expected_gap"):
        if key in inputs:
            record[key] = inputs[key]
    return record


def export_reviewed_case(trace: Trace, destination: Path) -> None:
    """Write a reviewed reference case as JSON."""
    destination.write_text(
        json.dumps(reviewed_case_record(trace), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5000, help="Local MLflow port")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    tracking_uri = f"http://127.0.0.1:{args.port}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    seed_fit_check(MlflowClient(tracking_uri), experiment.experiment_id)
    print(f"{tracking_uri}/#/experiments/{experiment.experiment_id}")


if __name__ == "__main__":
    main()
