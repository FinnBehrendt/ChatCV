"""Compare local MLflow judges with a completed ChatCV review export."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow.genai.scorers import Scorer
from pypdf import PdfReader

from experiments.reference_review import FACT_LABELS

RESUME = Path(__file__).resolve().parents[1] / "data" / "CV_Demo.pdf"
JUDGE_EXPERIMENT = "ChatCV - Local Judge Comparison"
INFERENCE_PARAMS = {"temperature": 0, "max_tokens": 1200, "reasoning_effort": "low"}
RETRIEVAL_PROMPT = """Evaluate retrieval using {{ inputs }}.
Return a list with one label per expected fact in F1, F2, ... order.
Use found when the retrieved passages together support that fact and missing otherwise. A topically related passage is insufficient.
Return an empty list when there are no expected facts. Give a brief rationale.
"""
ANSWER_PROMPT = """Evaluate the saved answer in {{ outputs }} using the question and references in {{ expectations }}.
Return a list of labels in this order:
1. Unsupported claims: present if any factual claim is unsupported by the complete resume, or none otherwise.
2. Information gap: n/a for a fully answerable question. For other cases, appropriate if the answer acknowledges the expected gap without guessing, or inappropriate otherwise.
3. One label per expected fact in F1, F2, ... order: correct if accurately conveyed, missing if omitted, incorrect if contradicted.
Always include the first two labels, even when there are no expected facts. For example, ["none", "n/a", "correct"] has one fact.
Give a brief rationale that identifies unsupported claims and cites source text where relevant.
"""


def _dataset(cases: list[dict[str, Any]], resume: str) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        facts = {
            f"F{number}": fact for number, fact in enumerate(case["expected_facts"], 1)
        }
        rows.append(
            {
                "inputs": {
                    "case_id": case["case_id"],
                    "question": case["question"],
                    "expected_facts": facts,
                    "retrieved_passages": case["evidence"],
                },
                "outputs": case["answer"],
                "expectations": {
                    "question": case["question"],
                    "expected_facts": facts,
                    "answerability": case["answerability"],
                    "expected_gap": case.get("expected_gap") or "",
                    "complete_resume": resume,
                },
            }
        )
    return rows


def _compare_case(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    human = case["normalized_review"]
    verdicts = {}
    rationale = {}
    errors = {}
    for judge in ("retrieval_judge", "answer_judge"):
        value = row.get(f"{judge}/value")
        verdicts[judge] = value if isinstance(value, list) else []
        explanation = row.get(f"{judge}/rationale")
        if isinstance(explanation, str):
            rationale[judge] = explanation
        error = row.get(f"{judge}/error_message")
        if isinstance(error, str):
            errors[judge] = error

    comparisons = {}
    unexpected_fields = {}
    for judge, dimensions in (
        ("retrieval_judge", ["retrieval_by_fact"]),
        ("answer_judge", ["answer_by_fact", "unsupported_claims", "information_gaps"]),
    ):
        values = verdicts[judge]
        prefix = judge.removesuffix("_judge")
        remaining = (
            dict(zip(("unsupported_claims", "information_gaps"), values[:2]))
            if judge == "answer_judge"
            else {}
        )
        facts = values[2:] if judge == "answer_judge" else values
        remaining.update(
            {f"{prefix}_f{number}": label for number, label in enumerate(facts, 1)}
        )
        for dimension in dimensions:
            expected = human[dimension]
            if dimension in FACT_LABELS:
                prefix = dimension.removesuffix("_by_fact")
                labels = (
                    {
                        f"{prefix}_{fact.lower()}": label
                        for fact, label in expected.items()
                    }
                    if isinstance(expected, dict)
                    else {}
                )
                allowed = FACT_LABELS[dimension]
            else:
                labels = {dimension: expected} if expected != "n/a" else {}
                allowed = (
                    {"none", "present"}
                    if dimension == "unsupported_claims"
                    else {"appropriate", "inappropriate"}
                )
                if expected == "n/a" and remaining.get(dimension) == "n/a":
                    remaining.pop(dimension)
            comparisons[dimension] = {
                key: _compare_label(human_label, remaining.pop(key, None), allowed)
                for key, human_label in labels.items()
            }
        if remaining:
            unexpected_fields[judge] = remaining
    return {
        "case_id": case["case_id"],
        "trace_id": case["trace_id"],
        "comparisons": comparisons,
        "rationale": rationale,
        "errors": errors,
        "unexpected_fields": unexpected_fields,
    }


def _compare_label(human: str, judge: Any, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(judge, str) or judge not in allowed:
        status = "invalid_or_missing"
    else:
        status = "matches" if judge == human else "disagreements"
    return {"human": human, "judge": judge, "status": status}


def _summarize(cases: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary = {}
    for dimension in (*FACT_LABELS, "unsupported_claims", "information_gaps"):
        labels = [
            label for case in cases for label in case["comparisons"][dimension].values()
        ]
        counts = {
            status: sum(label["status"] == status for label in labels)
            for status in ("matches", "disagreements", "invalid_or_missing")
        }
        summary[dimension] = {
            "applicable": len(labels),
            "valid": counts["matches"] + counts["disagreements"],
            **counts,
        }
    return summary


def compare_reference_judges(
    baseline: Path,
    destination: Path,
    *,
    models: dict[str, str],
    experiment_id: str,
    ollama_port: int = 11434,
) -> None:
    """Run the same rubric over saved cases, then compare with human labels."""
    if baseline.resolve() == destination.resolve():
        raise ValueError("The judge report needs a separate path from the human export")
    if not models or not 1 <= ollama_port <= 65535:
        raise ValueError("Provide a local model and a valid Ollama port")
    content = baseline.read_bytes()
    exported = json.loads(content)
    cases = exported["cases"]
    if (
        not exported["review_complete"]
        or not cases
        or len({case["case_id"] for case in cases}) != len(cases)
        or any(
            case["validation_errors"] or case["review_status"] != "complete"
            for case in cases
        )
    ):
        raise ValueError("A validated and complete human review export is required")
    resume_sha = hashlib.sha256(RESUME.read_bytes()).hexdigest()
    if any(case["configuration"]["resume_sha256"] != resume_sha for case in cases):
        raise ValueError("The baseline resume does not match the local demo PDF")
    resume = "\n\n".join(
        f"Page {number}: {page.extract_text()}"
        for number, page in enumerate(PdfReader(RESUME).pages, 1)
    )
    rows = _dataset(cases, resume)
    baseline_sha = hashlib.sha256(content).hexdigest()
    endpoint = f"http://127.0.0.1:{ollama_port}/v1/chat/completions"
    prompts = {"retrieval_judge": RETRIEVAL_PROMPT, "answer_judge": ANSWER_PROMPT}
    report: dict[str, Any] = {
        "baseline_sha256": baseline_sha,
        "mlflow_version": mlflow.__version__,
        "experiment_id": experiment_id,
        "endpoint": endpoint,
        "inference_params": INFERENCE_PARAMS,
        "prompts": prompts,
        "judges": [],
    }
    for model, digest in models.items():
        model_uri = f"ollama:/{model}"
        scorers: list[Scorer] = [
            mlflow.genai.make_judge(
                name=name,
                instructions=prompt,
                model=model_uri,
                base_url=endpoint,
                feedback_value_type=list[str],
                inference_params=INFERENCE_PARAMS,
            )
            for name, prompt in prompts.items()
        ]
        with mlflow.start_run(experiment_id=experiment_id, run_name=model) as run:
            mlflow.log_params(
                {
                    "model": model_uri,
                    "model_digest": digest,
                    "ollama_endpoint": endpoint,
                    "human_baseline_sha256": baseline_sha,
                    **INFERENCE_PARAMS,
                }
            )
            mlflow.log_dict(prompts, "judge_prompts.json")
            result = mlflow.genai.evaluate(data=rows, scorers=scorers)
            evaluations = (
                {
                    row["request"]["case_id"]: row
                    for row in result.result_df.to_dict("records")
                }
                if result.result_df is not None
                else {}
            )
            comparisons = [
                _compare_case(case, evaluations.get(case["case_id"], {}))
                for case in cases
            ]
            summary = _summarize(comparisons)
            mlflow.log_metrics(
                {
                    f"human_agreement.{dimension}.{key}": value
                    for dimension, counts in summary.items()
                    for key, value in counts.items()
                }
            )
            report["judges"].append(
                {
                    "model": model_uri,
                    "model_digest": digest,
                    "run_id": run.info.run_id,
                    "summary": summary,
                    "cases": comparisons,
                }
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--mlflow-port", type=int, default=5000)
    args = parser.parse_args()
    if not all(1 <= port <= 65535 for port in (args.ollama_port, args.mlflow_port)):
        parser.error("Ports must be between 1 and 65535")

    os.environ.update(
        {
            "MLFLOW_DISABLE_TELEMETRY": "true",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "NO_PROXY": "127.0.0.1,localhost",
            "MLFLOW_GENAI_EVAL_MAX_WORKERS": "1",
            "MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS": "120",
        }
    )
    from ollama import Client

    endpoint = f"http://127.0.0.1:{args.ollama_port}"
    installed = {
        model.model: model.digest
        for model in Client(host=endpoint, timeout=10, trust_env=False).list().models
    }
    models = {}
    for name in args.model:
        if not (digest := installed.get(name)):
            parser.error(f"Model {name} is not installed in local Ollama")
        models[name] = digest
    mlflow.set_tracking_uri(f"http://127.0.0.1:{args.mlflow_port}")
    experiment = mlflow.set_experiment(JUDGE_EXPERIMENT)
    compare_reference_judges(
        args.baseline,
        args.destination,
        models=models,
        experiment_id=experiment.experiment_id,
        ollama_port=args.ollama_port,
    )
    print(f"Saved {args.destination}")


if __name__ == "__main__":
    main()
