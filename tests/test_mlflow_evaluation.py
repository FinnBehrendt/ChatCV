import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import mlflow
import pytest
import requests
from mlflow import MlflowClient
from mlflow.entities import AssessmentSource, AssessmentSourceType

from experiments.mlflow_evaluation import export_reviewed_case, seed_fit_check
from experiments.reference_review import (
    create_case_review,
    create_dropdown_review,
    create_uniform_review,
    export_case_reviews,
)
from experiments.run_reference_case import REFERENCE_CASES, run_reference_case


@pytest.fixture
def local_mlflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MlflowClient]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'tracking.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    mlflow.tracing.reset()
    try:
        yield MlflowClient()
    finally:
        mlflow.tracing.reset()


def test_judge_comparison_scores_saved_cases_without_exposing_human_labels(
    tmp_path: Path, local_mlflow: MlflowClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.compare_reference_judges import compare_reference_judges

    resume = Path(__file__).resolve().parents[1] / "data" / "CV_Demo.pdf"
    baseline = tmp_path / "human.json"
    baseline.write_text(
        json.dumps(
            {
                "review_complete": True,
                "cases": [
                    {
                        "case_id": "partial-test",
                        "trace_id": "saved-trace",
                        "question": "Which language, and for how long?",
                        "answerability": "partial",
                        "answer": "Python. The duration is unspecified.",
                        "evidence": ["Page 1: Developed Python tools."],
                        "expected_facts": ["The candidate used Python."],
                        "expected_gap": "The resume does not state a duration.",
                        "configuration": {
                            "resume_sha256": hashlib.sha256(
                                resume.read_bytes()
                            ).hexdigest()
                        },
                        "normalized_review": {
                            "retrieval_by_fact": {"F1": "found"},
                            "answer_by_fact": {"F1": "correct"},
                            "unsupported_claims": "none",
                            "information_gaps": "appropriate",
                        },
                        "reviews": [{"rationale": "Private reviewer comment"}],
                        "review_status": "complete",
                        "validation_errors": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    exported = json.loads(baseline.read_text(encoding="utf-8"))
    partial = exported["cases"][0]
    full = {
        **partial,
        "case_id": "full-test",
        "trace_id": "saved-full-trace",
        "answerability": "full",
        "normalized_review": {
            **partial["normalized_review"],
            "information_gaps": "n/a",
        },
    }
    full.pop("expected_gap")
    exported["cases"].append(full)
    baseline.write_text(json.dumps(exported), encoding="utf-8")

    def complete(**kwargs: object) -> requests.Response:
        prompt = str(kwargs["json"])
        assert "Private reviewer comment" not in prompt
        assert "normalized_review" not in prompt
        assert kwargs["url"] == "http://127.0.0.1:11434/v1/chat/completions"
        if "Evaluate retrieval" in prompt:
            assert "Developed Python tools." in prompt
            assert "complete_resume" not in prompt
            assert "Python. The duration is unspecified." not in prompt
            labels = ["found"]
        else:
            assert "Python. The duration is unspecified." in prompt
            labels = ["none", "n/a", "missing"]
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            {
                "id": "test-ollama-response",
                "object": "chat.completion",
                "created": 0,
                "model": "test-judge:latest",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"result": labels, "rationale": "Review rationale."}
                            ),
                        },
                    }
                ],
            }
        ).encode()
        return response

    monkeypatch.setattr(requests, "post", complete)
    experiment_id = local_mlflow.create_experiment("Judge test")
    destination = tmp_path / "comparison.json"
    compare_reference_judges(
        baseline,
        destination,
        models={"test-judge:latest": "local-digest"},
        experiment_id=experiment_id,
    )
    result = json.loads(destination.read_text(encoding="utf-8"))
    judge = result["judges"][0]
    assert judge["cases"][0]["trace_id"] == "saved-trace"
    assert judge["summary"]["retrieval_by_fact"]["matches"] == 2
    assert judge["summary"]["answer_by_fact"]["disagreements"] == 2
    assert judge["summary"]["information_gaps"]["invalid_or_missing"] == 1
    assert local_mlflow.get_run(judge["run_id"]).data.params["model_digest"] == (
        "local-digest"
    )


def test_fit_check_creates_three_reference_cases_once(
    local_mlflow: MlflowClient,
) -> None:
    experiment_id = local_mlflow.create_experiment("ChatCV test")

    seed_fit_check(local_mlflow, experiment_id)
    seed_fit_check(local_mlflow, experiment_id)

    traces = local_mlflow.search_traces(locations=[experiment_id])
    roots = [trace.data.spans[0] for trace in traces]
    assert len(roots) == 3
    assert {root.inputs["answerability"] for root in roots} == {
        "full",
        "partial",
        "none",
    }
    assert all(root.outputs["answer"] for root in roots)
    partial = next(
        trace for trace in traces if trace.data.spans[0].inputs["case_id"] == "demo-2"
    )
    assert "F1. The candidate used Python." in partial.info.request_preview
    assert "Retrieved sources" in partial.info.response_preview
    assert "Page 1: Developed Python tools for data processing." in (
        partial.info.response_preview
    )


def test_export_preserves_reviewed_reference_case(
    tmp_path: Path, local_mlflow: MlflowClient
) -> None:
    with mlflow.start_span("reference-question") as span:
        span.set_inputs(
            {
                "case_id": "demo-1",
                "answerability": "full",
                "question": "Which language did the candidate use?",
                "expected_facts": ["The candidate used Python."],
                "expected_evidence": ["Page 1: Developed Python tools."],
            }
        )
        span.set_outputs(
            {"answer": "Python.", "evidence": ["Page 1: Developed Python tools."]}
        )
    mlflow.log_feedback(
        trace_id=span.trace_id,
        name="fact_review",
        value="correct",
        rationale="The resume explicitly mentions Python.",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
    )
    trace = local_mlflow.get_trace(span.trace_id)

    destination = tmp_path / "reviewed-case.json"
    export_reviewed_case(trace, destination)

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "case_id": "demo-1",
        "answerability": "full",
        "question": "Which language did the candidate use?",
        "expected_facts": ["The candidate used Python."],
        "expected_evidence": ["Page 1: Developed Python tools."],
        "answer": "Python.",
        "evidence": ["Page 1: Developed Python tools."],
        "reviews": [
            {
                "name": "fact_review",
                "value": "correct",
                "rationale": "The resume explicitly mentions Python.",
            }
        ],
    }


def test_chatcv_answer_and_sources_reach_the_same_mlflow_case(
    tmp_path: Path, local_mlflow: MlflowClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "false")
    from app.config import AppConfig, DataPaths, EmbeddingConfig, LLMConfig
    from app.modules import model_provider

    resume = Path(__file__).resolve().parents[1] / "data" / "CV_Demo.pdf"
    empty_supplement = tmp_path / "about.txt"
    empty_supplement.touch()
    settings = AppConfig(
        _env_file=None,
        llm=LLMConfig(
            provider="openai_compatible",
            model="test-chat",
            base_url="http://model.test/v1",
        ),
        embedding=EmbeddingConfig(
            provider="openai_compatible",
            model="test-embedding",
            base_url="http://model.test/v1",
        ),
        data=DataPaths(
            cv_path=str(resume),
            about_me_path=str(empty_supplement),
            vector_db_path=str(tmp_path / "index"),
        ),
        chat_system_prompt="Use the resume to answer about {candidate_name}. {{context}}",
    )
    answer = "Deployment time fell by 75 percent."

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if request.url.path == "/v1/embeddings":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"index": i, "embedding": [1.0, 0.0], "object": "embedding"}
                        for i, _ in enumerate(payload["input"])
                    ],
                },
            )
        assert request.url.path == "/v1/chat/completions"
        assert [message["role"] for message in payload["messages"]] == [
            "system",
            "user",
        ]
        assert "reducing deployment time by 75%" in payload["messages"][0]["content"]
        return httpx.Response(
            200,
            json={
                "id": "test-answer",
                "object": "chat.completion",
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    chat_model = model_provider.ChatOpenAI
    embeddings = model_provider.OpenAIEmbeddings
    experiment_id = local_mlflow.create_experiment("Real reference case test")
    with httpx.Client(transport=httpx.MockTransport(respond)) as model_client:
        monkeypatch.setattr(
            model_provider,
            "ChatOpenAI",
            lambda **kw: chat_model(http_client=model_client, **kw),
        )
        monkeypatch.setattr(
            model_provider,
            "OpenAIEmbeddings",
            lambda **kw: embeddings(http_client=model_client, **kw),
        )
        trace_id = run_reference_case(settings, experiment_id)

    trace = local_mlflow.get_trace(trace_id)
    root = trace.data.spans[0]
    assert root.inputs["case_id"] == "cv-deployment-time-01"
    assert root.outputs["answer"] == answer
    assert any(
        passage.startswith("Page 2:") and "reducing deployment time by 75%" in passage
        for passage in root.outputs["evidence"]
    )
    assert root.inputs["configuration"]["embedding"]["model"] == "test-embedding"
    assert answer in (trace.info.response_preview or "")


def test_review_queue_and_export_preserve_and_validate_human_judgments(
    local_mlflow: MlflowClient, tmp_path: Path
) -> None:
    from mlflow.entities import MlflowExperimentLocation
    from mlflow.genai.label_schemas import InputCategorical, list_label_schemas
    from mlflow.genai.review_queues import (
        list_review_queue_items,
        list_review_queues,
        set_review_queue_item_status,
    )

    experiment_id = local_mlflow.create_experiment("Reference review test")
    trace_ids = []
    for case in REFERENCE_CASES[:3]:
        with mlflow.start_span(
            case["case_id"],
            trace_destination=MlflowExperimentLocation(experiment_id=experiment_id),
        ) as span:
            inputs = dict(case)
            if case["case_id"] == "cv-deployment-time-01":
                inputs["expected_facts"] = [
                    *case["expected_facts"],
                    "A second fact.",
                    "A third fact.",
                ]
            span.set_inputs(inputs)
            span.set_outputs(
                {"answer": "Controlled answer.", "evidence": ["Page 2: Evidence."]}
            )
            trace_ids.append(span.trace_id)

    queue_id = create_case_review(experiment_id, trace_ids)
    assert create_case_review(experiment_id, trace_ids) == queue_id
    assert {item.item_id for item in list_review_queue_items(queue_id)} == set(
        trace_ids
    )
    queue = next(
        queue
        for queue in list_review_queues(experiment_id=experiment_id)
        if queue.queue_id == queue_id
    )
    assert {
        field.name
        for field in list_label_schemas(experiment_id)
        if field.schema_id in queue.schema_ids
    } == {
        "retrieval_by_fact",
        "answer_by_fact",
        "unsupported_claims",
        "information_gaps",
    }

    source = AssessmentSource(source_type=AssessmentSourceType.HUMAN)
    valid = {
        "retrieval_by_fact": " F1= Found F2 = found, F3=found ",
        "answer_by_fact": "F1=correct F2=correct, F3=correct",
        "unsupported_claims": "none",
        "information_gaps": "n/a",
    }
    bad = {
        "retrieval_by_fact": "F2=found",
        "answer_by_fact": "F1=plausible",
        "information_gaps": "n/a",
    }
    inapplicable = {
        "retrieval_by_fact": "F1=found",
        "answer_by_fact": "n/a",
        "unsupported_claims": "none",
        "information_gaps": "appropriate",
    }
    for trace_id, labels in zip(trace_ids, [valid, bad, inapplicable], strict=True):
        for name, value in labels.items():
            mlflow.log_feedback(
                trace_id=trace_id, name=name, value=value, source=source
            )

    destination = tmp_path / "reference-review.json"
    expected_ids = {case["case_id"] for case in REFERENCE_CASES}
    assert (
        export_case_reviews(
            local_mlflow, experiment_id, destination, expected_case_ids=expected_ids
        )
        is False
    )
    export = json.loads(destination.read_text(encoding="utf-8"))
    records = {case["case_id"]: case for case in export["cases"]}
    assert records.keys() == {case["case_id"] for case in REFERENCE_CASES[:3]}
    assert set(export["missing_case_ids"]) == expected_ids - records.keys()
    good_case = records["cv-deployment-time-01"]
    bad_case = records["cv-deployment-rollout-02"]
    gap_case = records["cv-masters-university-03"]
    assert good_case["answer"] == "Controlled answer."
    assert good_case["evidence"] == ["Page 2: Evidence."]
    assert {review["name"]: review["value"] for review in good_case["reviews"]} == valid
    assert not good_case["validation_errors"]
    assert {error["field"] for error in bad_case["validation_errors"]} == set(valid)
    assert [error["field"] for error in gap_case["validation_errors"]] == [
        "retrieval_by_fact"
    ]

    dropdown_queue = create_dropdown_review(
        experiment_id, "Retrieval corrections", [trace_ids[0]], [1, 2]
    )
    assert [item.item_id for item in list_review_queue_items(dropdown_queue)] == [
        trace_ids[0]
    ]
    mlflow.log_feedback(
        trace_id=trace_ids[0], name="retrieval_f1", value="missing", source=source
    )
    export_case_reviews(
        local_mlflow, experiment_id, destination, expected_case_ids=expected_ids
    )
    corrected = next(
        case
        for case in json.loads(destination.read_text(encoding="utf-8"))["cases"]
        if case["case_id"] == good_case["case_id"]
    )
    assert corrected["normalized_review"]["retrieval_by_fact"] == {
        "F1": "missing",
        "F2": "found",
        "F3": "found",
    }
    assert any(
        review["name"] == "retrieval_by_fact"
        and review["value"] == valid["retrieval_by_fact"]
        for review in corrected["reviews"]
    )

    for trace_id in trace_ids:
        set_review_queue_item_status(
            queue_id, item_id=trace_id, status="complete", completed_by="default"
        )
    uniform_id = create_uniform_review(local_mlflow, experiment_id, trace_ids)
    assert create_uniform_review(local_mlflow, experiment_id, trace_ids) == uniform_id
    assert {
        item.item_id for item in list_review_queue_items(uniform_id, status="pending")
    } == {trace_ids[1]}
    uniform = next(
        queue
        for queue in list_review_queues(experiment_id=experiment_id)
        if queue.queue_id == uniform_id
    )
    dropdowns = [
        field
        for field in list_label_schemas(experiment_id)
        if field.schema_id in uniform.schema_ids
    ]
    assert len(dropdowns) == 8 and all(
        isinstance(field.input, InputCategorical) for field in dropdowns
    )

    export_case_reviews(
        local_mlflow, experiment_id, destination, expected_case_ids=expected_ids
    )
    records = {
        case["case_id"]: case
        for case in json.loads(destination.read_text(encoding="utf-8"))["cases"]
    }
    migrated = records[good_case["case_id"]]
    assert migrated["normalized_review"]["retrieval_by_fact"] == {
        "F1": "missing",
        "F2": "found",
        "F3": "found",
    }
    assert migrated["normalized_review"]["answer_by_fact"] == {
        "F1": "correct",
        "F2": "correct",
        "F3": "correct",
    }
    assert records[gap_case["case_id"]]["unused_fact_slots"] == {
        "F1": "n/a",
        "F2": "n/a",
        "F3": "n/a",
    }
    assert not migrated["validation_errors"]
