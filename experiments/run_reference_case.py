"""Run ChatCV reference questions and review the results in local MLflow."""

import argparse
import hashlib
import json
import os
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import mlflow
import yaml
from mlflow import MlflowClient
from mlflow.entities import MlflowExperimentLocation, TraceState

from experiments.reference_review import (
    DROPDOWN_QUEUE_NAME,
    create_uniform_review,
    export_case_reviews,
)

if TYPE_CHECKING:
    from app.config import AppConfig

EXPERIMENT_NAME = "ChatCV - Demo Resume Baseline"
CASE_TAG = "chatcv.reference_case_id"


class ReferenceCase(TypedDict):
    case_id: str
    answerability: Literal["full", "partial", "none"]
    question: str
    expected_facts: list[str]
    expected_evidence: list[str]
    expected_gap: str | None


REFERENCE_CASES: list[ReferenceCase] = json.loads(
    Path(__file__).with_name("reference_cases.json").read_text(encoding="utf-8")
)


def _configuration(settings: "AppConfig") -> dict[str, Any]:
    return {
        "resume_sha256": hashlib.sha256(
            Path(settings.data.cv_path).read_bytes()
        ).hexdigest(),
        "candidate": settings.candidate.name,
        "chat_system_prompt": (settings.model_extra or {}).get("chat_system_prompt"),
        "chat": settings.llm.model_dump(exclude={"base_url"}),
        "embedding": settings.embedding.model_dump(exclude={"base_url"}),
        "ollama": settings.ollama.model_dump(exclude={"models"}),
    }


def run_reference_case(
    settings: "AppConfig", experiment_id: str, case: ReferenceCase | None = None
) -> str:
    """Capture the generated answer and the exact passages returned by ChatCV."""
    from app.modules.rag_pipeline import ChatRAGPipeline

    case = REFERENCE_CASES[0] if case is None else case
    case_id = case["case_id"]
    question = case["question"]
    facts = (
        "\n\n".join(
            f"F{i}. {fact}" for i, fact in enumerate(case["expected_facts"], start=1)
        )
        or "None."
    )
    request = f"**Question**\n\n{question}\n\n**Expected facts**\n\n{facts}"
    if case["expected_gap"]:
        request += f"\n\n**Information gap**\n\n{case['expected_gap']}"
    else:
        request += "\n\n**Information gaps**\n\nFully answerable. Select n/a."
    with mlflow.start_span(
        name=case_id,
        trace_destination=MlflowExperimentLocation(experiment_id=experiment_id),
    ) as span:
        mlflow.update_current_trace(
            tags={
                CASE_TAG: case_id,
                "chatcv.result_kind": "generated",
            },
            request_preview=request,
        )
        try:
            span.set_inputs({**case, "configuration": _configuration(settings)})
            result = ChatRAGPipeline(settings).get_completion(question)
            answer = result["answer"]["answer"]
            if not result["completed"]:
                raise RuntimeError(answer)
            evidence = []
            for source in result["sources"]:
                page = source.metadata.get("page")
                label = f"Page {page + 1}" if isinstance(page, int) else "Resume"
                evidence.append(f"{label}: {source.page_content}")
        except Exception as error:
            span.set_outputs({"answer": None, "evidence": [], "error": str(error)})
            mlflow.update_current_trace(
                response_preview=f"**Execution failed**\n\n{error}"
            )
            raise

        span.set_outputs({"answer": answer, "evidence": evidence})
        sources = (
            "\n\n".join(f"- {passage}" for passage in evidence)
            or "No passages retrieved."
        )
        mlflow.update_current_trace(
            response_preview=f"**Answer**\n\n{answer}\n\n**Retrieved sources**\n\n{sources}",
        )
        return span.trace_id


def _prepare_reviews(
    client: MlflowClient, experiment_id: str, settings: "AppConfig"
) -> None:
    expected = {case["case_id"] for case in REFERENCE_CASES}
    configuration = _configuration(settings)
    captured = {}
    for trace in client.search_traces(locations=[experiment_id]):
        case_id = trace.info.tags.get(CASE_TAG)
        if case_id not in expected or trace.info.state != TraceState.OK:
            continue
        root = trace.data.spans[0]
        case = next(case for case in REFERENCE_CASES if case["case_id"] == case_id)
        if (
            case_id in captured
            or root.inputs["configuration"] != configuration
            or any(root.inputs.get(key) != value for key, value in case.items())
        ):
            raise ValueError(
                f"Existing traces for {case_id} differ from this run. Choose a new --experiment."
            )
        captured[case_id] = trace.trace_id

    trace_ids = []
    for case in REFERENCE_CASES:
        trace_id = captured.get(case["case_id"])
        if trace_id is None:
            trace_id = run_reference_case(settings, experiment_id, case)
        print(f"{case['case_id']}: {trace_id}")
        trace_ids.append(trace_id)
    mlflow.flush_trace_async_logging()
    create_uniform_review(client, experiment_id, trace_ids)
    print(f"Open Review → {DROPDOWN_QUEUE_NAME}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat-model", help="Installed local Ollama chat model")
    parser.add_argument(
        "--embedding-model",
        help="Installed local Ollama embedding model",
    )
    parser.add_argument("--mlflow-port", type=int, default=5000)
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--prepare-review",
        action="store_true",
        help="Record missing cases and create the review queue",
    )
    action.add_argument(
        "--export-review",
        type=Path,
        help="Write the queue and its human judgments to a private JSON file",
    )
    args = parser.parse_args()
    if not all(1 <= port <= 65535 for port in (args.mlflow_port, args.ollama_port)):
        parser.error("Ports must be between 1 and 65535")
    if args.export_review is None and not (args.chat_model and args.embedding_model):
        parser.error(
            "--chat-model and --embedding-model are required to run reference cases"
        )

    os.environ.update(
        {
            "MLFLOW_DISABLE_TELEMETRY": "true",
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
            "PYTHON_DOTENV_DISABLED": "true",
            "ANONYMIZED_TELEMETRY": "false",
            "NO_PROXY": "127.0.0.1,localhost",
            "LOG_DIR": "logs",
        }
    )
    tracking_uri = f"http://127.0.0.1:{args.mlflow_port}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    if args.export_review is not None:
        experiment = client.get_experiment_by_name(args.experiment)
        if experiment is None:
            parser.error("The local review experiment does not exist")
        complete = export_case_reviews(
            client,
            experiment.experiment_id,
            args.export_review,
            expected_case_ids={case["case_id"] for case in REFERENCE_CASES},
        )
        print(f"Saved {args.export_review}")
        if not complete:
            raise SystemExit(
                "Review is incomplete or has validation errors. See the export."
            )
        return

    repo = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((repo / "config" / "base.yml").read_text(encoding="utf-8"))
    with TemporaryDirectory(prefix="chatcv-reference-case-") as temporary, chdir(
        temporary
    ):
        from app.config import (
            AppConfig,
            CandidateConfig,
            DataPaths,
            EmbeddingConfig,
            LLMConfig,
            OllamaConfig,
        )

        work = Path(temporary)
        supplement = work / "empty-about-me.txt"
        supplement.touch()
        settings = AppConfig(
            _env_file=None,
            llm=LLMConfig(
                provider="ollama",
                model=args.chat_model,
                temperature=base["llm"]["temperature"],
            ),
            embedding=EmbeddingConfig(provider="ollama", model=args.embedding_model),
            ollama=OllamaConfig(endpoint=f"http://127.0.0.1:{args.ollama_port}"),
            candidate=CandidateConfig(name="John Doe"),
            data=DataPaths(
                cv_path=str(repo / "data" / "CV_Demo.pdf"),
                about_me_path=str(supplement),
                vector_db_path=str(work / "index"),
            ),
            chat_system_prompt=base["chat_system_prompt"],
            chat_fallback_response=base["chat_fallback_response"],
        )
        experiment = mlflow.set_experiment(args.experiment)
        print(f"{tracking_uri}/#/experiments/{experiment.experiment_id}")
        try:
            if args.prepare_review:
                _prepare_reviews(client, experiment.experiment_id, settings)
            else:
                trace_id = run_reference_case(settings, experiment.experiment_id)
                print(f"{REFERENCE_CASES[0]['case_id']}: {trace_id}")
        finally:
            mlflow.flush_trace_async_logging()


if __name__ == "__main__":
    main()
