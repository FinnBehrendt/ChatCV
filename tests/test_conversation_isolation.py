"""Conversation behavior through HTTP, without external models or private data."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pytest import MonkeyPatch
from pytest_mock import MockerFixture

from app.api import routes
from app.auth import auth
from app.config import AppConfig, DataPaths, EmbeddingConfig, LLMConfig
from app.modules import guardrails, model_provider, rag_pipeline
from app.utils import analytics


def respond_as_model(request: httpx.Request) -> httpx.Response:
    """Report topic markers in model input; do not emulate session management."""
    body = json.loads(request.content)
    if request.url.path == "/v1/embeddings":
        inputs = body["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": body["model"],
                "data": [
                    {"object": "embedding", "index": index, "embedding": [1.0, 0.0]}
                    for index in range(len(inputs))
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )
    if request.url.path != "/v1/chat/completions":
        raise AssertionError(f"Unexpected model request: {request.url.path}")

    messages = body["messages"]
    if not any(message["role"] == "system" for message in messages):
        # History-aware retrieval rewrites the query before the answer request.
        answer = "candidate professional skills"
    else:
        user_text = " ".join(
            message["content"] for message in messages if message["role"] == "user"
        )
        markers = [marker for marker in ("ALPHA", "BETA") if marker in user_text]
        answer = "Topics: " + ", ".join(markers)
    return httpx.Response(
        200,
        json={
            "id": "synthetic-completion",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
        },
    )


@pytest.fixture
def visitors(
    monkeypatch: MonkeyPatch, mocker: MockerFixture, tmp_path: Path
) -> Iterator[tuple[TestClient, TestClient]]:
    """Keep real routing, RAG, document loading and a disposable Chroma index."""
    monkeypatch.setenv("INVITE_CODES", "{}")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "false")
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(tmp_path / "resume.pdf")
    (tmp_path / "about.md").write_text("Example Candidate develops Python software.")
    settings = AppConfig(
        _env_file=None,
        LANGSMITH_API_KEY=None,
        OPENAI_API_KEY=None,
        LLM_API_KEY=None,
        EMBEDDING_API_KEY=None,
        llm=LLMConfig(
            provider="openai_compatible",
            model="synthetic-chat",
            base_url="http://model.test/v1",
        ),
        embedding=EmbeddingConfig(
            provider="openai_compatible",
            model="synthetic-embedding",
            base_url="http://model.test/v1",
        ),
        data=DataPaths(
            cv_path=str(tmp_path / "resume.pdf"),
            about_me_path=str(tmp_path / "about.md"),
            vector_db_path=str(tmp_path / "index"),
            analytics_log_path=str(tmp_path / "analytics.log"),
        ),
        chat_system_prompt="Answer about {candidate_name}. Context: {{context}}",
    )
    settings.candidate.name = "Example Candidate"
    monkeypatch.setattr(rag_pipeline, "config", settings)
    monkeypatch.setattr(analytics, "config", settings)
    monkeypatch.setattr(auth, "config", settings)
    monkeypatch.setattr(auth, "INVITE_CODES", {})
    monkeypatch.setattr(auth, "active_sessions", {})
    monkeypatch.setattr(rag_pipeline.ChatRAGPipeline, "_instance", None)
    monkeypatch.setattr(routes, "advanced_analytics", analytics.AdvancedAnalytics())

    class ValidationAdapter:
        """No external Guardrails models; normal local input checks still run."""

        def validate(self, text: str) -> str:
            return text

    monkeypatch.setattr(guardrails, "GUARDRAILS_AVAILABLE", False)
    monkeypatch.setattr(guardrails, "Guard", ValidationAdapter)
    chat_client = model_provider.ChatOpenAI
    embedding_client = model_provider.OpenAIEmbeddings
    with httpx.Client(transport=httpx.MockTransport(respond_as_model)) as client:
        mocker.patch.object(
            model_provider,
            "ChatOpenAI",
            side_effect=lambda **options: chat_client(http_client=client, **options),
        )
        mocker.patch.object(
            model_provider,
            "OpenAIEmbeddings",
            side_effect=lambda **options: embedding_client(
                http_client=client, **options
            ),
        )
        application = FastAPI()
        application.include_router(routes.router)
        with (
            TestClient(application, base_url="http://localhost") as visitor_a,
            TestClient(application, base_url="http://localhost") as visitor_b,
        ):
            yield visitor_a, visitor_b


def test_anonymous_visitors_keep_independent_follow_up_context(
    visitors: tuple[TestClient, TestClient],
) -> None:
    visitor_a, visitor_b = visitors
    answers = []
    for visitor, question in (
        (visitor_a, "Please remember my professional topic ALPHA."),
        (visitor_b, "Please remember my professional topic BETA."),
        (visitor_a, "Which professional topic did I mention earlier?"),
        (visitor_b, "Which professional topic did I mention earlier?"),
    ):
        response = visitor.post("/chat", json={"query": question})
        assert response.status_code == 200
        data: dict[str, Any] = response.json()
        answers.append(data["answer"]["answer"])

    assert answers == ["Topics: ALPHA", "Topics: BETA", "Topics: ALPHA", "Topics: BETA"]
