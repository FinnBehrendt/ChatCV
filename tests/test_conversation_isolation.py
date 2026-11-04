"""Conversation behavior through HTTP, without external models or private data."""

import asyncio
import json
import re
import socket
from collections.abc import Awaitable, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread
from typing import Any

import anyio.to_thread
import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.testclient import TestClient
from playwright.sync_api import Page, expect, sync_playwright
from pypdf import PdfWriter
from pytest import FixtureRequest, MonkeyPatch
from pytest_mock import MockerFixture

import main
from app.api import routes
from app.auth import auth
from app.config import AppConfig, DataPaths, EmbeddingConfig, LLMConfig
from app.modules import (
    conversations,
    guardrails,
    model_provider,
    rag_pipeline,
    summary_pipeline,
)
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
    if any(
        message["content"].startswith("Summarize candidate documents for")
        for message in messages
    ):
        answer = "Example Candidate develops Python software."
    elif not any(message["role"] == "system" for message in messages):
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


class ModelServer:
    """Control an external answer's timing without replacing application logic."""

    def __init__(self) -> None:
        self.pause_next_answer = False
        self.answer_requests = 0
        self.fail_answers = False
        self.answer_override: str | None = None
        self.summary_inputs: list[list[dict[str, Any]]] = []
        self.fail_summaries = False
        self.started = Event()
        self.release = Event()

    def respond(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            messages = json.loads(request.content)["messages"]
            if any(
                message["content"].startswith("Summarize candidate documents for")
                for message in messages
            ):
                self.summary_inputs.append(messages)
                if self.fail_summaries:
                    return httpx.Response(
                        400, json={"error": {"message": "Synthetic summary failure"}}
                    )
        is_answer = request.url.path == "/v1/chat/completions" and any(
            message["role"] == "system"
            for message in json.loads(request.content)["messages"]
        )
        if is_answer:
            self.answer_requests += 1
            if self.fail_answers:
                return httpx.Response(
                    400, json={"error": {"message": "Synthetic model failure"}}
                )
        if self.pause_next_answer and request.url.path == "/v1/chat/completions":
            messages = json.loads(request.content)["messages"]
            if any(message["role"] == "system" for message in messages):
                self.pause_next_answer = False
                self.started.set()
                if not self.release.wait(timeout=10):
                    raise AssertionError("Test did not release the model response")
        response = respond_as_model(request)
        if is_answer and self.answer_override is not None:
            body = response.json()
            body["choices"][0]["message"]["content"] = self.answer_override
            return httpx.Response(200, json=body)
        return response


@pytest.fixture
def model_server() -> ModelServer:
    return ModelServer()


class Clock:
    """Advance only the application's elapsed-time source, without real sleeps."""

    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


@pytest.fixture
def clock(monkeypatch: MonkeyPatch) -> Clock:
    source = Clock()
    monkeypatch.setattr(conversations, "monotonic", source)
    return source


class PreparationGate:
    """Delay one incoming bootstrap request before the real app handles it."""

    def __init__(self) -> None:
        self.pause_next = False
        self.started = Event()
        self.release = Event()


@pytest.fixture
def preparation_gate() -> PreparationGate:
    return PreparationGate()


@pytest.fixture
def worker_gate(monkeypatch: MonkeyPatch) -> PreparationGate:
    """Pause external threadpool scheduling, not our generation/state logic."""
    gate = PreparationGate()
    original = anyio.to_thread.run_sync

    async def schedule(function: Any, *args: Any, **kwargs: Any) -> Any:
        target = getattr(function, "func", function)
        if gate.pause_next and getattr(target, "__name__", "") == "get_chat_completion":
            gate.pause_next = False
            gate.started.set()
            assert await asyncio.to_thread(
                gate.release.wait, 10
            ), "Worker was not released"
        return await original(function, *args, **kwargs)

    monkeypatch.setattr(anyio.to_thread, "run_sync", schedule)
    return gate


@pytest.fixture
def conversation_max_turns(request: FixtureRequest) -> int:
    return getattr(request, "param", 20)


@pytest.fixture
def conversation_capacity(request: FixtureRequest) -> int:
    return getattr(request, "param", 100)


@pytest.fixture
def visitors(
    monkeypatch: MonkeyPatch,
    mocker: MockerFixture,
    tmp_path: Path,
    model_server: ModelServer,
    preparation_gate: PreparationGate,
    conversation_max_turns: int,
    conversation_capacity: int,
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
        conversation_max_turns=conversation_max_turns,
        conversation_capacity=conversation_capacity,
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
        summary_prompt_template="Summarize candidate documents for {candidate_name}: {{context}}",
    )
    settings.candidate.name = "Example Candidate"
    monkeypatch.setattr(rag_pipeline, "config", settings)
    monkeypatch.setattr(summary_pipeline, "config", settings)
    monkeypatch.setattr(summary_pipeline.SummaryGenerator, "_instance", None)
    monkeypatch.setattr(analytics, "config", settings)
    monkeypatch.setattr(auth, "config", settings)
    monkeypatch.setattr(auth, "INVITE_CODES", {})
    monkeypatch.setattr(auth, "active_sessions", {})
    monkeypatch.setattr(main, "config", settings)
    monkeypatch.setattr(conversations, "config", settings)
    monkeypatch.setattr(rag_pipeline.ChatRAGPipeline, "_instance", None)
    monkeypatch.setattr(routes, "advanced_analytics", analytics.AdvancedAnalytics())

    class ValidationAdapter:
        """No external Guardrails models; normal local input checks still run."""

        def validate(self, text: str) -> str:
            if "UNSAFE_SYNTHETIC_OUTPUT" in text:
                raise ValueError("Synthetic external content guard rejected output")
            return text

    monkeypatch.setattr(guardrails, "GUARDRAILS_AVAILABLE", False)
    monkeypatch.setattr(guardrails, "Guard", ValidationAdapter)
    chat_client = model_provider.ChatOpenAI
    embedding_client = model_provider.OpenAIEmbeddings
    with httpx.Client(transport=httpx.MockTransport(model_server.respond)) as client:
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

        @application.middleware("http")
        async def delay_one_bootstrap(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            if (
                preparation_gate.pause_next
                and request.method == "POST"
                and request.url.path == "/chat/conversation"
            ):
                preparation_gate.pause_next = False
                preparation_gate.started.set()
                released = await asyncio.to_thread(preparation_gate.release.wait, 10)
                assert released, "Test did not release the bootstrap request"
            return await call_next(request)

        application.include_router(routes.router)
        application.add_api_route("/", main.index, response_class=HTMLResponse)
        with (
            TestClient(application, base_url="http://localhost") as visitor_a,
            TestClient(application, base_url="http://localhost") as visitor_b,
        ):
            yield visitor_a, visitor_b


@pytest.fixture
def browser_page(
    visitors: tuple[TestClient, TestClient],
    request: FixtureRequest,
    monkeypatch: MonkeyPatch,
) -> Iterator[Page]:
    """Serve the real page and API on an ephemeral local port for Chromium."""
    authenticated = getattr(request, "param", False)
    if authenticated:
        monkeypatch.setattr(
            auth,
            "INVITE_CODES",
            {"shared-code": {"active": True, "company": "Example Company"}},
        )
    ready = Event()

    class BrowserTestServer(uvicorn.Server):
        async def startup(self, sockets: list[socket.socket] | None = None) -> None:
            await super().startup(sockets=sockets)
            ready.set()

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = BrowserTestServer(
            uvicorn.Config(visitors[0].app, lifespan="off", log_level="warning")
        )
        thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            assert ready.wait(timeout=5), "Browser test server did not start"
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                try:
                    context = browser.new_context()
                    page = context.new_page()

                    page.context.route(
                        re.compile(r"^https?://(?!127\.0\.0\.1:)"),
                        lambda route: route.abort(),
                    )
                    page.clock.install()
                    page.goto(f"http://127.0.0.1:{port}/")
                    # External Tailwind is not needed to test behavior offline.
                    page.add_style_tag(content=".hidden { display: none !important; }")
                    if authenticated:
                        expect(page.locator("#login-modal")).to_be_visible()
                    else:
                        expect(page.locator("#login-modal")).to_be_hidden()
                    yield page
                finally:
                    browser.close()
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive(), "Browser test server did not stop"


@pytest.mark.parametrize("fail_summary", [False, True])
def test_summary_uses_candidate_documents_without_changing_either_conversation(
    visitors: tuple[TestClient, TestClient],
    model_server: ModelServer,
    fail_summary: bool,
) -> None:
    visitor_a, visitor_b = visitors
    for visitor, topic in ((visitor_a, "ALPHA"), (visitor_b, "BETA")):
        assert (
            visitor.post(
                "/chat",
                json={"query": f"Please remember my professional topic {topic}."},
            ).status_code
            == 200
        )

    model_server.fail_summaries = fail_summary
    result = visitor_a.post("/summary")
    assert result.status_code == 200
    assert result.json()["skills"] == []
    if fail_summary:
        assert (
            result.json()["summary_md"]
            == "Unable to generate summary. Please try again."
        )
        assert "error" in result.json()
    else:
        assert (
            result.json()["summary_md"] == "Example Candidate develops Python software."
        )
        assert result.json()["style"] == "bullet"
    assert model_server.summary_inputs, "No real summary model request was made"
    for messages in model_server.summary_inputs:
        prompt = " ".join(message["content"] for message in messages)
        assert "Example Candidate develops Python software." in prompt
        assert "ALPHA" not in prompt
        assert "BETA" not in prompt

    for visitor, topic in ((visitor_a, "ALPHA"), (visitor_b, "BETA")):
        assert visitor.get("/chat/conversation").json() == {
            "messages": [
                {
                    "role": "user",
                    "content": f"Please remember my professional topic {topic}.",
                },
                {"role": "assistant", "content": f"Topics: {topic}"},
            ]
        }
        follow_up = visitor.post(
            "/chat", json={"query": "Which professional topic did I mention earlier?"}
        )
        assert follow_up.json()["answer"]["answer"] == f"Topics: {topic}"


@pytest.mark.browser
@pytest.mark.parametrize("browser_page", [True], indirect=True)
@pytest.mark.parametrize("transition", ["clear", "logout_login"])
def test_clear_during_bootstrap_does_not_submit_the_old_question(
    browser_page: Page, preparation_gate: PreparationGate, transition: str
) -> None:
    page = browser_page
    page.locator("#invite-code").fill("shared-code")
    page.locator("#login-form button").click()
    expect(page.locator("#login-modal")).to_be_hidden()
    page.locator("#chat-query").fill("Please remember my professional topic ALPHA.")
    page.locator("#submit-btn").click()
    expect(page.locator("#response")).to_contain_text("Assistant: Topics: ALPHA")
    page.clock.fast_forward(3001)
    preparation_gate.pause_next = True
    page.locator("#chat-query").fill("Please remember my professional topic BETA.")
    page.locator("#submit-btn").click()
    try:
        assert preparation_gate.started.wait(timeout=5)
        other = page.context.new_page()
        other.goto(page.url)
        other.add_style_tag(content=".hidden { display: none !important; }")
        expect(other.locator("#response")).to_contain_text("Assistant: Topics: ALPHA")
        if transition == "clear":
            other.locator("#clear-chat").click()
        else:
            other.locator("#logout-btn").click()
            expect(other.locator("#login-modal")).to_be_visible()
            other.locator("#invite-code").fill("shared-code")
            other.locator("#login-form button").click()
            expect(other.locator("#login-modal")).to_be_hidden()
        expect(other.locator("#response")).to_be_empty()
        expect(page.locator("#response")).to_be_empty()
        with page.expect_response(
            lambda response: response.request.method == "POST"
            and response.url.endswith("/chat/conversation")
        ):
            preparation_gate.release.set()
        fresh = "Which professional topic did I mention earlier?"
        page.clock.fast_forward(3001)
        page.locator("#chat-query").fill(fresh)
        page.locator("#submit-btn").click()
        expect(page.locator("#response > div")).to_have_text(
            [f"You: {fresh}", "Assistant: Topics: "]
        )
        assert page.context.request.get(
            f"{page.url.rstrip('/')}/chat/conversation"
        ).json() == {
            "messages": [
                {"role": "user", "content": fresh},
                {"role": "assistant", "content": "Topics: "},
            ]
        }
    finally:
        preparation_gate.release.set()


@pytest.mark.parametrize("transition", ["clear", "login", "logout"])
def test_reset_invalidates_chat_waiting_for_a_worker(
    visitors: tuple[TestClient, TestClient],
    worker_gate: PreparationGate,
    model_server: ModelServer,
    monkeypatch: MonkeyPatch,
    transition: str,
) -> None:
    visitor, _ = visitors
    monkeypatch.setattr(
        auth,
        "INVITE_CODES",
        {
            "shared-code": {"active": True, "company": "Example Company"},
            "other-code": {"active": True, "company": "Another Company"},
        },
    )
    assert (
        visitor.post("/auth/login", json={"invite_code": "shared-code"}).status_code
        == 200
    )
    assert (
        visitor.post(
            "/chat", json={"query": "Please remember my professional topic ALPHA."}
        ).status_code
        == 200
    )
    worker_gate.pause_next = True
    before = model_server.answer_requests
    with ThreadPoolExecutor(max_workers=1) as requests:
        pending = requests.submit(
            visitor.post,
            "/chat",
            json={"query": "Please remember my professional topic BETA."},
        )
        try:
            assert worker_gate.started.wait(timeout=5)
            if transition == "clear":
                assert visitor.delete("/chat/conversation").status_code == 204
            elif transition == "login":
                assert (
                    visitor.post(
                        "/auth/login", json={"invite_code": "other-code"}
                    ).status_code
                    == 200
                )
            else:
                assert visitor.post("/auth/logout").status_code == 200
        finally:
            worker_gate.release.set()
        rejected = pending.result(timeout=5)
    assert rejected.status_code == 409
    assert model_server.answer_requests == before
    if transition == "logout":
        assert (
            visitor.post("/auth/login", json={"invite_code": "shared-code"}).status_code
            == 200
        )
    assert visitor.get("/chat/conversation").json() == {"messages": []}
    fresh = visitor.post(
        "/chat", json={"query": "Which professional topic did I mention earlier?"}
    )
    assert fresh.status_code == 200
    assert fresh.json()["answer"]["answer"] == "Topics: "


def test_a_queued_generation_already_makes_its_conversation_busy(
    visitors: tuple[TestClient, TestClient], worker_gate: PreparationGate
) -> None:
    visitor_a, visitor_b = visitors
    assert visitor_a.post("/chat/conversation").status_code == 204
    worker_gate.pause_next = True
    with ThreadPoolExecutor(max_workers=1) as requests:
        pending = requests.submit(
            visitor_a.post,
            "/chat",
            json={"query": "Please remember my professional topic ALPHA."},
        )
        try:
            assert worker_gate.started.wait(timeout=5)
            refused = visitor_a.post(
                "/chat", json={"query": "Please remember my professional topic BETA."}
            )
            assert refused.status_code == 409
            assert refused.headers["X-Conversation-Status"] == "busy"
            assert (
                visitor_b.post(
                    "/chat", json={"query": "Describe the candidate."}
                ).status_code
                == 200
            )
        finally:
            worker_gate.release.set()
        assert pending.result(timeout=5).status_code == 200
    follow_up = visitor_a.post(
        "/chat", json={"query": "Which professional topic did I mention earlier?"}
    )
    assert follow_up.json()["answer"]["answer"] == "Topics: ALPHA"


def test_cancelling_a_request_before_worker_start_releases_its_reservation(
    visitors: tuple[TestClient, TestClient],
    worker_gate: PreparationGate,
    model_server: ModelServer,
) -> None:
    visitor, _ = visitors
    assert (
        visitor.post(
            "/chat", json={"query": "Please remember my professional topic ALPHA."}
        ).status_code
        == 200
    )
    before = model_server.answer_requests
    worker_gate.pause_next = True

    async def cancel_pending() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=visitor.app),
            base_url="http://localhost",
            cookies=visitor.cookies,
        ) as client:
            pending = asyncio.create_task(
                client.post(
                    "/chat",
                    json={"query": "Please remember my professional topic BETA."},
                )
            )
            try:
                assert await asyncio.to_thread(worker_gate.started.wait, 5)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            finally:
                worker_gate.release.set()

    asyncio.run(cancel_pending())
    assert model_server.answer_requests == before
    follow_up = visitor.post(
        "/chat", json={"query": "Which professional topic did I mention earlier?"}
    )
    assert follow_up.status_code == 200
    assert follow_up.json()["answer"]["answer"] == "Topics: ALPHA"


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


@pytest.mark.parametrize(
    "failure", ["invalid_input", "provider", "rejected_output", "empty_output"]
)
def test_failed_turns_preserve_successful_context_and_allow_a_follow_up(
    visitors: tuple[TestClient, TestClient], model_server: ModelServer, failure: str
) -> None:
    visitor_a, visitor_b = visitors
    for visitor, topic in ((visitor_a, "ALPHA"), (visitor_b, "BETA")):
        assert (
            visitor.post(
                "/chat",
                json={"query": f"Please remember my professional topic {topic}."},
            ).status_code
            == 200
        )
    model_server.fail_answers = failure == "provider"
    if failure == "rejected_output":
        model_server.answer_override = "UNSAFE_SYNTHETIC_OUTPUT"
    elif failure == "empty_output":
        model_server.answer_override = "   "
    question = (
        "Discuss topic BETA and the weather."
        if failure == "invalid_input"
        else "Please remember my professional topic BETA."
    )
    failed = visitor_a.post("/chat", json={"query": question})
    assert failed.status_code == 200  # Preserve the existing error response shape.
    model_server.fail_answers = False
    model_server.answer_override = None

    for visitor, topic in ((visitor_a, "ALPHA"), (visitor_b, "BETA")):
        assert visitor.get("/chat/conversation").json() == {
            "messages": [
                {
                    "role": "user",
                    "content": f"Please remember my professional topic {topic}.",
                },
                {"role": "assistant", "content": f"Topics: {topic}"},
            ]
        }
        follow_up = visitor.post(
            "/chat", json={"query": "Which professional topic did I mention earlier?"}
        )
        assert follow_up.status_code == 200
        assert follow_up.json()["answer"]["answer"] == f"Topics: {topic}"


@pytest.mark.parametrize("configured_limit", [None, 2])
def test_only_latest_completed_turns_remain_in_transcript_and_model_context(
    visitors: tuple[TestClient, TestClient],
    monkeypatch: MonkeyPatch,
    configured_limit: int | None,
) -> None:
    visitor, _ = visitors
    if configured_limit is not None:
        monkeypatch.setattr(main.config, "conversation_max_turns", configured_limit)
    limit = configured_limit if configured_limit is not None else 20
    assert (
        visitor.post(
            "/chat", json={"query": "Please remember my professional topic ALPHA."}
        ).status_code
        == 200
    )
    for index in range(limit):
        assert (
            visitor.post(
                "/chat",
                json={"query": f"Describe the candidate's skills, request {index}."},
            ).status_code
            == 200
        )

    follow_up = "Which professional topic did I mention earlier?"
    response = visitor.post("/chat", json={"query": follow_up})
    assert response.status_code == 200
    assert response.json()["answer"]["answer"] == "Topics: "
    transcript = visitor.get("/chat/conversation").json()["messages"]
    assert len(transcript) == 2 * limit
    assert transcript[0]["content"] == "Describe the candidate's skills, request 1."
    assert transcript[-2:] == [
        {"role": "user", "content": follow_up},
        {"role": "assistant", "content": "Topics: "},
    ]


@pytest.mark.parametrize("configured_minutes", [None, 1])
@pytest.mark.parametrize("submission_path", ["/chat", "/chat/conversation"])
def test_idle_context_expires_despite_passive_reads_and_requires_resubmission(
    visitors: tuple[TestClient, TestClient],
    clock: Clock,
    model_server: ModelServer,
    monkeypatch: MonkeyPatch,
    configured_minutes: int | None,
    submission_path: str,
) -> None:
    visitor_a, visitor_b = visitors
    if configured_minutes is not None:
        monkeypatch.setattr(
            main.config, "conversation_idle_minutes", configured_minutes
        )
    idle_seconds = 60 * (configured_minutes if configured_minutes is not None else 30)
    assert (
        visitor_a.post(
            "/chat", json={"query": "Please remember my professional topic ALPHA."}
        ).status_code
        == 200
    )
    clock.advance(idle_seconds / 2)
    assert len(visitor_a.get("/chat/conversation").json()["messages"]) == 2
    assert visitor_a.post("/chat/conversation").status_code == 204
    assert (
        visitor_b.post(
            "/chat", json={"query": "Please remember my professional topic BETA."}
        ).status_code
        == 200
    )
    clock.advance(idle_seconds / 2)

    expired = visitor_a.get("/chat/conversation")
    assert expired.status_code == 200
    assert expired.json() == {"messages": []}
    assert expired.headers["X-Conversation-Status"] == "expired"
    before = model_server.answer_requests
    rejected = visitor_a.post(
        submission_path,
        json={"query": "Which professional topic did I mention earlier?"},
    )
    assert rejected.status_code == 409
    assert rejected.headers["X-Conversation-Status"] == "expired"
    assert model_server.answer_requests == before
    assert visitor_a.get("/chat/conversation").json() == {"messages": []}

    for visitor, expected in ((visitor_a, "Topics: "), (visitor_b, "Topics: BETA")):
        retry = visitor.post(
            "/chat", json={"query": "Which professional topic did I mention earlier?"}
        )
        assert retry.status_code == 200
        assert retry.json()["answer"]["answer"] == expected


@pytest.mark.parametrize("configured_limit", [None, 2])
@pytest.mark.parametrize("submission_path", ["/chat", "/chat/conversation"])
def test_capacity_rejects_new_visitors_without_evicting_active_context(
    visitors: tuple[TestClient, TestClient],
    monkeypatch: MonkeyPatch,
    model_server: ModelServer,
    configured_limit: int | None,
    submission_path: str,
) -> None:
    visitor_a, newcomer = visitors
    if configured_limit is not None:
        monkeypatch.setattr(main.config, "conversation_capacity", configured_limit)
    limit = configured_limit if configured_limit is not None else 100
    assert (
        visitor_a.post(
            "/chat", json={"query": "Please remember my professional topic ALPHA."}
        ).status_code
        == 200
    )
    for _ in range(limit - 1):
        with TestClient(visitor_a.app, base_url="http://localhost") as other:
            assert other.post("/chat/conversation").status_code == 204

    before = model_server.answer_requests
    refused = newcomer.post(submission_path, json={"query": "Describe the candidate."})
    assert refused.status_code == 503
    assert refused.headers["X-Conversation-Status"] == "capacity"
    assert model_server.answer_requests == before
    assert "visitor_session" not in newcomer.cookies
    follow_up = visitor_a.post(
        "/chat", json={"query": "Which professional topic did I mention earlier?"}
    )
    assert follow_up.status_code == 200
    assert follow_up.json()["answer"]["answer"] == "Topics: ALPHA"


def test_capacity_reclaims_expired_context_but_keeps_active_visitors(
    visitors: tuple[TestClient, TestClient], clock: Clock, monkeypatch: MonkeyPatch
) -> None:
    abandoned, newcomer = visitors
    monkeypatch.setattr(main.config, "conversation_capacity", 2)
    assert (
        abandoned.post(
            "/chat", json={"query": "Please remember my professional topic ALPHA."}
        ).status_code
        == 200
    )
    clock.advance(900)
    with TestClient(abandoned.app, base_url="http://localhost") as active:
        assert (
            active.post(
                "/chat", json={"query": "Please remember my professional topic BETA."}
            ).status_code
            == 200
        )
        clock.advance(900)
        admitted = newcomer.post("/chat", json={"query": "Describe the candidate."})
        assert admitted.status_code == 200
        assert admitted.json()["answer"]["answer"] == "Topics: "
        retained = active.post(
            "/chat", json={"query": "Which professional topic did I mention earlier?"}
        )
        assert retained.json()["answer"]["answer"] == "Topics: BETA"

    expired = abandoned.get("/chat/conversation")
    assert expired.json() == {"messages": []}
    assert expired.headers["X-Conversation-Status"] == "expired"
    rejected = abandoned.post(
        "/chat", json={"query": "Which professional topic did I mention earlier?"}
    )
    assert rejected.status_code == 409
    assert rejected.headers["X-Conversation-Status"] == "expired"
    # Forget the reclaimed identity without retaining an unbounded tombstone map.
    assert "visitor_session" not in abandoned.cookies
    assert abandoned.post("/chat/conversation").status_code == 503
    clock.advance(1800)
    fresh = abandoned.post(
        "/chat", json={"query": "Which professional topic did I mention earlier?"}
    )
    assert fresh.status_code == 200
    assert fresh.json()["answer"]["answer"] == "Topics: "


def test_reclaiming_a_slot_invalidates_its_still_running_generation(
    visitors: tuple[TestClient, TestClient],
    clock: Clock,
    monkeypatch: MonkeyPatch,
    model_server: ModelServer,
) -> None:
    old_visitor, newcomer = visitors
    monkeypatch.setattr(main.config, "conversation_capacity", 1)
    assert (
        old_visitor.post(
            "/chat", json={"query": "Please remember my professional topic ALPHA."}
        ).status_code
        == 200
    )
    model_server.pause_next_answer = True
    with ThreadPoolExecutor(max_workers=1) as requests:
        pending = requests.submit(
            old_visitor.post,
            "/chat",
            json={"query": "Please remember my professional topic BETA."},
        )
        try:
            assert model_server.started.wait(timeout=5), "Model request never started"
            clock.advance(1800)
            admitted = newcomer.post("/chat", json={"query": "Describe the candidate."})
            assert admitted.status_code == 200
        finally:
            model_server.release.set()
        stale = pending.result(timeout=5)

    assert stale.status_code == 409
    assert stale.headers["X-Conversation-Status"] == "expired"
    assert old_visitor.get("/chat/conversation").json() == {"messages": []}
    retained = newcomer.get("/chat/conversation").json()["messages"]
    assert retained == [
        {"role": "user", "content": "Describe the candidate."},
        {"role": "assistant", "content": "Topics: "},
    ]


@pytest.mark.parametrize("reset_action", ["clear", "login", "logout"])
def test_explicit_reset_forgets_a_reclaimed_visitor_identity(
    visitors: tuple[TestClient, TestClient],
    clock: Clock,
    monkeypatch: MonkeyPatch,
    reset_action: str,
) -> None:
    old_visitor, newcomer = visitors
    monkeypatch.setattr(main.config, "conversation_capacity", 1)
    assert old_visitor.post("/chat/conversation").status_code == 204
    clock.advance(1800)
    assert newcomer.post("/chat/conversation").status_code == 204
    if reset_action == "login":
        monkeypatch.setattr(
            auth,
            "INVITE_CODES",
            {"shared-code": {"active": True, "company": "Example Company"}},
        )
        reset = old_visitor.post("/auth/login", json={"invite_code": "shared-code"})
    elif reset_action == "logout":
        reset = old_visitor.post("/auth/logout")
    else:
        reset = old_visitor.delete("/chat/conversation")
    assert reset.status_code in (200, 204)
    assert "visitor_session" not in old_visitor.cookies
    # An explicit reset already acknowledges the old context's removal.
    assert old_visitor.post("/chat/conversation").status_code == 503


def test_expiration_discards_a_generation_that_finishes_after_the_idle_deadline(
    visitors: tuple[TestClient, TestClient], clock: Clock, model_server: ModelServer
) -> None:
    visitor, _ = visitors
    assert (
        visitor.post(
            "/chat", json={"query": "Please remember my professional topic ALPHA."}
        ).status_code
        == 200
    )
    model_server.pause_next_answer = True
    with ThreadPoolExecutor(max_workers=1) as requests:
        pending = requests.submit(
            visitor.post,
            "/chat",
            json={"query": "Please remember my professional topic BETA."},
        )
        try:
            assert model_server.started.wait(timeout=5), "Model request never started"
            clock.advance(1800)
        finally:
            model_server.release.set()
        stale = pending.result(timeout=5)

    assert stale.status_code == 409
    assert stale.headers["X-Conversation-Status"] == "expired"
    assert visitor.get("/chat/conversation").json() == {"messages": []}
    retry = visitor.post(
        "/chat", json={"query": "Which professional topic did I mention earlier?"}
    )
    assert retry.status_code == 200
    assert retry.json()["answer"]["answer"] == "Topics: "


@pytest.mark.parametrize("authenticated", [False, True])
def test_visitors_can_read_only_their_current_transcript(
    visitors: tuple[TestClient, TestClient],
    monkeypatch: MonkeyPatch,
    authenticated: bool,
) -> None:
    visitor_a, visitor_b = visitors
    if authenticated:
        monkeypatch.setattr(
            auth,
            "INVITE_CODES",
            {"shared-code": {"active": True, "company": "Example Company"}},
        )
    for visitor, topic in ((visitor_a, "ALPHA"), (visitor_b, "BETA")):
        if authenticated:
            assert (
                visitor.post(
                    "/auth/login", json={"invite_code": "shared-code"}
                ).status_code
                == 200
            )
        question = f"Please remember my professional topic {topic}."
        sent = visitor.post("/chat", json={"query": question})
        assert sent.status_code == 200
        transcript = visitor.get("/chat/conversation")
        assert transcript.status_code == 200
        assert transcript.headers["cache-control"] == "no-store"
        assert transcript.json() == {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"Topics: {topic}"},
            ]
        }

    cleared = visitor_a.delete("/chat/conversation")
    assert cleared.status_code == 204
    assert visitor_a.get("/chat/conversation").json() == {"messages": []}
    assert visitor_b.get("/chat/conversation").json() == {
        "messages": [
            {"role": "user", "content": "Please remember my professional topic BETA."},
            {"role": "assistant", "content": "Topics: BETA"},
        ]
    }
    if authenticated:
        assert visitor_a.post("/auth/logout").status_code == 200
        assert visitor_a.get("/chat/conversation").status_code == 401


def test_clear_resets_only_the_current_visitors_context(
    visitors: tuple[TestClient, TestClient],
) -> None:
    visitor_a, visitor_b = visitors
    for visitor, topic in ((visitor_a, "ALPHA"), (visitor_b, "BETA")):
        response = visitor.post(
            "/chat", json={"query": f"Please remember my professional topic {topic}."}
        )
        assert response.status_code == 200

    cleared = visitor_a.delete("/chat/conversation")
    assert cleared.status_code == 204

    answers = []
    for visitor in (visitor_a, visitor_b):
        response = visitor.post(
            "/chat", json={"query": "Which professional topic did I mention earlier?"}
        )
        assert response.status_code == 200
        answers.append(response.json()["answer"]["answer"])

    assert answers == ["Topics: ", "Topics: BETA"]


@pytest.mark.parametrize(
    ("login_code", "expected_status", "expected_topic"),
    [
        ("shared-code", 200, "Topics: "),
        ("different-code", 200, "Topics: "),
        ("invalid-code", 401, "Topics: ALPHA"),
    ],
)
def test_only_successful_login_resets_the_visitors_conversation(
    visitors: tuple[TestClient, TestClient],
    monkeypatch: MonkeyPatch,
    login_code: str,
    expected_status: int,
    expected_topic: str,
) -> None:
    visitor_a, visitor_b = visitors
    monkeypatch.setattr(
        auth,
        "INVITE_CODES",
        {
            "shared-code": {"active": True, "company": "Example Company"},
            "different-code": {"active": True, "company": "Other Company"},
        },
    )
    for visitor, topic in ((visitor_a, "ALPHA"), (visitor_b, "BETA")):
        login = visitor.post("/auth/login", json={"invite_code": "shared-code"})
        assert login.status_code == 200
        seeded = visitor.post(
            "/chat", json={"query": f"Please remember my professional topic {topic}."}
        )
        assert seeded.status_code == 200
        assert seeded.json()["answer"]["answer"] == f"Topics: {topic}"

    changed = visitor_a.post("/auth/login", json={"invite_code": login_code})
    assert changed.status_code == expected_status

    answers = []
    for visitor in (visitor_a, visitor_b):
        response = visitor.post(
            "/chat", json={"query": "Which professional topic did I mention earlier?"}
        )
        assert response.status_code == 200
        answers.append(response.json()["answer"]["answer"])

    assert answers == [expected_topic, "Topics: BETA"]


@pytest.mark.parametrize("authenticated", [False, True])
def test_logout_resets_only_the_visitors_context_and_discards_a_pending_answer(
    visitors: tuple[TestClient, TestClient],
    model_server: ModelServer,
    monkeypatch: MonkeyPatch,
    authenticated: bool,
) -> None:
    visitor_a, visitor_b = visitors
    if authenticated:
        monkeypatch.setattr(
            auth,
            "INVITE_CODES",
            {"shared-code": {"active": True, "company": "Example Company"}},
        )
    for visitor, topic in ((visitor_a, "ALPHA"), (visitor_b, "BETA")):
        if authenticated:
            login = visitor.post("/auth/login", json={"invite_code": "shared-code"})
            assert login.status_code == 200
        seeded = visitor.post(
            "/chat", json={"query": f"Please remember my professional topic {topic}."}
        )
        assert seeded.status_code == 200
        assert seeded.json()["answer"]["answer"] == f"Topics: {topic}"

    model_server.pause_next_answer = True
    with ThreadPoolExecutor(max_workers=2) as requests:
        pending = requests.submit(
            visitor_a.post,
            "/chat",
            json={"query": "Which professional topic did I mention earlier?"},
        )
        try:
            assert model_server.started.wait(timeout=5), "Model request never started"
            logout = requests.submit(visitor_a.post, "/auth/logout")
            assert logout.result(timeout=2).status_code == 200
        finally:
            model_server.release.set()
        stale = pending.result(timeout=5)

    assert stale.status_code == 409
    own_follow_up = visitor_a.post(
        "/chat", json={"query": "Which professional topic did I mention earlier?"}
    )
    if authenticated:
        assert own_follow_up.status_code == 401
        status = visitor_a.get("/auth/status")
        assert status.json()["authenticated"] is False
    else:
        assert own_follow_up.status_code == 200
        assert own_follow_up.json()["answer"]["answer"] == "Topics: "

    other_follow_up = visitor_b.post(
        "/chat", json={"query": "Which professional topic did I mention earlier?"}
    )
    assert other_follow_up.status_code == 200
    assert other_follow_up.json()["answer"]["answer"] == "Topics: BETA"


def test_busy_conversation_rejects_a_second_request_but_not_another_visitor(
    visitors: tuple[TestClient, TestClient], model_server: ModelServer
) -> None:
    visitor_a, visitor_b = visitors
    initial = visitor_a.post(
        "/chat", json={"query": "Please remember my professional topic ALPHA."}
    )
    assert initial.status_code == 200
    model_server.pause_next_answer = True
    with ThreadPoolExecutor(max_workers=2) as requests:
        pending = requests.submit(
            visitor_a.post,
            "/chat",
            json={"query": "Which professional topic did I mention earlier?"},
        )
        try:
            assert model_server.started.wait(timeout=5), "Model request never started"
            busy = requests.submit(
                visitor_a.post,
                "/chat",
                json={"query": "Please remember my professional topic BETA."},
            ).result(timeout=2)
            assert busy.status_code == 409
            assert busy.json() == {
                "detail": "Conversation is busy. Please wait for the current answer."
            }
            other = requests.submit(
                visitor_b.post,
                "/chat",
                json={"query": "Please remember my professional topic BETA."},
            ).result(timeout=2)
            assert other.status_code == 200
            assert other.json()["answer"]["answer"] == "Topics: BETA"
        finally:
            model_server.release.set()
        assert pending.result(timeout=5).status_code == 200

    recovered = visitor_a.post(
        "/chat", json={"query": "Which professional topic did I mention earlier?"}
    )
    assert recovered.status_code == 200
    assert recovered.json()["answer"]["answer"] == "Topics: ALPHA"


def test_clear_discards_an_in_flight_answer(
    visitors: tuple[TestClient, TestClient], model_server: ModelServer
) -> None:
    visitor, _ = visitors
    # Establish ownership before starting a second request from the same browser.
    initial = visitor.post("/chat", json={"query": "What are the candidate's skills?"})
    assert initial.status_code == 200
    model_server.pause_next_answer = True

    with ThreadPoolExecutor(max_workers=2) as requests:
        pending = requests.submit(
            visitor.post,
            "/chat",
            json={"query": "Please remember my professional topic ALPHA."},
        )
        try:
            assert model_server.started.wait(timeout=5), "Model request never started"
            cleared = requests.submit(visitor.delete, "/chat/conversation")
            assert cleared.result(timeout=2).status_code == 204
        finally:
            model_server.release.set()

        stale = pending.result(timeout=5)

    assert stale.status_code == 409
    assert stale.json() == {"detail": "Conversation was reset; answer discarded."}
    follow_up = visitor.post(
        "/chat", json={"query": "Which professional topic did I mention earlier?"}
    )
    assert follow_up.status_code == 200
    assert follow_up.json()["answer"]["answer"] == "Topics: "


@pytest.mark.browser
@pytest.mark.parametrize("fail_summary", [False, True])
def test_summary_button_keeps_its_result_separate_from_chat_and_draft(
    browser_page: Page, model_server: ModelServer, fail_summary: bool
) -> None:
    page = browser_page
    query = page.locator("#chat-query")
    query.fill("Please remember my professional topic ALPHA.")
    page.locator("#submit-btn").click()
    expect(page.locator("#response")).to_contain_text("Assistant: Topics: ALPHA")
    transcript = [
        "You: Please remember my professional topic ALPHA.",
        "Assistant: Topics: ALPHA",
    ]
    draft = "An unsent question about the candidate's skills."
    query.fill(draft)
    model_server.fail_summaries = fail_summary
    page.locator("#summary-button").click()
    expected = (
        "Unable to generate summary. Please try again."
        if fail_summary
        else "Example Candidate develops Python software."
    )
    expect(page.locator("#summary-response")).to_contain_text(expected)
    expect(page.locator("#response > div")).to_have_text(transcript)
    expect(query).to_have_value(draft)
    page.reload()
    page.add_style_tag(content=".hidden { display: none !important; }")
    expect(page.locator("#response > div")).to_have_text(transcript)
    query.fill("Which professional topic did I mention earlier?")
    page.locator("#submit-btn").click()
    expect(page.locator("#response > div")).to_have_count(4)
    expect(page.locator("#response > div").last).to_have_text(
        "Assistant: Topics: ALPHA"
    )


@pytest.mark.browser
def test_local_input_validation_does_not_add_an_assistant_turn(
    browser_page: Page, model_server: ModelServer
) -> None:
    page = browser_page
    page.locator("#chat-query").fill("Please remember my professional topic ALPHA.")
    page.locator("#submit-btn").click()
    expect(page.locator("#response")).to_contain_text("Assistant: Topics: ALPHA")
    before = model_server.answer_requests
    page.locator("#chat-query").fill("   ")
    page.locator("#submit-btn").click()
    expect(page.locator("#chat-status")).to_have_text("Query cannot be empty")
    expect(page.locator("#response > div")).to_have_text(
        [
            "You: Please remember my professional topic ALPHA.",
            "Assistant: Topics: ALPHA",
        ]
    )
    expect(page.locator("#chat-query")).to_have_value("   ")
    assert model_server.answer_requests == before


@pytest.mark.browser
@pytest.mark.parametrize("failure", ["provider", "rejected_output", "transport"])
def test_failed_generation_is_a_notice_not_a_visible_or_stored_turn(
    browser_page: Page, model_server: ModelServer, failure: str
) -> None:
    page = browser_page
    query = page.locator("#chat-query")
    query.fill("Please remember my professional topic ALPHA.")
    page.locator("#submit-btn").click()
    expect(page.locator("#response")).to_contain_text("Assistant: Topics: ALPHA")
    transcript = [
        "You: Please remember my professional topic ALPHA.",
        "Assistant: Topics: ALPHA",
    ]
    model_server.fail_answers = failure == "provider"
    if failure == "rejected_output":
        model_server.answer_override = "UNSAFE_SYNTHETIC_OUTPUT"
    if failure == "transport":
        page.route("**/chat", lambda route: route.abort(), times=1)
    page.clock.fast_forward(3001)
    draft = "Please remember my professional topic BETA."
    query.fill(draft)
    page.locator("#submit-btn").click()
    expected_notice = (
        "An error occurred. Please try again."
        if failure == "transport"
        else "I'm sorry, but I cannot provide a response to that query."
    )
    expect(page.locator("#chat-status")).to_have_text(expected_notice)
    expect(page.locator("#response > div")).to_have_text(transcript)
    expect(query).to_have_value(draft)
    expect(page.locator("#submit-btn")).to_be_enabled()
    model_server.fail_answers = False
    model_server.answer_override = None
    page.reload()
    page.add_style_tag(content=".hidden { display: none !important; }")
    expect(page.locator("#response > div")).to_have_text(transcript)
    query.fill("Which professional topic did I mention earlier?")
    page.locator("#submit-btn").click()
    expect(page.locator("#response > div")).to_have_count(4)
    expect(page.locator("#response > div").last).to_have_text(
        "Assistant: Topics: ALPHA"
    )


@pytest.mark.browser
@pytest.mark.parametrize("conversation_capacity", [2], indirect=True)
def test_capacity_notice_preserves_the_draft_until_a_slot_is_available(
    browser_page: Page,
    visitors: tuple[TestClient, TestClient],
    clock: Clock,
    model_server: ModelServer,
) -> None:
    for visitor in visitors:
        assert visitor.post("/chat/conversation").status_code == 204
    page = browser_page
    draft = "What are the candidate's professional skills?"
    page.locator("#chat-query").fill(draft)
    before = model_server.answer_requests
    page.locator("#submit-btn").click()
    expect(page.locator("#chat-status")).to_have_text(
        "Conversation capacity is temporarily full. Please try again later."
    )
    expect(page.locator("#response")).to_be_empty()
    expect(page.locator("#chat-query")).to_have_value(draft)
    expect(page.locator("#submit-btn")).to_be_enabled()
    assert model_server.answer_requests == before

    clock.advance(1800)
    page.clock.fast_forward(3001)
    page.locator("#submit-btn").click()
    expect(page.locator("#response")).to_contain_text("Assistant: Topics:")
    expect(page.locator("#chat-query")).to_have_value("")


@pytest.mark.browser
@pytest.mark.parametrize("reload_before_submission", [False, True])
def test_expired_submission_preserves_draft_and_requires_explicit_retry(
    browser_page: Page,
    clock: Clock,
    model_server: ModelServer,
    reload_before_submission: bool,
) -> None:
    page = browser_page
    query = page.locator("#chat-query")
    query.fill("Please remember my professional topic ALPHA.")
    page.locator("#submit-btn").click()
    expect(page.locator("#response")).to_contain_text("Assistant: Topics: ALPHA")
    other_tab = page.context.new_page()
    other_tab.goto(page.url)
    other_tab.add_style_tag(content=".hidden { display: none !important; }")
    expect(other_tab.locator("#response")).to_contain_text("Assistant: Topics: ALPHA")
    other_tab.locator("#chat-query").fill("A separate unsent draft.")
    clock.advance(1800)
    if reload_before_submission:
        page.reload()
        page.add_style_tag(content=".hidden { display: none !important; }")
        expect(page.locator("#chat-status")).to_have_text(
            "Conversation expired after inactivity. Review and resubmit your question."
        )
    page.clock.fast_forward(3001)
    draft = "Which professional topic did I mention earlier?"
    query.fill(draft)
    before = model_server.answer_requests
    page.locator("#submit-btn").click()
    expect(page.locator("#chat-status")).to_have_text(
        "Conversation expired after inactivity. Review and resubmit your question."
    )
    expect(page.locator("#response")).to_be_empty()
    expect(query).to_have_value(draft)
    expect(page.locator("#submit-btn")).to_be_enabled()
    expect(other_tab.locator("#response")).to_be_empty()
    expect(other_tab.locator("#chat-status")).to_have_text(
        "Conversation expired after inactivity. Review and resubmit your question."
    )
    expect(other_tab.locator("#chat-query")).to_have_value("A separate unsent draft.")
    assert model_server.answer_requests == before

    page.clock.fast_forward(3001)
    page.locator("#submit-btn").click()
    expect(page.locator("#response")).to_contain_text("Assistant: Topics:")
    expect(page.locator("#response")).not_to_contain_text("ALPHA")
    expect(query).to_have_value("")


@pytest.mark.browser
@pytest.mark.parametrize("conversation_max_turns", [2], indirect=True)
def test_browser_shows_only_the_configured_context_window(
    browser_page: Page,
) -> None:
    page = browser_page
    expect(
        page.get_by_text(
            "Only the latest 2 completed question/answer pairs are remembered and displayed."
        )
    ).to_be_visible()
    questions = [
        "Please remember my professional topic ALPHA.",
        "Describe the candidate's skills, request 0.",
        "Describe the candidate's skills, request 1.",
        "Which professional topic did I mention earlier?",
    ]
    for question in questions:
        page.locator("#chat-query").fill(question)
        page.locator("#submit-btn").click()
        expect(page.locator("#submit-btn")).to_be_enabled()
        page.clock.fast_forward(3001)

    transcript = page.locator("#response")
    expect(transcript.locator(":scope > div")).to_have_count(4)
    expect(transcript).not_to_contain_text(questions[0])
    expect(transcript).not_to_contain_text(questions[1])
    expect(transcript.locator(":scope > div").last).to_have_text("Assistant: Topics: ")
    page.reload()
    expect(transcript.locator(":scope > div")).to_have_count(4)
    expect(transcript).to_contain_text(questions[-1])


@pytest.mark.browser
@pytest.mark.parametrize("first_query", [True, False])
def test_clear_button_resets_context_and_hides_a_late_answer(
    browser_page: Page, model_server: ModelServer, first_query: bool
) -> None:
    page = browser_page
    query = page.locator("#chat-query")
    if not first_query:
        query.fill("What are the candidate's skills?")
        page.locator("#submit-btn").click()
        expect(page.locator("#response")).to_contain_text("Topics:")
        page.clock.fast_forward(3001)
    model_server.pause_next_answer = True
    query.fill("Please remember my professional topic ALPHA.")
    with page.expect_request(
        lambda request: request.method == "POST" and request.url.endswith("/chat")
    ):
        page.locator("#submit-btn").click()

    try:
        assert model_server.started.wait(timeout=5), "Model request never started"
        with page.expect_response(
            lambda response: response.request.method == "DELETE"
            and response.url.endswith("/chat/conversation"),
            timeout=3000,
        ) as reset:
            page.locator("#clear-chat").click()
        assert reset.value.status == 204
        expect(page.locator("#response")).to_be_empty()
        with page.expect_response(
            lambda response: response.request.method == "POST"
            and response.url.endswith("/chat"),
        ) as pending:
            model_server.release.set()
        assert pending.value.status == 409
        expect(page.locator("#submit-btn")).to_be_enabled()
        expect(page.locator("#response")).to_be_empty()
    finally:
        model_server.release.set()

    page.clock.fast_forward(3001)
    query.fill("Which professional topic did I mention earlier?")
    page.locator("#submit-btn").click()
    expect(page.locator("#response")).to_contain_text("Assistant: Topics:")
    expect(page.locator("#response")).not_to_contain_text("ALPHA")


@pytest.mark.browser
def test_failed_clear_keeps_the_visible_and_server_context(browser_page: Page) -> None:
    page = browser_page
    query = page.locator("#chat-query")
    query.fill("Please remember my professional topic ALPHA.")
    page.locator("#submit-btn").click()
    expect(page.locator("#response")).to_contain_text("Assistant: Topics: ALPHA")
    transcript = page.locator("#response").inner_text()

    # Simulate a dropped reset request, not a replacement backend implementation.
    page.route(
        "**/chat/conversation",
        lambda route: (
            route.abort() if route.request.method == "DELETE" else route.continue_()
        ),
    )
    page.locator("#clear-chat").click()
    expect(page.locator("#chat-status")).to_have_text(
        "Could not clear the conversation. Please try again."
    )
    expect(page.locator("#response")).to_have_text(transcript, use_inner_text=True)
    expect(page.locator("#clear-chat")).to_be_enabled()
    expect(page.locator("#submit-btn")).to_be_enabled()

    page.clock.fast_forward(3001)
    query.fill("Which professional topic did I mention earlier?")
    page.locator("#submit-btn").click()
    expect(page.locator("#response > div")).to_have_count(4)
    expect(page.locator("#response > div").last).to_have_text(
        "Assistant: Topics: ALPHA"
    )


@pytest.mark.browser
@pytest.mark.parametrize("browser_page", [True], indirect=True)
def test_logout_hides_the_transcript_and_late_answer_before_logging_in_again(
    browser_page: Page, model_server: ModelServer
) -> None:
    page = browser_page
    page.locator("#invite-code").fill("shared-code")
    page.locator("#login-form button").click()
    expect(page.locator("#login-modal")).to_be_hidden()
    query = page.locator("#chat-query")
    query.fill("Please remember my professional topic ALPHA.")
    page.locator("#submit-btn").click()
    expect(page.locator("#response")).to_contain_text("Assistant: Topics: ALPHA")

    page.clock.fast_forward(3001)
    model_server.pause_next_answer = True
    query.fill("Which professional topic did I mention earlier?")
    with page.expect_request(
        lambda request: request.method == "POST" and request.url.endswith("/chat")
    ):
        page.locator("#submit-btn").click()
    try:
        assert model_server.started.wait(timeout=5), "Model request never started"
        with page.expect_response("**/auth/logout") as logged_out:
            page.locator("#logout-btn").click()
        assert logged_out.value.status == 200
        expect(page.locator("#login-modal")).to_be_visible()
        expect(page.locator("#response")).to_be_empty()
        with page.expect_response("**/chat") as pending:
            model_server.release.set()
        assert pending.value.status == 409
        expect(page.locator("#response")).to_be_empty()
        expect(page.locator("#submit-btn")).to_be_enabled()
    finally:
        model_server.release.set()

    page.locator("#invite-code").fill("shared-code")
    page.locator("#login-form button").click()
    expect(page.locator("#login-modal")).to_be_hidden()
    page.clock.fast_forward(3001)
    query.fill("Which professional topic did I mention earlier?")
    page.locator("#submit-btn").click()
    expect(page.locator("#response")).to_contain_text("Assistant: Topics:")
    expect(page.locator("#response > div")).to_have_count(2)
    expect(page.locator("#response")).not_to_contain_text("ALPHA")


@pytest.mark.browser
@pytest.mark.parametrize("browser_page", [True], indirect=True)
@pytest.mark.parametrize("failure", ["http", "network"])
def test_failed_logout_keeps_the_authenticated_conversation(
    browser_page: Page, failure: str
) -> None:
    page = browser_page
    page.locator("#invite-code").fill("shared-code")
    page.locator("#login-form button").click()
    expect(page.locator("#login-modal")).to_be_hidden()
    query = page.locator("#chat-query")
    query.fill("Please remember my professional topic ALPHA.")
    page.locator("#submit-btn").click()
    expect(page.locator("#response")).to_contain_text("Assistant: Topics: ALPHA")
    transcript = page.locator("#response").inner_text()

    page.route(
        "**/auth/logout",
        lambda route: (
            route.abort()
            if failure == "network"
            else route.fulfill(status=503, json={"detail": "Temporarily unavailable"})
        ),
    )
    page.locator("#logout-btn").click()
    expect(page.locator("#chat-status")).to_have_text(
        "Could not log out. Please try again."
    )
    expect(page.locator("#login-modal")).to_be_hidden()
    expect(page.locator("#user-info")).to_be_visible()
    expect(page.locator("#response")).to_have_text(transcript, use_inner_text=True)

    page.clock.fast_forward(3001)
    query.fill("Which professional topic did I mention earlier?")
    page.locator("#submit-btn").click()
    expect(page.locator("#response > div")).to_have_count(4)
    expect(page.locator("#response > div").last).to_have_text(
        "Assistant: Topics: ALPHA"
    )


@pytest.mark.browser
@pytest.mark.parametrize("browser_page", [False, True], indirect=True)
def test_reload_restores_live_history_but_never_cleared_history(
    browser_page: Page,
) -> None:
    page = browser_page
    if page.locator("#login-modal").is_visible():
        page.locator("#invite-code").fill("shared-code")
        page.locator("#login-form button").click()
        expect(page.locator("#login-modal")).to_be_hidden()
    question = (
        "Please remember my professional topic ALPHA. Keep <b>literal</b> as text."
    )
    page.locator("#chat-query").fill(question)
    page.locator("#submit-btn").click()
    expect(page.locator("#response")).to_contain_text("Assistant: Topics: ALPHA")

    for cleared in (False, True):
        if cleared:
            page.locator("#clear-chat").click()
            expect(page.locator("#response")).to_be_empty()
        with page.expect_response(
            lambda response: response.request.method == "GET"
            and response.url.endswith("/chat/conversation"),
            timeout=3000,
        ) as restored:
            page.reload()
        assert restored.value.status == 200
        page.add_style_tag(content=".hidden { display: none !important; }")
        expect(page.locator("#login-modal")).to_be_hidden()
        expect(page.locator("#submit-btn")).to_be_enabled()
        if cleared:
            expect(page.locator("#response")).to_be_empty()
        else:
            expect(page.locator("#response > div")).to_have_text(
                [
                    f"You: {question}",
                    "Assistant: Topics: ALPHA",
                ]
            )


@pytest.mark.browser
def test_failed_restore_requires_retry_before_accepting_more_questions(
    browser_page: Page,
) -> None:
    page = browser_page
    page.locator("#chat-query").fill("Please remember my professional topic ALPHA.")
    page.locator("#submit-btn").click()
    expect(page.locator("#response")).to_contain_text("Assistant: Topics: ALPHA")
    pattern = "**/chat/conversation"
    page.route(
        pattern,
        lambda route: (
            route.abort() if route.request.method == "GET" else route.continue_()
        ),
    )
    page.reload()
    expect(page.locator("#chat-status")).to_have_text(
        "Could not restore the conversation. Reload to retry."
    )
    expect(page.locator("#submit-btn")).to_be_disabled()
    expect(page.locator("#clear-chat")).to_be_disabled()

    page.unroute(pattern)
    page.reload()
    expect(page.locator("#submit-btn")).to_be_enabled()
    expect(page.locator("#response > div")).to_have_text(
        [
            "You: Please remember my professional topic ALPHA.",
            "Assistant: Topics: ALPHA",
        ]
    )


@pytest.mark.browser
def test_tabs_share_completed_turns_and_clear_without_syncing_other_profiles(
    browser_page: Page,
) -> None:
    first = browser_page
    first.locator("#chat-query").fill("Please remember my professional topic ALPHA.")
    first.locator("#submit-btn").click()
    expect(first.locator("#response")).to_contain_text("Assistant: Topics: ALPHA")

    second = first.context.new_page()
    second.goto(first.url)
    second.add_style_tag(content=".hidden { display: none !important; }")
    expect(second.locator("#response")).to_contain_text("Assistant: Topics: ALPHA")
    browser = first.context.browser
    assert browser is not None
    other_profile = browser.new_context()
    try:
        other_profile.route(
            re.compile(r"^https?://(?!127\.0\.0\.1:)"), lambda route: route.abort()
        )
        outsider = other_profile.new_page()
        outsider.goto(first.url)
        expect(outsider.locator("#submit-btn")).to_be_enabled()
        outsider.locator("#chat-query").fill(
            "Please remember my professional topic BETA."
        )
        outsider.locator("#submit-btn").click()
        expect(outsider.locator("#response")).to_contain_text("Assistant: Topics: BETA")

        second.locator("#chat-query").fill(
            "Which professional topic did I mention earlier?"
        )
        second.locator("#submit-btn").click()
        expect(second.locator("#response > div")).to_have_count(4)
        expect(first.locator("#response > div")).to_have_text(
            [
                "You: Please remember my professional topic ALPHA.",
                "Assistant: Topics: ALPHA",
                "You: Which professional topic did I mention earlier?",
                "Assistant: Topics: ALPHA",
            ]
        )
        expect(outsider.locator("#response > div")).to_have_count(2)
        expect(outsider.locator("#response")).not_to_contain_text("ALPHA")

        second.locator("#clear-chat").click()
        expect(second.locator("#response")).to_be_empty()
        expect(first.locator("#response")).to_be_empty()
        expect(outsider.locator("#response")).to_contain_text("Assistant: Topics: BETA")
    finally:
        other_profile.close()


@pytest.mark.browser
@pytest.mark.parametrize("browser_page", [True], indirect=True)
@pytest.mark.parametrize("transition", ["clear", "logout_login"])
def test_tabs_discard_pending_answers_after_clear_or_auth_changes(
    browser_page: Page, model_server: ModelServer, transition: str
) -> None:
    first = browser_page
    first.locator("#invite-code").fill("shared-code")
    first.locator("#login-form button").click()
    expect(first.locator("#login-modal")).to_be_hidden()
    first.locator("#chat-query").fill("Please remember my professional topic ALPHA.")
    first.locator("#submit-btn").click()
    expect(first.locator("#response")).to_contain_text("Assistant: Topics: ALPHA")
    second = first.context.new_page()
    second.goto(first.url)
    second.add_style_tag(content=".hidden { display: none !important; }")
    expect(second.locator("#response")).to_contain_text("Assistant: Topics: ALPHA")
    first.clock.fast_forward(3001)
    model_server.pause_next_answer = True
    first.locator("#chat-query").fill("Which professional topic did I mention earlier?")
    with first.expect_request(
        lambda request: request.method == "POST" and request.url.endswith("/chat")
    ):
        first.locator("#submit-btn").click()

    try:
        assert model_server.started.wait(timeout=5), "Model request never started"
        if transition == "clear":
            second.locator("#clear-chat").click()
        else:
            second.locator("#logout-btn").click()
            expect(second.locator("#login-modal")).to_be_visible()
            expect(first.locator("#login-modal")).to_be_visible()
            expect(first.locator("#user-info")).to_be_hidden()
            expect(first.locator("#response")).to_be_empty()
            second.locator("#invite-code").fill("shared-code")
            second.locator("#login-form button").click()
        expect(first.locator("#response")).to_be_empty()
        expect(first.locator("#login-modal")).to_be_hidden()
        expect(first.locator("#user-info")).to_be_visible()
        with first.expect_response("**/chat") as pending:
            model_server.release.set()
        assert pending.value.status == 409
        expect(first.locator("#response")).to_be_empty()
    finally:
        model_server.release.set()

    first.clock.fast_forward(3001)
    first.locator("#chat-query").fill("Which professional topic did I mention earlier?")
    first.locator("#submit-btn").click()
    expect(first.locator("#response")).to_contain_text("Assistant: Topics:")
    expect(second.locator("#response > div")).to_have_text(
        ["You: Which professional topic did I mention earlier?", "Assistant: Topics:"]
    )


@pytest.mark.browser
def test_busy_tab_keeps_the_question_for_explicit_retry(
    browser_page: Page, model_server: ModelServer
) -> None:
    first = browser_page
    first.locator("#chat-query").fill("Please remember my professional topic ALPHA.")
    first.locator("#submit-btn").click()
    expect(first.locator("#response")).to_contain_text("Assistant: Topics: ALPHA")
    second = first.context.new_page()
    second.goto(first.url)
    expect(second.locator("#response")).to_contain_text("Assistant: Topics: ALPHA")
    first.clock.fast_forward(3001)
    model_server.pause_next_answer = True
    first.locator("#chat-query").fill("Which professional topic did I mention earlier?")
    with first.expect_request("**/chat"):
        first.locator("#submit-btn").click()
    question = "Please remember my professional topic BETA."
    try:
        assert model_server.started.wait(timeout=5), "Model request never started"
        second.locator("#chat-query").fill(question)
        with second.expect_response("**/chat") as rejected:
            second.locator("#submit-btn").click()
        assert rejected.value.status == 409
        expect(second.locator("#chat-query")).to_have_value(question)
        expect(second.locator("#chat-status")).to_have_text(
            "Conversation is busy. Please wait for the current answer."
        )
        expect(second.locator("#response > div")).to_have_count(2)
        with first.expect_response("**/chat"):
            model_server.release.set()
        expect(second.locator("#response > div")).to_have_count(4)
        expect(second.locator("#chat-query")).to_have_value(question)
    finally:
        model_server.release.set()

    second.clock.fast_forward(3001)
    second.locator("#submit-btn").click()
    expect(second.locator("#response > div").last).to_have_text(
        "Assistant: Topics: ALPHA, BETA"
    )
    expect(first.locator("#response > div")).to_have_count(6)


@pytest.mark.browser
def test_fresh_tabs_bootstrap_one_session_before_parallel_generation(
    browser_page: Page, model_server: ModelServer, preparation_gate: PreparationGate
) -> None:
    first = browser_page
    second = first.context.new_page()
    second.goto(first.url)
    expect(second.locator("#submit-btn")).to_be_enabled()
    preparation_gate.pause_next = True
    model_server.pause_next_answer = True
    first.locator("#chat-query").fill("Please remember my professional topic ALPHA.")
    first.locator("#submit-btn").click()
    try:
        assert preparation_gate.started.wait(timeout=5), "Bootstrap never started"
        second.locator("#chat-query").fill(
            "Please remember my professional topic BETA."
        )
        second.locator("#submit-btn").click()
        # Let an uncoordinated second tab reach the model before releasing the
        # first bootstrap. Correct coordination keeps it waiting instead.
        model_server.started.wait(timeout=2)
        with first.context.expect_event(
            "response",
            predicate=lambda response: response.url.endswith("/chat")
            and response.status == 409,
            timeout=3000,
        ) as rejected:
            preparation_gate.release.set()
        assert model_server.started.wait(timeout=5), "Model request never started"
        rejected_question = rejected.value.request.post_data_json["query"]
        with first.context.expect_event(
            "response",
            predicate=lambda response: response.url.endswith("/chat")
            and response.status == 200,
        ) as accepted:
            model_server.release.set()
        accepted_question = accepted.value.request.post_data_json["query"]
        topic = "ALPHA" if "ALPHA" in accepted_question else "BETA"
        for page in (first, second):
            expect(page.locator("#response > div")).to_have_text(
                [f"You: {accepted_question}", f"Assistant: Topics: {topic}"]
            )
        assert sorted(
            page.locator("#chat-query").input_value() for page in (first, second)
        ) == sorted(["", rejected_question])
    finally:
        preparation_gate.release.set()
        model_server.release.set()


@pytest.mark.browser
def test_unavailable_browser_coordination_preserves_input_without_sending_it(
    browser_page: Page,
) -> None:
    page = browser_page
    page.add_init_script(
        "Object.defineProperty(navigator, 'locks', {value: undefined});"
    )
    page.reload()
    expect(page.locator("#submit-btn")).to_be_enabled()
    requests: list[str] = []
    page.on("request", lambda request: requests.append(request.url))
    question = "Please remember my professional topic ALPHA."
    page.locator("#chat-query").fill(question)
    page.locator("#submit-btn").click()
    expect(page.locator("#chat-status")).to_have_text(
        "Session setup requires a browser with Web Locks over HTTPS or localhost."
    )
    expect(page.locator("#chat-query")).to_have_value(question)
    expect(page.locator("#response")).to_be_empty()
    assert not any(url.endswith("/chat") for url in requests)
    assert page.request.get(f"{page.url.rstrip('/')}/chat/conversation").json() == {
        "messages": []
    }
