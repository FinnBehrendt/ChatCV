"""Conversation behavior through HTTP, without external models or private data."""

import json
import re
import socket
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient
from playwright.sync_api import Page, expect, sync_playwright
from pypdf import PdfWriter
from pytest import FixtureRequest, MonkeyPatch
from pytest_mock import MockerFixture

import main
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


class ModelServer:
    """Control an external answer's timing without replacing application logic."""

    def __init__(self) -> None:
        self.pause_next_answer = False
        self.started = Event()
        self.release = Event()

    def respond(self, request: httpx.Request) -> httpx.Response:
        if self.pause_next_answer and request.url.path == "/v1/chat/completions":
            messages = json.loads(request.content)["messages"]
            if any(message["role"] == "system" for message in messages):
                self.pause_next_answer = False
                self.started.set()
                if not self.release.wait(timeout=10):
                    raise AssertionError("Test did not release the model response")
        return respond_as_model(request)


@pytest.fixture
def model_server() -> ModelServer:
    return ModelServer()


@pytest.fixture
def visitors(
    monkeypatch: MonkeyPatch,
    mocker: MockerFixture,
    tmp_path: Path,
    model_server: ModelServer,
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
    monkeypatch.setattr(main, "config", settings)
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
                    page = browser.new_page()

                    page.route(
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
