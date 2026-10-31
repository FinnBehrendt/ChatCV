import json
import logging
from collections.abc import Iterator

import httpx
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from openai import AuthenticationError
from pydantic import SecretStr
from pytest import MonkeyPatch
from pytest_mock import MockerFixture

from app.config import AppConfig, EmbeddingConfig, LLMConfig
from app.modules import model_provider


@pytest.fixture
def settings(monkeypatch: MonkeyPatch) -> AppConfig:
    monkeypatch.setenv("INVITE_CODES", "{}")
    return AppConfig(
        _env_file=None,
        invite_codes={},
        OPENAI_API_KEY="test-official-openai-secret",
        LLM_API_KEY="test-chat-endpoint-secret",
        EMBEDDING_API_KEY="test-embedding-endpoint-secret",
        llm=LLMConfig(
            provider="openai_compatible",
            model="custom-chat-model",
            base_url="http://chat-server:8000/v1",
        ),
        embedding=EmbeddingConfig(
            provider="openai_compatible",
            model="custom-embedding-model",
            base_url="http://embedding-server:8001/v1",
            timeout=13,
        ),
    )


@pytest.fixture
def requests_sent(
    mocker: MockerFixture,
) -> Iterator[list[httpx.Request]]:
    """Use real SDK clients with an in-memory HTTP boundary, never a server."""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            payload = {
                "object": "list",
                "model": body["model"],
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        else:
            if body.get("stream"):
                chunk = {
                    "id": "test-completion",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "Example answer"},
                            "finish_reason": None,
                        }
                    ],
                }
                return httpx.Response(
                    200,
                    text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
                    headers={"content-type": "text/event-stream"},
                )
            payload = {
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Example answer"},
                        "finish_reason": "stop",
                    }
                ],
            }
        return httpx.Response(200, json=payload)

    real_chat = model_provider.ChatOpenAI
    real_embeddings = model_provider.OpenAIEmbeddings
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        mocker.patch.object(
            model_provider,
            "ChatOpenAI",
            side_effect=lambda **kwargs: real_chat(http_client=client, **kwargs),
        )
        mocker.patch.object(
            model_provider,
            "OpenAIEmbeddings",
            side_effect=lambda **kwargs: real_embeddings(http_client=client, **kwargs),
        )
        yield requests


def test_compatible_chat_uses_its_endpoint_and_key(
    settings: AppConfig, requests_sent: list[httpx.Request]
) -> None:
    model = model_provider.ModelProvider(
        settings, logging.getLogger(__name__)
    ).get_language_model()

    assert isinstance(model, BaseChatModel)
    assert model.invoke("Example question").content == "Example answer"
    request = requests_sent[0]
    assert str(request.url) == "http://chat-server:8000/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-chat-endpoint-secret"
    assert json.loads(request.content)["model"] == "custom-chat-model"
    assert request.extensions["timeout"]["read"] == settings.llm.timeout


def test_compatible_embeddings_send_text_not_openai_token_ids(
    settings: AppConfig, requests_sent: list[httpx.Request]
) -> None:
    model = model_provider.ModelProvider(
        settings, logging.getLogger(__name__)
    ).get_embedding_model()

    assert isinstance(model, Embeddings)
    assert model.embed_query("Example text") == [0.1, 0.2]
    request = requests_sent[0]
    assert str(request.url) == "http://embedding-server:8001/v1/embeddings"
    assert request.headers["authorization"] == "Bearer test-embedding-endpoint-secret"
    body = json.loads(request.content)
    assert body["model"] == "custom-embedding-model"
    assert body["input"] == ["Example text"]
    assert body["encoding_format"] == "float"
    assert request.extensions["timeout"]["read"] == settings.embedding.timeout


@pytest.mark.parametrize("role", ["chat", "embedding"])
@pytest.mark.parametrize("key", [None, SecretStr("")])
def test_local_compatible_endpoint_never_inherits_openai_credentials(
    settings: AppConfig,
    requests_sent: list[httpx.Request],
    role: str,
    key: SecretStr | None,
) -> None:
    settings.llm_api_key = key
    settings.embedding_api_key = key
    provider = model_provider.ModelProvider(settings, logging.getLogger(__name__))

    if role == "chat":
        provider.get_language_model().invoke("Example question")
    else:
        provider.get_embedding_model().embed_query("Example text")

    assert requests_sent[0].headers["authorization"] == "Bearer not-required"


@pytest.mark.parametrize("role", ["chat", "embedding"])
@pytest.mark.parametrize(
    "endpoint",
    [
        None,
        "",
        "ftp://server/v1",
        "https://user:secret@server/v1",
        "http://server/v1?key=secret",
        "http://server/v1#secret",
    ],
)
def test_compatible_provider_requires_an_explicit_safe_endpoint(
    settings: AppConfig, mocker: MockerFixture, role: str, endpoint: str | None
) -> None:
    chat = mocker.patch.object(model_provider, "ChatOpenAI")
    embedding = mocker.patch.object(model_provider, "OpenAIEmbeddings")
    provider = model_provider.ModelProvider(settings, logging.getLogger(__name__))

    with pytest.raises(ValueError, match="BASE_URL") as error:
        if role == "chat":
            settings.llm.base_url = endpoint
            provider.get_language_model()
        else:
            settings.embedding.base_url = endpoint
            provider.get_embedding_model()

    chat.assert_not_called()
    embedding.assert_not_called()
    assert "secret" not in str(error.value)


def test_models_can_omit_temperature_for_api_compatibility(
    settings: AppConfig, requests_sent: list[httpx.Request]
) -> None:
    settings.llm.temperature = None
    model_provider.ModelProvider(
        settings, logging.getLogger(__name__)
    ).get_language_model().invoke("Example")

    assert "temperature" not in json.loads(requests_sent[0].content)


def test_compatible_chat_supports_standard_text_streaming(
    settings: AppConfig, requests_sent: list[httpx.Request]
) -> None:
    model = model_provider.ModelProvider(
        settings, logging.getLogger(__name__)
    ).get_language_model()

    assert (
        "".join(str(chunk.content) for chunk in model.stream("Example"))
        == "Example answer"
    )
    body = json.loads(requests_sent[0].content)
    assert body["stream"] is True
    assert "stream_options" not in body


def test_provider_keys_are_redacted_in_configuration(
    settings: AppConfig, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        provider = model_provider.ModelProvider(settings, logging.getLogger(__name__))
        provider.get_language_model()
        provider.get_embedding_model()
    assert isinstance(settings.llm_api_key, SecretStr)
    assert isinstance(settings.embedding_api_key, SecretStr)
    for secret in ("test-chat-endpoint-secret", "test-embedding-endpoint-secret"):
        assert secret not in repr(settings)
        assert secret not in caplog.text


@pytest.mark.parametrize("role", ["chat", "embedding"])
def test_official_openai_does_not_use_a_custom_endpoint(
    settings: AppConfig,
    requests_sent: list[httpx.Request],
    monkeypatch: MonkeyPatch,
    role: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_BASE", "http://unrelated-server/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://unrelated-server/v1")
    settings.llm.provider = "openai"
    settings.embedding.provider = "openai"
    settings.llm_api_key = None
    settings.embedding_api_key = None
    provider = model_provider.ModelProvider(settings, logging.getLogger(__name__))

    if role == "chat":
        provider.get_language_model().invoke("Example question")
    else:
        provider.get_embedding_model().embed_query("Example text")

    assert requests_sent[0].url.host == "api.openai.com"
    assert (
        requests_sent[0].headers["authorization"]
        == "Bearer test-official-openai-secret"
    )


@pytest.mark.parametrize("role", ["chat", "embedding"])
def test_authentication_errors_propagate_without_falling_back(
    settings: AppConfig, mocker: MockerFixture, role: str
) -> None:
    requests: list[httpx.Request] = []

    def reject(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            401,
            json={
                "error": {"message": "Invalid API key", "type": "authentication_error"}
            },
        )

    real_chat = model_provider.ChatOpenAI
    real_embeddings = model_provider.OpenAIEmbeddings
    with httpx.Client(transport=httpx.MockTransport(reject)) as client:
        mocker.patch.object(
            model_provider,
            "ChatOpenAI",
            side_effect=lambda **kwargs: real_chat(http_client=client, **kwargs),
        )
        mocker.patch.object(
            model_provider,
            "OpenAIEmbeddings",
            side_effect=lambda **kwargs: real_embeddings(http_client=client, **kwargs),
        )
        provider = model_provider.ModelProvider(settings, logging.getLogger(__name__))

        with pytest.raises(AuthenticationError):
            if role == "chat":
                provider.get_language_model().invoke("Example")
            else:
                provider.get_embedding_model().embed_query("Example")

    assert len(requests) == 1
    assert requests[0].url.host in {"chat-server", "embedding-server"}
