import logging
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch
from pytest_mock import MockerFixture

from app.config import AppConfig, EmbeddingConfig, LLMConfig, OllamaConfig
from app.modules import model_provider


@pytest.fixture
def settings(monkeypatch: MonkeyPatch) -> AppConfig:
    monkeypatch.setenv("INVITE_CODES", "{}")
    return AppConfig(
        _env_file=None,
        invite_codes={},
        llm=LLMConfig(provider="ollama", model="test-chat", temperature=0.2),
        embedding=EmbeddingConfig(provider="ollama", model="test-embedding"),
        ollama=OllamaConfig(endpoint="http://model-server:11434", timeout=17),
    )


@pytest.fixture
def factories(mocker: MockerFixture) -> dict[str, MagicMock]:
    return {
        name: mocker.patch.object(model_provider, name)
        for name in ("ChatOllama", "ChatOpenAI", "OllamaEmbeddings", "OpenAIEmbeddings")
    }


def test_ollama_chat_receives_endpoint_and_timeout(
    settings: AppConfig, factories: dict[str, MagicMock]
) -> None:
    provider = model_provider.ModelProvider(settings, logging.getLogger(__name__))

    result = provider.get_language_model()

    assert result is factories["ChatOllama"].return_value
    factories["ChatOllama"].assert_called_once_with(
        model="test-chat",
        temperature=0.2,
        base_url="http://model-server:11434",
        client_kwargs={"timeout": 17},
    )
    factories["ChatOpenAI"].assert_not_called()


def test_ollama_embeddings_use_provider_not_model_name(
    settings: AppConfig, factories: dict[str, MagicMock]
) -> None:
    provider = model_provider.ModelProvider(settings, logging.getLogger(__name__))

    result = provider.get_embedding_model()

    assert result is factories["OllamaEmbeddings"].return_value
    factories["OllamaEmbeddings"].assert_called_once_with(
        model="test-embedding",
        base_url="http://model-server:11434",
        client_kwargs={"timeout": 17},
    )
    factories["OpenAIEmbeddings"].assert_not_called()


@pytest.mark.parametrize("chat_provider", ["openai", "ollama"])
@pytest.mark.parametrize("embedding_provider", ["openai", "ollama"])
def test_chat_and_embedding_providers_are_independent(
    settings: AppConfig,
    factories: dict[str, MagicMock],
    chat_provider: str,
    embedding_provider: str,
) -> None:
    settings.llm.provider = chat_provider
    settings.embedding.provider = embedding_provider
    provider = model_provider.ModelProvider(settings, logging.getLogger(__name__))

    chat = provider.get_language_model()
    embedding = provider.get_embedding_model()

    chat_factory = "ChatOllama" if chat_provider == "ollama" else "ChatOpenAI"
    embedding_factory = (
        "OllamaEmbeddings" if embedding_provider == "ollama" else "OpenAIEmbeddings"
    )
    assert chat is factories[chat_factory].return_value
    assert embedding is factories[embedding_factory].return_value


@pytest.mark.parametrize("role", ["chat", "embedding"])
@pytest.mark.parametrize("unsupported", ["unknown", "not-ollama", ""])
def test_unknown_provider_is_rejected_without_creating_a_client(
    settings: AppConfig,
    factories: dict[str, MagicMock],
    role: str,
    unsupported: str,
) -> None:
    provider = model_provider.ModelProvider(settings, logging.getLogger(__name__))
    if role == "chat":
        settings.llm.provider = unsupported
    else:
        settings.embedding.provider = unsupported

    with pytest.raises(ValueError, match="Unsupported.*provider"):
        if role == "chat":
            provider.get_language_model()
        else:
            provider.get_embedding_model()

    for factory in factories.values():
        factory.assert_not_called()


def test_openai_embedding_selection_does_not_depend_on_model_name(
    settings: AppConfig, factories: dict[str, MagicMock]
) -> None:
    settings.embedding.provider = "openai"
    settings.embedding.model = "custom-model-containing-ollama"
    provider = model_provider.ModelProvider(settings, logging.getLogger(__name__))

    assert provider.get_embedding_model() is factories["OpenAIEmbeddings"].return_value
    factories["OllamaEmbeddings"].assert_not_called()


def test_actual_ollama_clients_accept_configuration_without_an_api_key(
    settings: AppConfig, monkeypatch: MonkeyPatch
) -> None:
    # Real SDK constructors validate arguments but do not contact the server.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings.openai_api_key = None
    provider = model_provider.ModelProvider(settings, logging.getLogger(__name__))

    chat = provider.get_language_model()
    embedding = provider.get_embedding_model()

    assert isinstance(chat, model_provider.ChatOllama)
    assert isinstance(embedding, model_provider.OllamaEmbeddings)
    assert chat.base_url == settings.ollama.endpoint
    assert embedding.base_url == settings.ollama.endpoint
    assert chat.client_kwargs == {"timeout": settings.ollama.timeout}
    assert embedding.client_kwargs == {"timeout": settings.ollama.timeout}
