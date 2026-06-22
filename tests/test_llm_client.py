from unittest.mock import MagicMock

import pytest

from eu_taxonomy_rag.llm.config import LLMCredentials, LLMConfig, resolve_provider
from eu_taxonomy_rag.llm.client import OpenAIChatClient, create_chat_client, create_chat_client_from_credentials


def test_resolve_provider_openai_first() -> None:
    creds = LLMCredentials(
        openai_api_key="sk-test",
        azure_api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com/",
        azure_deployment="gpt-4o",
    )
    assert resolve_provider(creds) == "openai"


def test_resolve_provider_azure_when_no_openai() -> None:
    creds = LLMCredentials(
        azure_api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com/",
        azure_deployment="gpt-4o",
    )
    assert resolve_provider(creds) == "azure"


def test_resolve_provider_prefers_explicit_choice() -> None:
    creds = LLMCredentials(
        openai_api_key="sk-test",
        azure_api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com/",
        azure_deployment="gpt-4o",
    )
    assert resolve_provider(creds, preferred="azure") == "azure"
    assert resolve_provider(creds, preferred="openai") == "openai"


def test_create_chat_client_openai() -> None:
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        credentials=LLMCredentials(openai_api_key="sk-test"),
    )
    client = create_chat_client(config)
    assert isinstance(client, OpenAIChatClient)


def test_openai_chat_client_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMessage:
        content = "Generated answer"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["model"] == "gpt-4o-mini"
            assert kwargs["messages"][0]["role"] == "system"
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    client = OpenAIChatClient(model="gpt-4o-mini", api_key="sk-test", temperature=0.1)
    answer = client.complete("system", "user")
    assert answer == "Generated answer"


def test_create_chat_client_from_credentials_raises_without_keys() -> None:
    with pytest.raises(ValueError, match="No LLM provider"):
        create_chat_client_from_credentials(LLMCredentials())
