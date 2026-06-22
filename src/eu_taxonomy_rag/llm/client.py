import os
from typing import Any, Protocol

from eu_taxonomy_rag.llm.config import LLMConfig, LLMCredentials, ProviderName, resolve_provider


class ChatClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class OpenAIChatClient:
    """OpenAI or OpenAI-compatible chat completion client."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = 1024,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

        if not self.api_key:
            raise ValueError("OpenAI API key is missing.")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client = OpenAI(**client_kwargs)
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens

        response = client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty LLM response.")
        return content.strip()


class AzureOpenAIChatClient:
    """Azure OpenAI chat completion client."""

    def __init__(
        self,
        deployment: str,
        api_key: str,
        azure_endpoint: str,
        api_version: str = "2024-02-15-preview",
        temperature: float = 0.2,
        max_tokens: int | None = 1024,
    ) -> None:
        self.deployment = deployment
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
        )
        request_kwargs: dict[str, Any] = {
            "model": self.deployment,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens

        response = client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty LLM response.")
        return content.strip()


class BedrockChatClient:
    """AWS Bedrock converse API (Claude / other chat models)."""

    def __init__(
        self,
        model_id: str,
        region: str,
        access_key_id: str,
        secret_access_key: str,
        temperature: float = 0.2,
        max_tokens: int | None = 1024,
    ) -> None:
        self.model_id = model_id
        self.region = region
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.temperature = temperature
        self.max_tokens = max_tokens or 1024

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError("Install boto3 to use AWS Bedrock: pip install boto3") from exc

        client = boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
        )
        response = client.converse(
            modelId=self.model_id,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={
                "temperature": self.temperature,
                "maxTokens": self.max_tokens,
            },
        )
        output = response["output"]["message"]["content"][0]["text"]
        if not output:
            raise ValueError("Empty LLM response.")
        return output.strip()


def create_chat_client(config: LLMConfig) -> ChatClient:
    """Build a chat client from provider configuration."""
    creds = config.credentials

    if config.provider == "openai":
        return OpenAIChatClient(
            model=config.model,
            api_key=creds.effective_openai_key(),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    if config.provider == "azure":
        api_key = creds.azure_api_key or os.getenv("AZURE_OPENAI_API_KEY", "")
        endpoint = creds.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        deployment = creds.azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "") or config.model
        api_version = creds.azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        return AzureOpenAIChatClient(
            deployment=deployment,
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    if config.provider == "aws":
        return BedrockChatClient(
            model_id=config.model,
            region=creds.aws_region or os.getenv("AWS_DEFAULT_REGION", "eu-west-1"),
            access_key_id=creds.aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID", ""),
            secret_access_key=creds.aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    if config.provider == "openai_compatible":
        return OpenAIChatClient(
            model=config.model,
            api_key=creds.compat_api_key or os.getenv("OPENAI_COMPAT_API_KEY", ""),
            base_url=creds.compat_base_url or os.getenv("OPENAI_COMPAT_BASE_URL", ""),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    raise ValueError(f"Unsupported provider: {config.provider}")


def create_chat_client_from_credentials(
    credentials: LLMCredentials | None = None,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = 1024,
    provider: ProviderName | None = None,
) -> ChatClient:
    """Create a client using the first configured provider (or an explicit one)."""
    creds = credentials or LLMCredentials.from_env()
    selected = provider or resolve_provider(creds)
    if selected is None:
        raise ValueError(
            "No LLM provider configured. Set credentials in the UI or environment variables."
        )

    from eu_taxonomy_rag.llm.config import default_model_for_provider

    config = LLMConfig(
        provider=selected,
        model=model or default_model_for_provider(selected),
        temperature=temperature,
        max_tokens=max_tokens,
        credentials=creds,
    )
    return create_chat_client(config)
