import os
from dataclasses import dataclass, field
from typing import Literal

ProviderName = Literal["openai", "azure", "aws", "openai_compatible"]

PROVIDER_PRIORITY: tuple[ProviderName, ...] = (
    "openai",
    "azure",
    "aws",
    "openai_compatible",
)

DEFAULT_MODELS: dict[ProviderName, str] = {
    "openai": "gpt-4o-mini",
    "azure": "gpt-4o-mini",
    "aws": "anthropic.claude-3-haiku-20240307-v1:0",
    "openai_compatible": "gpt-4o-mini",
}


@dataclass
class LLMCredentials:
    """Credentials entered in the UI or read from environment variables."""

    openai_api_key: str = ""
    azure_api_key: str = ""
    azure_endpoint: str = ""
    azure_deployment: str = ""
    azure_api_version: str = "2024-02-15-preview"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "eu-west-1"
    compat_api_key: str = ""
    compat_base_url: str = ""

    @classmethod
    def from_env(cls) -> "LLMCredentials":
        openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY", "")
        return cls(
            openai_api_key=openai_key,
            azure_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
            azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            aws_region=os.getenv("AWS_DEFAULT_REGION", "eu-west-1"),
            compat_api_key=os.getenv("OPENAI_COMPAT_API_KEY", ""),
            compat_base_url=os.getenv("OPENAI_COMPAT_BASE_URL", ""),
        )

    def effective_openai_key(self) -> str:
        return (
            self.openai_api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("OPEN_AI_API_KEY", "")
        )

    def provider_is_configured(self, provider: ProviderName) -> bool:
        if provider == "openai":
            return bool(self.openai_api_key)
        if provider == "azure":
            return bool(self.azure_api_key and self.azure_endpoint and self.azure_deployment)
        if provider == "aws":
            return bool(self.aws_access_key_id and self.aws_secret_access_key)
        if provider == "openai_compatible":
            return bool(self.compat_api_key and self.compat_base_url)
        return False


@dataclass
class LLMConfig:
    """Runtime LLM settings for a single generation call."""

    provider: ProviderName
    model: str
    temperature: float = 0.2
    max_tokens: int | None = 1024
    credentials: LLMCredentials = field(default_factory=LLMCredentials)

    @property
    def provider_label(self) -> str:
        return {
            "openai": "OpenAI",
            "azure": "Azure OpenAI",
            "aws": "AWS Bedrock",
            "openai_compatible": "OpenAI-compatible API",
        }[self.provider]


def list_configured_providers(credentials: LLMCredentials | None = None) -> list[ProviderName]:
    """Return all providers that have complete credentials."""
    creds = credentials or LLMCredentials.from_env()
    return [provider for provider in PROVIDER_PRIORITY if creds.provider_is_configured(provider)]


def resolve_provider(
    credentials: LLMCredentials | None = None,
    preferred: ProviderName | str | None = None,
) -> ProviderName | None:
    """Return the preferred provider if configured, otherwise the first available."""
    creds = credentials or LLMCredentials.from_env()
    configured = list_configured_providers(creds)
    if not configured:
        return None
    if preferred and preferred in configured:
        return preferred  # type: ignore[return-value]
    return configured[0]


def provider_display_name(provider: ProviderName) -> str:
    return {
        "openai": "OpenAI",
        "azure": "Azure OpenAI",
        "aws": "AWS Bedrock",
        "openai_compatible": "OpenAI-compatible API",
    }[provider]


def default_model_for_provider(provider: ProviderName) -> str:
    return DEFAULT_MODELS[provider]
