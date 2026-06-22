"""Read and write chatbot settings in the project-root `.env` file."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

DEFAULT_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"

# Legacy alias occasionally found in hand-written `.env` files.
OPENAI_KEY_ALIASES = ("OPENAI_API_KEY", "OPEN_AI_API_KEY")

CREDENTIAL_ENV_VARS: dict[str, str] = {
    "llm_openai_api_key": "OPENAI_API_KEY",
    "llm_azure_api_key": "AZURE_OPENAI_API_KEY",
    "llm_azure_endpoint": "AZURE_OPENAI_ENDPOINT",
    "llm_azure_deployment": "AZURE_OPENAI_DEPLOYMENT",
    "llm_azure_api_version": "AZURE_OPENAI_API_VERSION",
    "llm_aws_access_key_id": "AWS_ACCESS_KEY_ID",
    "llm_aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
    "llm_aws_region": "AWS_DEFAULT_REGION",
    "llm_compat_api_key": "OPENAI_COMPAT_API_KEY",
    "llm_compat_base_url": "OPENAI_COMPAT_BASE_URL",
}

CHATBOT_ENV_VARS: dict[str, str] = {
    "llm_model": "EU_TAXONOMY_LLM_MODEL",
    "llm_temperature": "EU_TAXONOMY_LLM_TEMPERATURE",
    "llm_max_tokens": "EU_TAXONOMY_LLM_MAX_TOKENS",
    "chat_retrieval_method": "EU_TAXONOMY_CHAT_RETRIEVAL_METHOD",
    "chat_top_k": "EU_TAXONOMY_CHAT_TOP_K",
    "chat_candidate_k": "EU_TAXONOMY_CHAT_CANDIDATE_K",
    "chat_llm_provider": "EU_TAXONOMY_LLM_PROVIDER",
}

SESSION_DEFAULTS: dict[str, Any] = {
    "llm_openai_api_key": "",
    "llm_azure_api_key": "",
    "llm_azure_endpoint": "",
    "llm_azure_deployment": "",
    "llm_azure_api_version": "2024-02-15-preview",
    "llm_aws_access_key_id": "",
    "llm_aws_secret_access_key": "",
    "llm_aws_region": "eu-west-1",
    "llm_compat_api_key": "",
    "llm_compat_base_url": "",
    "llm_model": "gpt-4o-mini",
    "llm_temperature": 0.2,
    "llm_max_tokens": 1024,
    "chat_retrieval_method": "hybrid_minilm",
    "chat_top_k": 5,
    "chat_candidate_k": 20,
    "chat_llm_provider": "",
}

_ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def parse_env_file(path: Path | None = None) -> dict[str, str]:
    """Parse a dotenv file into a flat key/value mapping."""
    env_path = path or DEFAULT_ENV_PATH
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_LINE_RE.match(raw_line)
        if not match:
            continue
        key, raw_value = match.group(1), match.group(2).strip()
        values[key] = _unquote(raw_value)
    return values


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _quote_if_needed(value: str) -> str:
    if not value:
        return ""
    if any(char.isspace() for char in value) or "#" in value:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def reload_env_into_memory(path: Path | None = None) -> dict[str, Any]:
    """Reload `.env` into `os.environ` and return mapped session-state values."""
    parsed = apply_env_file_to_os(path)
    return session_values_from_env(parsed)


def credential_session_keys() -> tuple[str, ...]:
    return tuple(CREDENTIAL_ENV_VARS.keys())


def persisted_credential_key(session_key: str) -> str:
    return f"_persisted_{session_key}"


def read_persisted_credential(session_state: Any, session_key: str) -> str:
    """Return a stored credential without relying on password widget state."""
    persisted = session_state.get(persisted_credential_key(session_key), "")
    if persisted:
        return str(persisted)
    return str(session_state.get(session_key, "") or "")


def store_persisted_credentials(session_state: Any, values: dict[str, Any]) -> None:
    """Persist credential values in internal session keys (not widget-bound keys)."""
    for session_key in CREDENTIAL_ENV_VARS:
        if session_key not in values:
            continue
        value = str(values[session_key]).strip()
        if value:
            session_state[persisted_credential_key(session_key)] = value


def sync_persisted_credentials_from_env(session_state: Any, path: Path | None = None) -> dict[str, str]:
    """Load credential secrets from `.env` into session memory and `os.environ`."""
    parsed = apply_env_file_to_os(path)
    credential_values = {
        session_key: value
        for session_key, value in session_values_from_env(parsed).items()
        if session_key in CREDENTIAL_ENV_VARS
    }
    store_persisted_credentials(session_state, credential_values)
    return parsed


def apply_env_file_to_os(path: Path | None = None) -> dict[str, str]:
    """Load `.env` values into `os.environ` and return the parsed mapping."""
    parsed = parse_env_file(path)
    for key, value in parsed.items():
        os.environ[key] = value

    openai_key = parsed.get("OPENAI_API_KEY") or parsed.get("OPEN_AI_API_KEY", "")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    return parsed


def session_values_from_env(parsed: dict[str, str] | None = None) -> dict[str, Any]:
    """Map `.env` keys to Streamlit session-state keys."""
    env = parsed if parsed is not None else parse_env_file()
    values: dict[str, Any] = {}

    for session_key, env_key in CREDENTIAL_ENV_VARS.items():
        if env_key == "OPENAI_API_KEY":
            raw = env.get("OPENAI_API_KEY") or env.get("OPEN_AI_API_KEY", "")
        else:
            raw = env.get(env_key, "")
        if raw:
            values[session_key] = raw

    for session_key, env_key in CHATBOT_ENV_VARS.items():
        raw = env.get(env_key, "")
        if not raw:
            continue
        default = SESSION_DEFAULTS[session_key]
        if isinstance(default, float):
            values[session_key] = float(raw)
        elif isinstance(default, int):
            values[session_key] = int(raw)
        else:
            values[session_key] = raw

    return values


def read_credential_for_env(session_state: Any, session_key: str) -> str:
    """Prefer the current widget value, then fall back to persisted memory."""
    widget_value = str(session_state.get(session_key, "") or "").strip()
    if widget_value:
        return widget_value
    return read_persisted_credential(session_state, session_key)


def env_updates_from_session(session_state: Any, *, credentials_only: bool = False) -> dict[str, str]:
    """Build `.env` updates from the current Streamlit session state."""
    mapping = CREDENTIAL_ENV_VARS if credentials_only else {**CREDENTIAL_ENV_VARS, **CHATBOT_ENV_VARS}
    updates: dict[str, str] = {}

    for session_key, env_key in mapping.items():
        if session_key in CREDENTIAL_ENV_VARS:
            value = read_credential_for_env(session_state, session_key)
        elif session_key in session_state:
            value = session_state[session_key]
        else:
            continue

        if value is None:
            continue
        if isinstance(value, float):
            text = f"{value:g}"
        else:
            text = str(value).strip()
        if text:
            updates[env_key] = text
        elif env_key in updates:
            updates[env_key] = ""

    return updates


def write_env_file(updates: dict[str, str], path: Path | None = None) -> Path:
    """Merge updates into the project `.env` file (preserving unrelated keys)."""
    env_path = path or DEFAULT_ENV_PATH
    lines: list[str] = []
    index_by_key: dict[str, int] = {}

    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            match = _ENV_LINE_RE.match(line)
            if match:
                index_by_key[match.group(1)] = index

    normalized = dict(updates)
    if "OPENAI_API_KEY" in normalized and "OPEN_AI_API_KEY" in index_by_key:
        legacy_index = index_by_key.pop("OPEN_AI_API_KEY")
        lines[legacy_index] = "# OPEN_AI_API_KEY= (migrated to OPENAI_API_KEY)"

    for key, value in normalized.items():
        if not value:
            if key in index_by_key:
                lines[index_by_key[key]] = f"# {key}="
                index_by_key.pop(key, None)
            continue

        rendered = f"{key}={_quote_if_needed(value)}"
        if key in index_by_key:
            lines[index_by_key[key]] = rendered
        else:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(rendered)
            index_by_key[key] = len(lines) - 1

    env_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content and not content.endswith("\n"):
        content += "\n"
    env_path.write_text(content, encoding="utf-8")

    apply_env_file_to_os(env_path)
    return env_path
