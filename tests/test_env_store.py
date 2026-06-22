from pathlib import Path
import os

import pytest

from eu_taxonomy_rag.llm.env_store import (
    apply_env_file_to_os,
    apply_session_defaults,
    apply_chatbot_env_to_session,
    coerce_chatbot_session_types,
    env_updates_from_session,
    parse_env_file,
    persisted_credential_key,
    read_persisted_credential,
    reload_env_into_memory,
    session_values_from_env,
    store_persisted_credentials,
    sync_persisted_credentials_from_env,
    write_env_file,
)


def test_parse_env_file_handles_comments_and_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        'OPENAI_API_KEY="sk-test"\n'
        "EU_TAXONOMY_LLM_TEMPERATURE=0.35\n",
        encoding="utf-8",
    )

    parsed = parse_env_file(env_file)
    assert parsed["OPENAI_API_KEY"] == "sk-test"
    assert parsed["EU_TAXONOMY_LLM_TEMPERATURE"] == "0.35"


def test_session_values_from_env_maps_chatbot_defaults(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=sk-test\n"
        "EU_TAXONOMY_LLM_MODEL=gpt-4o\n"
        "EU_TAXONOMY_CHAT_TOP_K=7\n"
        "EU_TAXONOMY_LLM_PROVIDER=openai\n",
        encoding="utf-8",
    )

    values = session_values_from_env(parse_env_file(env_file))
    assert values["llm_openai_api_key"] == "sk-test"
    assert values["llm_model"] == "gpt-4o"
    assert values["chat_top_k"] == 7
    assert values["chat_llm_provider"] == "openai"


def test_write_env_file_merges_and_applies_to_os(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_KEY=keep-me\n", encoding="utf-8")
    monkeypatch.setattr("eu_taxonomy_rag.llm.env_store.DEFAULT_ENV_PATH", env_file)

    path = write_env_file({"OPENAI_API_KEY": "sk-new", "EU_TAXONOMY_CHAT_TOP_K": "6"}, path=env_file)
    assert path == env_file

    parsed = parse_env_file(env_file)
    assert parsed["EXISTING_KEY"] == "keep-me"
    assert parsed["OPENAI_API_KEY"] == "sk-new"
    assert parsed["EU_TAXONOMY_CHAT_TOP_K"] == "6"
    assert apply_env_file_to_os(env_file)["OPENAI_API_KEY"] == "sk-new"


def test_env_updates_from_session() -> None:
    state = {
        persisted_credential_key("llm_openai_api_key"): "sk-test",
        "llm_model": "gpt-4o-mini",
        "llm_temperature": 0.25,
        "llm_max_tokens": 512,
        "chat_retrieval_method": "bm25",
        "chat_top_k": 4,
        "chat_candidate_k": 15,
        "chat_llm_provider": "openai",
    }

    updates = env_updates_from_session(state, credentials_only=False)
    assert updates["OPENAI_API_KEY"] == "sk-test"
    assert updates["EU_TAXONOMY_LLM_MODEL"] == "gpt-4o-mini"
    assert updates["EU_TAXONOMY_LLM_TEMPERATURE"] == "0.25"
    assert updates["EU_TAXONOMY_CHAT_TOP_K"] == "4"
    assert updates["EU_TAXONOMY_LLM_PROVIDER"] == "openai"


def test_store_persisted_credentials_uses_internal_keys_only() -> None:
    state: dict[str, str] = {}
    store_persisted_credentials(state, {"llm_openai_api_key": "sk-test"})

    assert state[persisted_credential_key("llm_openai_api_key")] == "sk-test"
    assert "llm_openai_api_key" not in state


def test_sync_persisted_credentials_from_env(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-from-file\n", encoding="utf-8")

    state: dict[str, str] = {}
    sync_persisted_credentials_from_env(state, path=env_file)

    assert read_persisted_credential(state, "llm_openai_api_key") == "sk-from-file"
    assert os.environ["OPENAI_API_KEY"] == "sk-from-file"


def test_reload_env_into_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=sk-reload\nEU_TAXONOMY_CHAT_TOP_K=8\n",
        encoding="utf-8",
    )

    values = reload_env_into_memory(env_file)
    assert values["llm_openai_api_key"] == "sk-reload"
    assert values["chat_top_k"] == 8
    assert os.environ["OPENAI_API_KEY"] == "sk-reload"
    assert os.environ["EU_TAXONOMY_CHAT_TOP_K"] == "8"


def test_apply_session_defaults_fills_missing_keys() -> None:
    state: dict[str, object] = {}
    apply_session_defaults(state)

    assert state["llm_model"] == "gpt-4o-mini"
    assert state["chat_top_k"] == 5
    assert state["llm_temperature"] == 0.2


def test_coerce_chatbot_session_types_repairs_invalid_values() -> None:
    state = {
        "llm_temperature": "0.35",
        "llm_max_tokens": "512",
        "chat_top_k": "7",
        "chat_candidate_k": "15",
        "chat_retrieval_method": "bm25",
        "llm_model": "gpt-4o",
    }
    coerce_chatbot_session_types(state)

    assert state["llm_temperature"] == pytest.approx(0.35)
    assert state["llm_max_tokens"] == 512
    assert state["chat_top_k"] == 7


def test_apply_chatbot_env_to_session(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "EU_TAXONOMY_LLM_MODEL=gpt-4o\n"
        "EU_TAXONOMY_CHAT_TOP_K=9\n",
        encoding="utf-8",
    )

    state: dict[str, object] = {"chat_top_k": 5, "llm_model": "gpt-4o-mini"}
    apply_chatbot_env_to_session(state, parse_env_file(env_file))

    assert state["llm_model"] == "gpt-4o"
    assert state["chat_top_k"] == 9
