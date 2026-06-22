"""Streamlit chatbot page — single-turn RAG Q&A."""

from __future__ import annotations

import streamlit as st

from eu_taxonomy_rag.llm.config import (
    LLMConfig,
    LLMCredentials,
    ProviderName,
    default_model_for_provider,
    list_configured_providers,
    provider_display_name,
    resolve_provider,
)
from eu_taxonomy_rag.llm.client import create_chat_client
from eu_taxonomy_rag.llm.env_store import (
    CREDENTIAL_ENV_VARS,
    SESSION_DEFAULTS,
    env_updates_from_session,
    read_persisted_credential,
    reload_env_into_memory,
    session_values_from_env,
    store_persisted_credentials,
    sync_persisted_credentials_from_env,
    write_env_file,
)
from eu_taxonomy_rag.pipelines.index_manager import DEFAULT_INDEX_DIR
from eu_taxonomy_rag.pipelines.rag_pipeline import generate_rag_answer
from eu_taxonomy_rag.retrieval.retrieval_methods import RetrievalMethod, available_retrieval_methods

INDEX_DIR = DEFAULT_INDEX_DIR
_SETTINGS_LOADED_KEY = "chatbot_settings_loaded"
_WIDGETS_SEEDED_KEY = "chatbot_widgets_seeded"

_SECRET_WIDGET_KEYS = {
    "llm_openai_api_key": "llm_openai_api_key_input",
    "llm_azure_api_key": "llm_azure_api_key_input",
    "llm_aws_access_key_id": "llm_aws_access_key_id_input",
    "llm_aws_secret_access_key": "llm_aws_secret_access_key_input",
    "llm_compat_api_key": "llm_compat_api_key_input",
}

_NON_SECRET_CREDENTIAL_KEYS = tuple(
    key for key in CREDENTIAL_ENV_VARS if key not in _SECRET_WIDGET_KEYS
)


def _sync_provider_selection() -> None:
    configured = list_configured_providers(_credentials_from_session())
    saved_provider = st.session_state.get("chat_llm_provider", "")
    if configured:
        if saved_provider not in configured:
            st.session_state["chat_llm_provider"] = configured[0]
    else:
        st.session_state["chat_llm_provider"] = ""


def _seed_widget_state_before_render(*, force: bool = False) -> None:
    """Apply `.env` values to widget keys once, or again after an explicit reload."""
    sync_persisted_credentials_from_env(st.session_state)

    if not force and st.session_state.get(_WIDGETS_SEEDED_KEY):
        _sync_provider_selection()
        return

    values = session_values_from_env()

    for key, default in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, default)

    for key, value in values.items():
        if key in _SECRET_WIDGET_KEYS:
            continue
        st.session_state[key] = value

    _sync_provider_selection()

    method_values = [method.value for method in available_retrieval_methods()]
    if st.session_state.get("chat_retrieval_method") not in method_values:
        st.session_state["chat_retrieval_method"] = SESSION_DEFAULTS["chat_retrieval_method"]

    st.session_state[_WIDGETS_SEEDED_KEY] = True


def reload_chatbot_settings_from_env() -> None:
    """Reload `.env` into `os.environ` and persisted memory, then refresh the UI."""
    reload_env_into_memory()
    sync_persisted_credentials_from_env(st.session_state)
    for widget_key in _SECRET_WIDGET_KEYS.values():
        st.session_state.pop(widget_key, None)
    _seed_widget_state_before_render(force=True)
    st.rerun()


def ensure_chatbot_settings_loaded() -> None:
    """Load persisted credentials from `.env` into internal session memory."""
    sync_persisted_credentials_from_env(st.session_state)

    for key, default in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, default)

    if not st.session_state.get(_SETTINGS_LOADED_KEY):
        values = session_values_from_env()
        store_persisted_credentials(
            st.session_state,
            {key: value for key, value in values.items() if key in CREDENTIAL_ENV_VARS},
        )
        st.session_state[_SETTINGS_LOADED_KEY] = True


def _init_session_state() -> None:
    ensure_chatbot_settings_loaded()


def _credentials_from_session() -> LLMCredentials:
    return LLMCredentials(
        openai_api_key=read_persisted_credential(st.session_state, "llm_openai_api_key"),
        azure_api_key=read_persisted_credential(st.session_state, "llm_azure_api_key"),
        azure_endpoint=read_persisted_credential(st.session_state, "llm_azure_endpoint"),
        azure_deployment=read_persisted_credential(st.session_state, "llm_azure_deployment"),
        azure_api_version=read_persisted_credential(st.session_state, "llm_azure_api_version")
        or "2024-02-15-preview",
        aws_access_key_id=read_persisted_credential(st.session_state, "llm_aws_access_key_id"),
        aws_secret_access_key=read_persisted_credential(st.session_state, "llm_aws_secret_access_key"),
        aws_region=read_persisted_credential(st.session_state, "llm_aws_region") or "eu-west-1",
        compat_api_key=read_persisted_credential(st.session_state, "llm_compat_api_key"),
        compat_base_url=read_persisted_credential(st.session_state, "llm_compat_base_url"),
    )


def _capture_secret_inputs() -> None:
    """Copy newly typed secrets from password widgets into persisted memory."""
    updates: dict[str, str] = {}
    for session_key, widget_key in _SECRET_WIDGET_KEYS.items():
        typed = str(st.session_state.get(widget_key, "")).strip()
        if typed:
            updates[session_key] = typed
    if updates:
        store_persisted_credentials(st.session_state, updates)


def _capture_non_secret_credentials() -> None:
    """Copy visible credential fields from widgets into persisted memory."""
    updates: dict[str, str] = {}
    for session_key in _NON_SECRET_CREDENTIAL_KEYS:
        typed = str(st.session_state.get(session_key, "")).strip()
        if typed:
            updates[session_key] = typed
    if updates:
        store_persisted_credentials(st.session_state, updates)


def _capture_all_credentials_to_persisted() -> None:
    _capture_secret_inputs()
    _capture_non_secret_credentials()


def _render_secret_input(label: str, session_key: str) -> None:
    widget_key = _SECRET_WIDGET_KEYS[session_key]
    configured = bool(read_persisted_credential(st.session_state, session_key))
    placeholder = "Configured — leave empty to keep current value" if configured else ""
    st.text_input(label, type="password", key=widget_key, placeholder=placeholder)


def _selected_provider(credentials: LLMCredentials) -> ProviderName | None:
    preferred = st.session_state.get("chat_llm_provider") or None
    return resolve_provider(credentials, preferred=preferred)


def _build_llm_config(credentials: LLMCredentials) -> LLMConfig:
    provider = _selected_provider(credentials)
    if provider is None:
        raise ValueError(
            "No LLM provider configured. Add credentials in the **LLM connection** tab "
            "or set environment variables."
        )
    return LLMConfig(
        provider=provider,
        model=st.session_state.get("llm_model") or default_model_for_provider(provider),
        temperature=float(st.session_state.get("llm_temperature", 0.2)),
        max_tokens=int(st.session_state.get("llm_max_tokens", 1024)),
        credentials=credentials,
    )


def _save_credentials_to_env() -> None:
    _capture_all_credentials_to_persisted()
    updates = env_updates_from_session(st.session_state, credentials_only=True)
    if not updates:
        st.warning("No credentials to save.")
        return

    write_env_file(updates)
    for widget_key in _SECRET_WIDGET_KEYS.values():
        st.session_state.pop(widget_key, None)
    st.toast("Credentials saved to `.env`.", icon="✅")
    st.rerun()


def _save_chatbot_defaults_to_env() -> None:
    _capture_all_credentials_to_persisted()
    updates = env_updates_from_session(st.session_state, credentials_only=False)
    write_env_file(updates)
    for widget_key in _SECRET_WIDGET_KEYS.values():
        st.session_state.pop(widget_key, None)
    st.toast("Chatbot defaults saved to `.env`.", icon="✅")
    st.rerun()


def _render_provider_dropdown(credentials: LLMCredentials) -> ProviderName | None:
    """Dropdown of configured LLM providers."""
    configured = list_configured_providers(credentials)
    if not configured:
        st.warning("Configure at least one LLM provider in the **LLM connection** tab first.")
        return None

    if st.session_state.get("chat_llm_provider") not in configured:
        st.session_state["chat_llm_provider"] = configured[0]

    st.selectbox(
        "LLM provider",
        options=configured,
        key="chat_llm_provider",
        format_func=provider_display_name,
        help="Choose which configured LLM provider to use. Save defaults to persist this choice.",
    )
    return st.session_state["chat_llm_provider"]


def render_connection_tab() -> None:
    st.subheader("LLM connection")
    st.caption(
        "Enter credentials for one or more providers. "
        "Choose the active provider in the **Parameters** tab."
    )

    _capture_secret_inputs()
    _render_secret_input("OpenAI API key", "llm_openai_api_key")

    st.markdown("**Azure OpenAI**")
    c1, c2 = st.columns(2)
    with c1:
        _render_secret_input("Azure API key", "llm_azure_api_key")
        st.text_input("Azure endpoint", key="llm_azure_endpoint", placeholder="https://….openai.azure.com/")
    with c2:
        st.text_input("Deployment name", key="llm_azure_deployment")
        st.text_input("API version", key="llm_azure_api_version")

    st.markdown("**AWS Bedrock**")
    c3, c4 = st.columns(2)
    with c3:
        _render_secret_input("AWS access key ID", "llm_aws_access_key_id")
        st.text_input("AWS region", key="llm_aws_region")
    with c4:
        _render_secret_input("AWS secret access key", "llm_aws_secret_access_key")

    st.markdown("**OpenAI-compatible API** (Ollama, vLLM, LiteLLM, …)")
    c5, c6 = st.columns(2)
    with c5:
        _render_secret_input("API key", "llm_compat_api_key")
    with c6:
        st.text_input("Base URL", key="llm_compat_base_url", placeholder="http://localhost:11434/v1")

    credentials = _credentials_from_session()
    configured = list_configured_providers(credentials)
    if configured:
        labels = ", ".join(provider_display_name(provider) for provider in configured)
        st.success(f"Configured providers: **{labels}**")
    else:
        st.warning("No provider configured yet.")

    c_save, c_reload = st.columns(2)
    with c_save:
        if st.button("Save credentials to .env", key="save_credentials_btn", use_container_width=True):
            _save_credentials_to_env()
    with c_reload:
        if st.button("Reload from .env", key="reload_credentials_btn", use_container_width=True):
            reload_chatbot_settings_from_env()


def render_parameters_tab(available_methods: list[RetrievalMethod]) -> None:
    st.subheader("Model & retrieval parameters")

    credentials = _credentials_from_session()
    active = _render_provider_dropdown(credentials)

    if active:
        st.caption(f"Model settings below apply to **{provider_display_name(active)}**.")

    st.text_input("Model name / deployment / model ID", key="llm_model")
    c1, c2 = st.columns(2)
    with c1:
        st.slider("Temperature", min_value=0.0, max_value=1.0, step=0.05, key="llm_temperature")
    with c2:
        st.number_input("Max tokens", min_value=128, max_value=4096, step=64, key="llm_max_tokens")

    st.divider()
    st.markdown("**Retrieval**")
    method_values = [method.value for method in available_methods]

    st.selectbox(
        "Retrieval method",
        options=method_values,
        key="chat_retrieval_method",
        format_func=lambda value: value.replace("_", " ").title(),
    )
    c3, c4 = st.columns(2)
    with c3:
        st.slider("Top-k chunks", min_value=1, max_value=10, key="chat_top_k")
    with c4:
        st.number_input("Candidate-k (hybrid)", min_value=5, max_value=50, key="chat_candidate_k")

    c_save, c_reload = st.columns(2)
    with c_save:
        if st.button("Save defaults to .env", key="save_defaults_btn", use_container_width=True):
            _save_chatbot_defaults_to_env()
    with c_reload:
        if st.button("Reload from .env", key="reload_defaults_btn", use_container_width=True):
            reload_chatbot_settings_from_env()


def render_chatbot_page(
    *,
    get_chunks,
    warm_indexes,
    method_label,
    filter_selected_methods,
) -> None:
    _init_session_state()
    _seed_widget_state_before_render()

    st.header("EU Taxonomy RAG Chatbot")
    st.caption("Single-turn Q&A — each question is answered from retrieved FAQ context only (no chat memory).")

    available_methods = available_retrieval_methods()
    tab_chat, tab_params, tab_conn = st.tabs(["Chat", "Parameters", "LLM connection"])

    with tab_conn:
        render_connection_tab()

    with tab_params:
        render_parameters_tab(list(available_methods))

    with tab_chat:
        _capture_secret_inputs()
        credentials = _credentials_from_session()
        active = _selected_provider(credentials)
        if active:
            st.caption(
                f"Using **{provider_display_name(active)}** · model `{st.session_state.get('llm_model', '')}` "
                f"(change in **Parameters**)"
            )
        else:
            st.warning("Configure at least one LLM provider in the **LLM connection** tab.")

        question = st.text_area(
            "Your question",
            placeholder="e.g. How should undertakings report Taxonomy-aligned CapEx?",
            height=120,
            key="chat_question",
        )

        if st.button("Ask", type="primary"):
            if not question.strip():
                st.warning("Enter a question.")
                return

            try:
                llm_config = _build_llm_config(credentials)
                client = create_chat_client(llm_config)
            except ValueError as exc:
                st.error(str(exc))
                return

            method_value = st.session_state["chat_retrieval_method"]
            methods = filter_selected_methods([method_value])
            if not methods:
                return

            warm_indexes((method_value,))
            chunks = get_chunks()

            with st.spinner(
                f"Retrieving context ({method_label(method_value)}) "
                f"and generating with {llm_config.provider_label}…"
            ):
                try:
                    result = generate_rag_answer(
                        question,
                        chunks,
                        client,
                        method=methods[0],
                        k=int(st.session_state["chat_top_k"]),
                        candidate_k=int(st.session_state["chat_candidate_k"]),
                        base_dir=INDEX_DIR,
                        build_indexes=False,
                    )
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")
                    return

            st.markdown("### Answer")
            st.write(result.answer)

            st.caption(
                f"Provider: {llm_config.provider_label} · model `{llm_config.model}` · "
                f"temperature {llm_config.temperature}"
            )

            with st.expander(f"Retrieved context ({len(result.chunk_ids)} chunks)", expanded=False):
                st.markdown(result.context)

            with st.expander("Retrieval details"):
                for item in result.retrieval.chunks:
                    st.markdown(
                        f"**#{item.rank}** `{item.chunk.chunk_id}` — score `{item.score:.4f}`  \n"
                        f"*{item.chunk.question[:150]}{'…' if len(item.chunk.question) > 150 else ''}*"
                    )
