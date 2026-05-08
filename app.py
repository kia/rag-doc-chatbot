import traceback
import warnings
from pathlib import Path
from urllib.parse import quote

warnings.filterwarnings("ignore")

import streamlit as st
from rag_engine import RAGEngine, get_local_ollama_models, validate_openai_key, get_ai_models


# CSS to make the container stick to the top
st.markdown(
    """
    <style>
        .stApp [data-testid="stHeader"] {
            display: none; /* Optional: hide default header */
        }
        .css-1544f8n {
            padding-top: 0rem; /* Adjust top padding */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom top container
with st.container():
    st.markdown("### 📚 Dokumenten-Chatbot")
    st.markdown("Ask questions to you pdf documents – powered by RAG & LLM")
    st.divider()

st.set_page_config(page_title="Document Chatbot", page_icon="📚")

# Session State for Engine
if "engine" not in st.session_state:
    st.session_state.engine = None

if "engine_ready" not in st.session_state:
    st.session_state.engine_ready = False

if "engine_error" not in st.session_state:
    st.session_state.engine_error = None


def initialize_engine():
    try:
        selected_model = (
            st.session_state.selected_openai_model
            if st.session_state.use_openai_api
            else st.session_state.selected_local_model
        )
        st.session_state.selected_model = selected_model
        st.session_state.engine = RAGEngine(
            model_name=selected_model,
            use_openai_api=st.session_state.use_openai_api,
        )
        if st.session_state.use_openai_api:
            print("using open ai model: " + selected_model)
        else:
            print("using local model: " + selected_model)
        st.session_state.engine_ready = True
        st.session_state.engine_error = None
    except Exception as e:
        st.session_state.engine = None
        st.session_state.engine_ready = False
        st.session_state.engine_error = str(e)


def render_estimated_usage(estimate: dict, as_info: bool = False):
    text = (
        f"🔢 Estimated — prompt: {estimate['input_tokens']}, "
        f"completion: {estimate['estimated_output_tokens']}, "
        f"cost: ${float(estimate['estimated_total_cost']):.6f}"
    )
    if as_info:
        st.info(text)
    else:
        st.caption(text)


def render_token_usage(usage: dict):
    st.caption(
        f"💰 Tokens — prompt: {usage['prompt_tokens']}, "
        f"completion: {usage['completion_tokens']}, "
        f"total: {usage['total_tokens']}, "
        f"cost: ${usage['total_cost']:.6f}"
    )


def resolve_source_path(source: str) -> Path | None:
    if not source or source == "Unknown":
        return None
    raw_path = Path(source)
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path
    candidate = (Path(__file__).resolve().parent / raw_path).resolve()
    return candidate if candidate.exists() else None


def render_sources(sources: list, key_prefix: str):
    for i, source in enumerate(sources):
        with st.expander(f"📄 Source {i}: {Path(source['source'])}"):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Page:** {source['page']}")
                st.markdown("**Excerpt:**")
                st.info(f"\"{source['chunk_preview']}...\"")

            with col2:
                if Path(source["source"]).exists():
                    st.markdown(Path(source['source']))


# Sidebar
with st.sidebar:

    st.header("Settings")


    local_models = get_local_ollama_models()
    if "provider_choice" not in st.session_state:
        st.session_state.provider_choice = "OpenAI" if st.session_state.get("use_openai_api", False) else "Ollama"

    provider_choice = st.radio(
        "Model provider",
        options=["Ollama", "OpenAI"],
        key="provider_choice",
        horizontal=True,
    )

    st.session_state.use_openai_api = provider_choice == "OpenAI"

    if st.session_state.use_openai_api:
        open_ai_valid = validate_openai_key()
        if open_ai_valid:
            ai_models = get_ai_models()
            if not ai_models:
                ai_models = ["gpt-4.1-mini"]

            if (
                "selected_openai_model" not in st.session_state
                or st.session_state.selected_openai_model not in ai_models
            ):
                st.session_state.selected_openai_model = ai_models[0]

            st.selectbox(
                "OpenAI model",
                options=ai_models,
                key="selected_openai_model",
                help="Select open ai models.",
            )

    else:
        if not local_models:
            local_models = ["gemma:2b"]
        if "selected_local_model" not in st.session_state or st.session_state.selected_local_model not in local_models:
            st.session_state.selected_local_model = local_models[0]

        st.selectbox(
            "Ollama model",
            options=local_models,
            key="selected_local_model",
            help="Select one of your local Ollama models.",
        )

    if st.button("🚀 Initialize / Load vector store"):
        initialize_engine()
        st.rerun()

    if st.button("🔄 Rebuild vector store", disabled=not st.session_state.engine_ready):
        st.session_state.engine.reset_vectorstore()
        st.rerun()

    st.markdown("---")
    st.markdown("**Note:** Vector store is reused for the selected provider/model and rebuilt automatically only when documents change.")
    documents_dir = Path("documents")
    documents_dir.mkdir(parents=True, exist_ok=True)
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Uploaded files are stored in the local documents folder.",
    )
    if uploaded_files:
        saved_files = []
        for uploaded_file in uploaded_files:
            destination = documents_dir / uploaded_file.name
            file_bytes = uploaded_file.getbuffer().tobytes()
            if not destination.exists() or destination.read_bytes() != file_bytes:
                destination.write_bytes(file_bytes)
                saved_files.append(uploaded_file.name)
        if saved_files:
            st.success(f"Saved {len(saved_files)} file(s): {', '.join(saved_files)}")
            st.info(
                "New PDFs are detected automatically. The vector store will rebuild on next initialization if documents changed.")

    loaded_pdfs = sorted(documents_dir.glob("**/*.pdf"), key=lambda p: str(p).lower())

    if loaded_pdfs:
        with st.container():
            st.write("Loaded pdf files")
            st.dataframe(loaded_pdfs)
    else:
        st.caption("No PDF files loaded yet.")

    st.markdown("---")

if st.session_state.engine_ready:
    st.success(f"✅ Vector store ready! Selected Model: {st.session_state.selected_model}")
    st.caption(f"Store: `{st.session_state.engine.persist_dir}`")

else:
    st.info("ℹ️ Vector store is not initialized yet. Click 'Initialize / Load vector store' in the sidebar.")
    if st.session_state.engine_error:
        st.error(f"❌ Error while loading: {st.session_state.engine_error}")
        st.info("💡 Put PDF files in the 'documents' folder and make sure your API key is set in .env.")

# Chat-Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "pending_estimate" not in st.session_state:
    st.session_state.pending_estimate = None

if "opened_source" not in st.session_state:
    st.session_state.opened_source = None

# Vorherige Nachrichten anzeigen
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("estimated_usage"):
            render_estimated_usage(msg["estimated_usage"])
        if msg.get("token_usage"):
            render_token_usage(msg["token_usage"])
        if "sources" in msg:
            with st.expander("📎 Sources"):
                render_sources(msg["sources"], key_prefix=f"history_{msg_idx}_{msg['role']}")

# Nutzereingabe
if prompt := st.chat_input(
    "Ask about your documents...",
    disabled=not st.session_state.engine_ready or st.session_state.pending_prompt is not None,
):
    st.session_state.pending_prompt = prompt
    st.session_state.pending_estimate = None
    if st.session_state.use_openai_api:
        try:
            estimate = st.session_state.engine.estimate_query_cost(
                question=prompt,
                context="",
            )
            st.session_state.pending_estimate = estimate
        except Exception:
            st.session_state.pending_estimate = None
    st.rerun()

if st.session_state.pending_prompt:
    pending_prompt = st.session_state.pending_prompt
    model_name = st.session_state.engine.model_name

    with st.chat_message("user"):
        st.write(pending_prompt)

    with st.chat_message("assistant"):
        model_is_local = st.session_state.provider_choice == "Ollama"

        if not model_is_local:
            estimate = st.session_state.pending_estimate
            if estimate:
                render_estimated_usage(estimate, as_info=True)
                st.caption("Confirm to send this query.")
            elif st.session_state.use_openai_api:
                st.warning("Could not estimate cost for this query. You can still choose to send it.")
            confirm_col, cancel_col = st.columns(2)
            confirm_send = confirm_col.button("✅ Confirm send", key="confirm_send_query")
            cancel_send = cancel_col.button("❌ Cancel", key="cancel_send_query")
        else:
            confirm_send = True
            cancel_send = False

        if cancel_send:
            st.session_state.pending_prompt = None
            st.session_state.pending_estimate = None
            st.rerun()

        if confirm_send:
            estimate_for_message = st.session_state.pending_estimate
            st.session_state.messages.append({"role": "user", "content": pending_prompt})
            with st.spinner("🤔 Searching documents..."):
                try:
                    result = st.session_state.engine.query(pending_prompt)
                    print(result)
                    sources = []
                    for doc in result['sources']:
                        print(doc)
                        sources.append({
                            "file": doc["source"],
                            "page": doc.get("page", 0) + 1,  # 1-based page number
                            "snippet": doc["chunk_preview"][:100],  # First 100 chars
                            "full_text": doc["search_text"]
                        })
                    st.write("**" + model_name + "**: " + result["answer"])

                    if result.get("token_usage"):
                        render_token_usage(result["token_usage"])

                    if result["sources"]:
                        st.markdown("---")
                        st.markdown(f"**📎 {len(result['sources'])} source(s) found:**")
                        render_sources(result["sources"], key_prefix="latest_answer")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "token_usage": result.get("token_usage"),
                        "estimated_usage": estimate_for_message,
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    print(traceback.format_exc())
                finally:
                    st.session_state.pending_prompt = None
                    st.session_state.pending_estimate = None


if st.session_state.opened_source:
    opened_source = st.session_state.opened_source
    opened_path = Path(opened_source.get("path", ""))
    if opened_path.exists() and opened_path.suffix.lower() == ".pdf":
        with st.expander(f"📂 Opened source: {opened_source.get('label', opened_path.name)}", expanded=True):
            st.pdf(opened_path.read_bytes(), height=700)
            page_hint = opened_source.get("page")
            search_hint = opened_source.get("search_text")
            if page_hint or search_hint:
                hint_parts = []
                if page_hint:
                    hint_parts.append(f"page {page_hint}")
                if search_hint:
                    hint_parts.append(f"search: {search_hint}")
                st.caption("Hint: " + " • ".join(hint_parts))
    else:
        st.warning("Selected source file is no longer available.")
        st.session_state.opened_source = None



