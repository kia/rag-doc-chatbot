from pathlib import Path

import streamlit as st

from rag_engine import get_local_ollama_models, validate_openai_key, get_ai_models

class Sidebar:
    def __init__(self, initialize_engine_fn):
        self.initialize_engine_fn = initialize_engine_fn

    def render(self):
        with st.sidebar:
            st.header("Settings")
            self._render_model_provider_settings()
            self._render_engine_actions()
            self._render_documents_section()
            st.markdown("---")

    def _render_model_provider_settings(self):
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

    def _render_engine_actions(self):
        if st.button("🚀 Initialize / Load vector store"):
            self.initialize_engine_fn()
            st.rerun()

        if st.button("🔄 Rebuild vector store", disabled=not st.session_state.engine_ready):
            st.session_state.engine.reset_vectorstore()
            st.rerun()

        st.markdown("---")
        st.markdown("**Note:** Vector store is reused for the selected provider/model and rebuilt automatically only when documents change.")

    def _render_documents_section(self):
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

        loaded_pdfs = sorted(documents_dir.glob("**/*.pdf"), key=lambda p: str(p))

        if loaded_pdfs:
            file_names = [o.name for o in loaded_pdfs]
            with st.expander("Loaded pdf files", expanded=False):
                st.write("Loaded pdf files")
                st.table(file_names)
        else:
            st.caption("No PDF files loaded yet.")
