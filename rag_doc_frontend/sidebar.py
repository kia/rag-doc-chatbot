from pathlib import Path

import streamlit as st
from rag_doc_frontend.rag_client import RagClient


class Sidebar:
    def __init__(self, initialize_engine_fn):
        self.initialize_engine_fn = initialize_engine_fn

    def render(self):
        with st.sidebar:
            st.markdown(
                """
                <style>
                div.stButton > button {
                    display: flex;
                    justify-content: flex-start;
                    width: 100%;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.header("Settings")
            self._render_model_provider_settings()
            self._render_engine_actions()
            self._render_documents_section()
            self._render_k_slider()
            st.markdown("---")

    def _render_model_provider_settings(self):

        provider_choice = st.radio(
            "Model provider",
            options=["Ollama", "OpenAI"],
            key="provider_choice",
            horizontal=True,
        )

        st.session_state.use_openai_api = provider_choice == "OpenAI"
        st.session_state.selected_model_family = "openai" if provider_choice == "OpenAI" else "ollama"

        if st.session_state.use_openai_api:
            ai_models = RagClient.get_open_ai_models()
            if not ai_models:
                ai_models = ["gpt-4.1-mini"]
            if (
                "selected_model" not in st.session_state
                or st.session_state.selected_model not in ai_models
            ):
                st.session_state.selected_model = ai_models[0]

            option = st.selectbox(
                "OpenAI model",
                options=ai_models,
                key="selected_openai_model",
                help="Select open ai models.",
            )
            st.session_state.selected_model = option
        else:
            local_models = RagClient.get_available_ollama_models()
            if not local_models:
                local_models = ["gemma:2b"]
            if "selected_local_model" not in st.session_state or st.session_state.selected_model not in local_models:
                st.session_state.selected_model = local_models[0]

            option = st.selectbox(
                "Ollama model",
                options=local_models,
                key="selected_local_model",
                help="Select one of your local Ollama models.",
            )
            st.session_state.selected_model = option

    def _render_engine_actions(self):
        if st.button("🚀 Refresh / Load vector store", use_container_width=True, ):
            self.initialize_engine_fn()
            st.rerun()

        if st.button("🔄 Rebuild vector store", disabled=not st.session_state.engine_ready, use_container_width=True):
            RagClient.reset_vectorstore(model_name=st.session_state.selected_model, model_family=st.session_state.selected_model_family, k=st.session_state.k)
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

    def _render_k_slider(self):
        st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=10,
            key="k",
            help="Number of documents to retrieve for each query",
        )




