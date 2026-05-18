import logging
logger = logging.getLogger("uvicorn.error")
import streamlit as st
from rag_doc_frontend.rag_client import RagClient


def initialize_engine():
    try:
        st.session_state.engine_status = "⏳ Initializing vector store..."
        selected_model = st.session_state.selected_model

        st.session_state.selected_model = selected_model
        st.session_state.engine = RagClient(
            model_name=selected_model,
            model_family=st.session_state.selected_model_family,
            status_callback=lambda message: st.session_state.__setitem__("engine_status", message),
            k=st.session_state.k
        )

        logger.debug(f"using {st.session_state.selected_model_family} model: " + selected_model)
        st.session_state.engine_ready = True
        st.session_state.engine_error = None
        if st.session_state.engine_status and not st.session_state.engine_status.startswith("⏳"):
            st.session_state.engine_status = "✅ Vector store ready"
    except Exception as e:
        st.session_state.engine = None
        st.session_state.engine_ready = False
        st.session_state.engine_error = str(e)
        st.session_state.engine_status = "❌ Vector store initialization failed"