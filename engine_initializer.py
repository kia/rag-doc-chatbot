import streamlit as st

from rag_engine import RAGEngine


def initialize_engine():
    try:
        st.session_state.engine_status = "⏳ Initializing vector store..."
        selected_model = (
            st.session_state.selected_openai_model
            if st.session_state.use_openai_api
            else st.session_state.selected_local_model
        )
        st.session_state.selected_model = selected_model
        st.session_state.engine = RAGEngine(
            model_name=selected_model,
            use_openai_api=st.session_state.use_openai_api,
            status_callback=lambda message: st.session_state.__setitem__("engine_status", message),
        )
        if st.session_state.use_openai_api:
            print("using open ai model: " + selected_model)
        else:
            print("using local model: " + selected_model)
        st.session_state.engine_ready = True
        st.session_state.engine_error = None
        if st.session_state.engine_status and not st.session_state.engine_status.startswith("⏳"):
            st.session_state.engine_status = "✅ Vector store ready"
    except Exception as e:
        st.session_state.engine = None
        st.session_state.engine_ready = False
        st.session_state.engine_error = str(e)
        st.session_state.engine_status = "❌ Vector store initialization failed"