import streamlit as st


def initialize_session_state():
    defaults = {
        "engine": None,
        "engine_ready": False,
        "engine_error": None,
        "engine_status": None,
        "messages": [],
        "pending_prompt": None,
        "pending_estimate": None,
        "opened_source": None,
        "score_threshold": 0.5,
        "k": 5
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
