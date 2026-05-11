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
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
