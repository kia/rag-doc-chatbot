import warnings

from chat_view import ChatView
from engine_initializer import initialize_engine
from session_state import initialize_session_state
from sidebar import Sidebar
from source_view import render_opened_source

warnings.filterwarnings("ignore")

import streamlit as st

st.set_page_config(page_title="Document Chatbot", page_icon="📚")

# Keep content as close to the top as possible
st.markdown(
    """
    <style>
        .stApp [data-testid="stHeader"] {
            display: none; /* Optional: hide default header */
        }
        .stApp [data-testid="stMainBlockContainer"] {
            padding-top: 0rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom top container
with st.container():
    st.markdown("### 📚 Dokumenten-Chatbot")
    st.markdown("Ask questions to your pdf documents – powered by RAG & LLM")
    st.divider()

initialize_session_state()

if st.session_state.engine_status:
    st.info(st.session_state.engine_status +" using **"+ st.session_state.selected_model+"**")
else:
    st.info("ℹ️ Vector store is not initialized yet. Click 'Initialize / Load vector store' in the sidebar.")
    if st.session_state.engine_error:
        st.error(f"❌ Error while loading: {st.session_state.engine_error}")
        st.info("💡 Put PDF files in the 'documents' folder and make sure your API key is set in .env.")

sidebar = Sidebar(initialize_engine)
sidebar.render()

chat_view = ChatView()
with st.container(height=600):
    chat_view.render()

render_opened_source()