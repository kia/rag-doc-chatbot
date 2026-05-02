import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="Dokumenten-Chatbot", page_icon="📚")

st.title("📚 Dokumenten-Chatbot")
st.markdown("Stelle Fragen zu deinen PDF-Dokumenten – powered by RAG & LLM")

# Session State für Engine
if "engine" not in st.session_state:
    try:
        st.session_state.engine = RAGEngine()
        st.success("✅ Vectorstore geladen / erstellt!")
    except Exception as e:
        st.error(f"❌ Fehler beim Laden: {str(e)}")
        st.info("💡 Lege PDF-Dateien in den 'documents' Ordner und stelle sicher, dass dein API-Key in .env steht.")
        st.stop()

# Sidebar
with st.sidebar:
    st.header("Einstellungen")
    if st.button("🔄 Vectorstore neu erstellen"):
        st.session_state.engine.reset_vectorstore()
        st.rerun()

    st.markdown("---")
    st.markdown("**Hinweis:** Neue PDFs erfordern ein Neuladen des Vectorstores.")

# Chat-Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Vorherige Nachrichten anzeigen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg:
            with st.expander("📎 Quellen"):
                for source in msg["sources"]:
                    st.write(f"- {source}")

# Nutzereingabe
if prompt := st.chat_input("Stelle eine Frage zu deinen Dokumenten..."):
    # User-Nachricht anzeigen
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Antwort generieren
    with st.chat_message("assistant"):
        with st.spinner("🤔 Suche in Dokumenten..."):
            try:
                result = st.session_state.engine.query(prompt)
                st.write(result["answer"])

                if result["sources"]:
                    with st.expander("📎 Quellen"):
                        for source in result["sources"]:
                            st.write(f"- {source}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
            except Exception as e:
                st.error(f"❌ Fehler: {str(e)}")