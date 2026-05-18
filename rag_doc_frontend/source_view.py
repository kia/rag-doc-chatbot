from pathlib import Path

import streamlit as st


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
                source_path = resolve_source_path(source.get("source", ""))
                if source_path:
                    st.markdown(source_path)


def render_opened_source():
    if not st.session_state.opened_source:
        return

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