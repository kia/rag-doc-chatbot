from pathlib import Path

import streamlit as st


def resolve_source_path(source: str) -> Path | None:
    if not source or source == "Unknown":
        return None
    raw_path = Path(source)
    # Check absolute path
    if raw_path.is_absolute():
        if raw_path.exists():
            return raw_path
        # If it's absolute but doesn't exist, maybe it was a relative path stored as absolute during indexing?
        # Try checking relative to project root (which is parent of rag_doc_frontend)
        root_relative = (Path(__file__).resolve().parent.parent / source).resolve()
        if root_relative.exists():
            return root_relative
            
    # Check relative to frontend dir
    candidate = (Path(__file__).resolve().parent / raw_path).resolve()
    if candidate.exists():
        return candidate
        
    # Check relative to project root
    root_candidate = (Path(__file__).resolve().parent.parent / raw_path).resolve()
    if root_candidate.exists():
        return root_candidate
        
    return None


def render_sources(sources: list, key_prefix: str):
    if not sources:
        return
    for i, source in enumerate(sources):
        rank = source.get("rank", i + 1)
        score = source.get("similarity_score")
        score_display = f" | Score: {score:.2f}" if score is not None else ""
        
        with st.expander(f"📄 Rank #{rank}: {Path(source['source']).name}{score_display}"):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Page:** {source['page']}")
                st.markdown("**Excerpt:**")
                st.info(f"\"{source['chunk_preview']}...\"")

            with col2:
                source_path = resolve_source_path(source.get("source", ""))
                if source_path:
                    try:
                        file_bytes = source_path.read_bytes()
                        st.download_button(
                            label="Download",
                            data=file_bytes,
                            file_name=source_path.name,
                            mime="application/pdf",
                            key=f"{key_prefix}_dl_{i}_{source_path.name}",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error loading file for download: {e}")
                else:
                    # Fallback: display the path if file not found to help debugging
                    st.caption(f"Path: {source.get('source')}")


def render_opened_source():
    if not st.session_state.opened_source:
        return

    opened_source = st.session_state.opened_source
    opened_path = Path(opened_source.get("path", ""))
    if opened_path.exists() and opened_path.suffix.lower() == ".pdf":
        with st.expander(f"📂 Opened source: {opened_source.get('label', opened_path.name)}", expanded=True):
            st.pdf(opened_path.read_bytes(), height=700)
            
            # Add download button in opened view too
            try:
                file_bytes = opened_path.read_bytes()
                st.download_button(
                    label="Download PDF",
                    data=file_bytes,
                    file_name=opened_path.name,
                    mime="application/pdf",
                    key=f"opened_dl_{opened_path.name}",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error loading file for download: {e}")

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