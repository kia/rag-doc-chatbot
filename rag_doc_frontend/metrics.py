import streamlit as st

def render_quality_metrics(metrics: dict, key_prefix: str):
    if not metrics:
        return

    with st.expander("📊 Quality Metrics", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            similarity = metrics.get("similarity_score")
            st.metric(
                label="Retrieval Score",
                value=f"{similarity:.2f}" if isinstance(similarity, (int, float)) else "n/a",
                help="Average retrieval score of retrieved chunks (if available)",
            )

        with col2:
            st.metric(
                label="Chunks Retrieved",
                value=metrics.get("num_chunks", 0),
                help="Number of document chunks used",
            )

        with col3:
            st.metric(
                label="Latency",
                value=f"{metrics.get('latency_ms', 0)} ms",
                help="Total response time",
            )

        st.caption(f"Token Count: {metrics.get('token_count', 0)}")
        st.caption(f"Timestamp: {metrics.get('timestamp', 'n/a')}")