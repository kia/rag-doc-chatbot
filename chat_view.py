import traceback

import streamlit as st

from source_view import render_sources
from token_usage import TokenUsage


class ChatView:
    def __init__(self):
        self.token_usage = TokenUsage()

    def render(self):
        self._render_message_history()
        self._handle_prompt_input()
        self._handle_pending_prompt()

    def _render_message_history(self):
        for msg_idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("estimated_usage"):
                    self.token_usage.render_estimated_usage(msg["estimated_usage"])
                if msg.get("token_usage"):
                    self.token_usage.render_token_usage(msg["token_usage"])
                if "sources" in msg:
                    with st.expander("📎 Sources"):
                        render_sources(msg["sources"], key_prefix=f"history_{msg_idx}_{msg['role']}")

    def _handle_prompt_input(self):
        prompt = st.chat_input(
            "Ask about your documents...",
            disabled=not st.session_state.engine_ready or st.session_state.pending_prompt is not None,
        )
        if not prompt:
            return

        st.session_state.pending_prompt = prompt
        st.session_state.pending_estimate = None

        if st.session_state.use_openai_api:
            try:
                estimate = st.session_state.engine.estimate_query_cost(
                    question=prompt,
                    context="",
                )
                st.session_state.pending_estimate = estimate
            except Exception:
                st.session_state.pending_estimate = None

        st.rerun()

    def _handle_pending_prompt(self):
        if not st.session_state.pending_prompt:
            return

        pending_prompt = st.session_state.pending_prompt
        model_name = st.session_state.engine.model_name

        with st.chat_message("user"):
            st.write(pending_prompt)

        with st.chat_message("assistant"):
            model_is_local = st.session_state.provider_choice == "Ollama"

            if not model_is_local:
                estimate = st.session_state.pending_estimate
                if estimate:
                    self.token_usage.render_estimated_usage(estimate, as_info=True)
                    st.caption("Confirm to send this query.")
                elif st.session_state.use_openai_api:
                    st.warning("Could not estimate cost for this query. You can still choose to send it.")
                confirm_col, cancel_col = st.columns(2)
                confirm_send = confirm_col.button("✅ Confirm send", key="confirm_send_query")
                cancel_send = cancel_col.button("❌ Cancel", key="cancel_send_query")
            else:
                confirm_send = True
                cancel_send = False

            if cancel_send:
                st.session_state.pending_prompt = None
                st.session_state.pending_estimate = None
                st.rerun()

            if confirm_send:
                estimate_for_message = st.session_state.pending_estimate
                st.session_state.messages.append({"role": "user", "content": pending_prompt})
                with st.spinner("🤔 Searching documents..."):
                    try:
                        result = st.session_state.engine.query(pending_prompt)
                        st.write("**" + model_name + "**: " + result["answer"])

                        if result.get("token_usage"):
                            self.token_usage.render_token_usage(result["token_usage"])

                        if result["sources"]:
                            st.markdown("---")
                            st.markdown(f"**📎 {len(result['sources'])} source(s) found:**")
                            render_sources(result["sources"], key_prefix="latest_answer")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                            "token_usage": result.get("token_usage"),
                            "estimated_usage": estimate_for_message,
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        print(traceback.format_exc())
                    finally:
                        st.session_state.pending_prompt = None
                        st.session_state.pending_estimate = None