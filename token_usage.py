import streamlit as st

class TokenUsage:
    def __init__(self):
        pass


    def render_estimated_usage(self, estimate: dict, as_info: bool = False):
        text = (
            f"🔢 Estimated — prompt: {estimate['input_tokens']}, "
            f"completion: {estimate['estimated_output_tokens']}, "
            f"cost: ${float(estimate['estimated_total_cost']):.6f}"
        )
        if as_info:
            st.info(text)
        else:
            st.caption(text)


    def render_token_usage(self, usage: dict):
        st.caption(
            f"💰 Tokens — prompt: {usage['prompt_tokens']}, "
            f"completion: {usage['completion_tokens']}, "
            f"total: {usage['total_tokens']}, "
            f"cost: ${usage['total_cost']:.6f}"
        )
