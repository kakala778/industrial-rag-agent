import streamlit as st


def render_report(report):
    st.markdown("### Research Report")
    if not report:
        st.info("A report will appear here after the Agent run completes.")
        return
    with st.container(border=True):
        st.markdown(report)
