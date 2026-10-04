import streamlit as st


_STATUS_TEXT = {
    "evidence_found": "Selected host-validated source records; this does not establish relevance or correctness.",
    "no_evidence_found": "The configured search returned no candidates; this does not establish document-wide absence.",
    "insufficient_scope": "Candidates were looked up, but none were selected.",
}


def render_evidence_panel(state):
    st.markdown("### Evidence Status")
    if state is None:
        st.info("Host-validated evidence references will appear here after a completed run.")
        return
    if state.status != "finished":
        st.info("The Agent did not finish. Any evidence in the report is marked preliminary.")
        return
    if not state.findings:
        st.info("The completed run contains no per-source outcomes.")
        return

    for finding in state.findings:
        scope = finding.get("scope", "unknown")
        status = finding.get("status", "unknown")
        with st.container(border=True):
            st.markdown(f"**Agent scope outcome:** `{status}`")
            st.caption(f"Source alias: `{scope}`")
            explanation = _STATUS_TEXT.get(status)
            if explanation:
                st.caption(explanation)
            references = finding.get("evidence", [])
            if references:
                st.caption("Host reference validation: passed; this does not establish relevance or correctness.")
                st.markdown("**Evidence Viewer**")
            for reference in references:
                if not isinstance(reference, dict):
                    continue
                evidence_id = reference.get("evidence_id", "unknown")
                page = reference.get("page")
                page_label = "unavailable" if page is None else str(page)
                block_type = reference.get("block_type", "unknown")
                block_index = reference.get("block_index", "unknown")
                st.markdown(f"**Citation:** `{evidence_id}`")
                st.caption(
                    f"Source: {reference.get('source', scope)} · Page: {page_label} · "
                    f"Block: {block_type}:{block_index}"
                )
                st.caption("Quote")
                st.text(reference.get("excerpt", ""))
