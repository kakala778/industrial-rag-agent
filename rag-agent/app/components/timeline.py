import streamlit as st


_SUCCESS = {"ok", "no_evidence"}
_FAILURE = {"error", "timeout", "invalid_action", "invalid_scope",
            "invalid_evidence_id", "budget_exceeded"}


def _summary(action, value):
    if isinstance(value, dict):
        statuses = value.get("scope_statuses")
        if isinstance(statuses, list):
            labels = [
                f"{row.get('scope')}: {row.get('status')}"
                for row in statuses
                if isinstance(row, dict)
                and isinstance(row.get("scope"), str)
                and isinstance(row.get("status"), str)
            ]
            if labels:
                return " · ".join(labels)
        count = value.get("result_count")
        if type(count) is int and count >= 0:
            noun = "candidate" if action == "SEARCH" else "evidence record"
            return f"{count} {noun}{'' if count == 1 else 's'} returned."
        terminal = value.get("terminal_status")
        if isinstance(terminal, str):
            return f"Stopped with status: {terminal}."
    if isinstance(value, str):
        return value[:180]
    return ""


def render_timeline(state):
    st.markdown("### Agent Timeline")
    st.caption("Observed Agent actions and terminal status from this run.")
    if state is None:
        st.info("The Agent action trace will appear here after a run.")
        return
    if not state.trace:
        st.info("The Agent returned no trace entries.")
        return

    for entry in state.trace:
        action = entry.get("action", "STEP")
        status = entry.get("tool_status", "unknown")
        marker = "✓" if status in _SUCCESS else "!" if status in _FAILURE else "·"
        with st.container(border=True):
            st.markdown(f"{marker} **{action}** · `{status}`")
            detail = _summary(action, entry.get("observation_summary"))
            if detail:
                st.caption(detail)

    st.caption(f"Terminal status: {state.status}")
