"""Local reviewer UI over the existing evidence research Agent."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.document_panel import (
    render_document_panel,
    render_document_status,
)
from app.components.evidence import render_evidence_panel
from app.components.report import render_report
from app.components.timeline import render_timeline
from app.runner import (
    AgentExecutionError,
    DocumentLoadError,
    run_uploaded_pdf_research,
)


def _document_name(uploaded_file):
    if uploaded_file is None:
        return None
    name = getattr(uploaded_file, "name", "uploaded.pdf")
    name = str(name).replace("\\", "/").rsplit("/", 1)[-1].strip()
    return name[:200] or "uploaded.pdf"


st.set_page_config(
    page_title="Industrial RAG Evidence Research Agent",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Industrial RAG Evidence Research Agent")
st.caption(
    "Research one local industrial document through bounded Agent actions and host-validated evidence references."
)
st.markdown(
    "**One PDF per run** · SEARCH / LOOKUP / CLARIFY / FINISH · "
    "evidence citations · Markdown research report"
)
st.info(
    "This demo makes source locations inspectable. A citation identifies where "
    "content came from; it does not prove relevance or engineering correctness."
)

left, center, right = st.columns([1.0, 1.35, 1.25], gap="medium")
with left:
    uploaded_file, task, submitted, status_slot = render_document_panel()
with center:
    timeline_slot = st.empty()

run_record = None
if submitted:
    st.session_state.pop("latest_research_run", None)
    if uploaded_file is None:
        run_record = {"error_kind": "input", "error": "Select one local PDF before running research."}
    elif not isinstance(task, str) or not task.strip():
        run_record = {"error_kind": "input", "error": "Enter a nonempty research task."}
    elif len(task) > 4000:
        run_record = {"error_kind": "input", "error": "The task must be at most 4000 characters."}
    else:
        with center:
            timeline_slot.info("The bounded Agent is running. Its captured action trace will appear when the run returns.")
            with st.spinner("Running evidence research…"):
                try:
                    result = run_uploaded_pdf_research(uploaded_file.getvalue(), task)
                    run_record = {"result": result}
                except DocumentLoadError as exc:
                    run_record = {"error_kind": "document", "error": str(exc)}
                except AgentExecutionError as exc:
                    run_record = {
                        "error_kind": "agent",
                        "error": str(exc),
                        "page_count": exc.page_count,
                    }
                except Exception as exc:
                    run_record = {
                        "error_kind": "agent",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
    st.session_state["latest_research_run"] = run_record
else:
    run_record = st.session_state.get("latest_research_run")

result = run_record.get("result") if isinstance(run_record, dict) else None
error = run_record.get("error") if isinstance(run_record, dict) else None
name = _document_name(uploaded_file)
render_document_status(
    status_slot,
    document_name=name,
    task=task,
    result=result,
    error_kind=run_record.get("error_kind") if isinstance(run_record, dict) else None,
    page_count=run_record.get("page_count") if isinstance(run_record, dict) else None,
)

with center:
    timeline_slot.empty()
    if error:
        labels = {
            "input": "Invalid task or input",
            "document": "PDF loading failed",
            "agent": "Agent execution failed",
        }
        label = labels.get(run_record.get("error_kind"), "Research run failed")
        st.error(label)
        st.text(error)
    elif result is not None and result.state.status in {
        "invalid_action", "invalid_scope", "timeout", "tool_error"
    }:
        st.error(f"Agent stopped with `{result.state.status}`.")
        st.text(result.state.answer)
    elif result is not None and result.state.status != "finished":
        st.info(f"Agent ended with `{result.state.status}`.")
        st.text(result.state.answer)
    render_timeline(result.state if result is not None else None)

with right:
    render_evidence_panel(result.state if result is not None else None)

st.divider()
render_report(result.report if result is not None else None)
