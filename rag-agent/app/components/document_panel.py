import streamlit as st


EXAMPLE_TASKS = (
    (
        "Operating Environment Requirements",
        "example_task_operating_environment",
        "Find the operating-environment requirements in the selected specification.",
    ),
    (
        "Equipment Configuration",
        "example_task_equipment_configuration",
        "Find a target equipment row's quantity and configuration in the structured schedule.",
    ),
    (
        "Check Unsupported Information",
        "example_task_unsupported_parameter",
        "Check whether [specify one protocol or security parameter] is supported by the selected specification.",
    ),
)


def _select_example_task(task):
    st.session_state["research_task"] = task
    st.session_state.pop("latest_research_run", None)


def render_document_panel():
    st.markdown("### Documents")
    with st.container(border=True):
        with st.form("research-inputs", clear_on_submit=False):
            uploaded_file = st.file_uploader(
                "Local industrial PDF", type=["pdf"], key="research_pdf"
            )
            task = st.text_area(
                "Research task",
                height=132,
                max_chars=4000,
                placeholder="Describe the information to locate in this document.",
                key="research_task",
            )
            submitted = st.form_submit_button(
                "Run research", type="primary", use_container_width=True
            )

        st.markdown("**Example tasks**")
        for label, key, task_text in EXAMPLE_TASKS:
            st.button(
                label,
                key=key,
                on_click=_select_example_task,
                args=(task_text,),
                use_container_width=True,
            )
        st.caption("Examples fill the task only. Replace bracketed text before running.")

    status_slot = st.empty()
    return uploaded_file, task, submitted, status_slot


def render_document_status(
    slot, *, document_name, task, result=None, error_kind=None, page_count=None
):
    with slot.container(border=True):
        st.markdown("**Document status**")
        if result is not None:
            st.success("Loaded")
            st.caption(f"Pages: {result.page_count}")
        elif error_kind == "document":
            st.error("PDF loading failed; see the run message for details.")
        elif error_kind == "agent":
            st.warning("PDF parsed; Agent execution failed.")
            if page_count is not None:
                st.caption(f"Pages: {page_count}")
        elif document_name:
            st.info("Selected · ready to run")
        else:
            st.caption("Upload one PDF to begin.")

        if document_name:
            st.caption("Document")
            st.text(document_name)
        st.caption("Current task")
        st.text(task.strip() if isinstance(task, str) and task.strip() else "No task entered.")
