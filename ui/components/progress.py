"""
Agent pipeline status display — shows each agent's phase with progress indicators.
"""

from __future__ import annotations

import streamlit as st

PHASE_ORDER = ["ingest", "sourcing", "assembly", "export", "qa", "done"]

AGENT_ICONS = {
    "ingest": "📥",
    "sourcing": "🌐",
    "assembly": "✂️",
    "export": "📤",
    "qa": "🔍",
    "done": "✅",
}

AGENT_LABELS = {
    "ingest": "Ingest & Analyse",
    "sourcing": "Source Assets",
    "assembly": "Assemble Edit",
    "export": "Export & Reframe",
    "qa": "Quality Check",
    "done": "Complete",
}


def render_pipeline_status(state: dict | None) -> None:
    """
    Render the pipeline status card.
    Shows each agent phase with: waiting / running / done / error status.
    """
    st.subheader("Pipeline Status")

    if not state:
        st.info("No active project. Start a new project or load an existing one.")
        return

    current_phase = state.get("current_phase", "init")
    errors = state.get("errors", [])
    phase_results = state.get("phase_results", {})
    agent_notes = state.get("agent_notes", {})

    current_idx = PHASE_ORDER.index(current_phase) if current_phase in PHASE_ORDER else -1

    for i, phase in enumerate(PHASE_ORDER):
        icon = AGENT_ICONS[phase]
        label = AGENT_LABELS[phase]

        if i < current_idx:
            # Completed
            note = agent_notes.get(phase, "")
            st.success(f"{icon} **{label}** — Done")
            if note:
                with st.expander("Agent note", expanded=False):
                    st.caption(note)

        elif i == current_idx:
            # Active
            agent_errors = [e for e in errors if f"[{phase}]" in e]
            if agent_errors:
                st.error(f"{icon} **{label}** — Error")
                for err in agent_errors:
                    st.caption(err)
            elif state.get("awaiting_human") and phase == current_phase:
                st.warning(f"{icon} **{label}** — Waiting for your input")
            else:
                st.info(f"{icon} **{label}** — Running...")
                st.progress(0.6)

        else:
            # Pending
            st.markdown(
                f"<div style='color: #888;'>{icon} {label} — Waiting</div>",
                unsafe_allow_html=True,
            )

    # QA score if available
    qa_result = phase_results.get("qa")
    if qa_result and isinstance(qa_result, dict):
        score = qa_result.get("score", 0)
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("QA Score", f"{score}/100")
        with col2:
            passed = qa_result.get("passed", False)
            st.metric("Status", "✅ Passed" if passed else "❌ Failed")
