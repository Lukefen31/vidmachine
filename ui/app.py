"""
Streamlit UI — main app file.

Layout:
  Left sidebar:  Chat with Director agent
  Main area:     Timeline | Agent Status | Asset Browser
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VidMachine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports (keep startup fast)
# ─────────────────────────────────────────────────────────────────────────────
from ui.components.chat import render_chat, format_agent_message, format_user_message
from ui.components.timeline import render_timeline
from ui.components.asset_browser import render_asset_browser
from ui.components.progress import render_pipeline_status
from storage.project_store import store
from state.project_state import make_initial_state


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _init_session():
    defaults = {
        "project_state": None,        # Current ProjectState dict
        "chat_messages": [],          # UI message list (separate from LangGraph messages)
        "agent_running": False,       # True while graph is executing
        "project_id": None,
        "graph": None,                # Compiled LangGraph
        "thread_id": str(uuid.uuid4()),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()


# ─────────────────────────────────────────────────────────────────────────────
# Graph runner (non-blocking via threading)
# ─────────────────────────────────────────────────────────────────────────────
def _run_graph(user_message: str):
    """Submit a user message to the LangGraph in a background thread."""
    from langchain_core.messages import HumanMessage

    if st.session_state.graph is None:
        _load_graph()

    graph = st.session_state.graph
    state = st.session_state.project_state

    # Append human message
    state["user_intent"] = user_message
    state["messages"] = state.get("messages", []) + [HumanMessage(content=user_message)]
    state["awaiting_human"] = False

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    st.session_state.agent_running = True

    def _run():
        try:
            result = graph.invoke(state, config=config)
            st.session_state.project_state = result
            store.save_state(result)
            # Extract Director's response for chat UI
            notes = result.get("agent_notes", {})
            director_note = notes.get("director", "")
            if director_note:
                st.session_state.chat_messages.append(
                    format_agent_message("director", director_note)
                )
        except Exception as exc:
            st.session_state.chat_messages.append(
                format_agent_message("system", f"❌ Error: {exc}")
            )
        finally:
            st.session_state.agent_running = False
            st.rerun()

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def _load_graph():
    """Lazy-load the compiled LangGraph."""
    from graph.workflow import compiled_graph
    st.session_state.graph = compiled_graph


# ─────────────────────────────────────────────────────────────────────────────
# Project management
# ─────────────────────────────────────────────────────────────────────────────
def _new_project(title: str = "New Project"):
    project_id = uuid.uuid4().hex
    project_dir = str(store.ensure_project_dirs(project_id))
    state = make_initial_state(
        project_id=project_id,
        project_dir=project_dir,
        user_intent="",
    )
    state["blueprint"]["title"] = title
    store.save_state(state)
    st.session_state.project_state = state
    st.session_state.project_id = project_id
    st.session_state.chat_messages = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat_messages.append(
        format_agent_message(
            "director",
            f"🎬 Project **{title}** created. Drop your footage into:\n"
            f"`{project_dir}/assets/raw/`\n\n"
            "Then tell me what you'd like to create — e.g. *'Create a 30s FPV compilation with cinematic music'*",
        )
    )


def _load_project(project_id: str):
    state = store.load_state(project_id)
    if state:
        st.session_state.project_state = state
        st.session_state.project_id = project_id
        st.session_state.chat_messages = []


# ─────────────────────────────────────────────────────────────────────────────
# Message send handler
# ─────────────────────────────────────────────────────────────────────────────
def _on_send(message: str):
    st.session_state.chat_messages.append(format_user_message(message))
    if st.session_state.project_state is None:
        _new_project()
    _run_graph(message)
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — chat + project controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 VidMachine")
    st.caption("AI-Orchestrated Video Production")
    st.divider()

    # Project controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("＋ New Project", use_container_width=True):
            _new_project()
            st.rerun()
    with col2:
        projects = store.list_projects()
        if st.button("📂 Load", use_container_width=True, disabled=len(projects) == 0):
            st.session_state["show_project_picker"] = True

    if st.session_state.get("show_project_picker") and projects:
        chosen = st.selectbox(
            "Select project",
            options=[p["project_id"] for p in projects],
            format_func=lambda pid: next((p["title"] for p in projects if p["project_id"] == pid), pid),
        )
        if st.button("Load selected"):
            _load_project(chosen)
            st.session_state["show_project_picker"] = False
            st.rerun()

    state = st.session_state.project_state
    if state:
        title = state.get("blueprint", {}).get("title", "Untitled")
        st.caption(f"**Project:** {title}")
        phase = state.get("current_phase", "init")
        st.caption(f"**Phase:** {phase}")

    st.divider()

    # Chat
    render_chat(
        messages=st.session_state.chat_messages,
        on_send=_on_send,
        disabled=st.session_state.agent_running,
        awaiting_human=state.get("awaiting_human", False) if state else False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
state = st.session_state.project_state
blueprint = state.get("blueprint") if state else None

st.title("🎬 VidMachine — Agentic Video Studio")

if state is None:
    st.info("Start a new project or load an existing one using the sidebar.")
    st.stop()

# Running indicator
if st.session_state.agent_running:
    st.warning("⚙️ Agents are working…", icon="⚙️")

# ── Top row: Timeline ────────────────────────────────────────────────────────
render_timeline(blueprint)

st.divider()

# ── Bottom row: Status | Assets ──────────────────────────────────────────────
col_status, col_assets = st.columns([1, 2])

with col_status:
    render_pipeline_status(state)

with col_assets:
    render_asset_browser(state, project_dir=state.get("project_dir", ""))

# ── Output files ─────────────────────────────────────────────────────────────
output_paths: dict = state.get("output_paths", {})
if output_paths:
    st.divider()
    st.subheader("📦 Outputs")
    for label, path in output_paths.items():
        p = Path(path)
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            col_a, col_b = st.columns([3, 1])
            col_a.success(f"**{label}** — `{p.name}` ({size_mb:.1f} MB)")
            with open(path, "rb") as f:
                col_b.download_button(
                    label=f"⬇ Download",
                    data=f,
                    file_name=p.name,
                    mime="video/mp4",
                    key=f"dl_{label}",
                )
        else:
            st.warning(f"**{label}** — file not found at `{path}`")
