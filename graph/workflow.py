"""
vidmachine LangGraph workflow.

Builds the full StateGraph connecting the Director supervisor to all
pipeline worker nodes.  Human-in-the-loop is supported via a MemorySaver
checkpointer and an explicit `human_feedback` interrupt node.

Worker agents are imported lazily with stub fallbacks so the graph compiles
even before every worker module has been written.  As real agent modules are
added they are picked up automatically.
"""

from __future__ import annotations

import logging
from typing import Callable

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from agents.director import director_agent
from graph.router import route_after_worker, route_from_director
from state.project_state import ProjectState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Worker agent loader with stub fallback
# ─────────────────────────────────────────────────────────────────────────────

def _load_worker(module_path: str, class_name: str, phase_name: str) -> Callable:
    """
    Attempt to import `class_name` from `module_path`.

    If the module does not exist yet, return a lightweight stub that:
      - Sets current_phase to phase_name
      - Writes a note explaining it is a placeholder
      - Does NOT advance next_phase (Director will handle that)

    This lets the graph compile before all workers are implemented.
    """
    try:
        import importlib

        mod = importlib.import_module(module_path)
        agent_cls = getattr(mod, class_name)
        instance = agent_cls()
        logger.debug("Loaded agent %s from %s", class_name, module_path)
        return instance
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "Worker agent %s not found (%s) — using stub for phase '%s'",
            class_name,
            exc,
            phase_name,
        )

        def _stub(state: ProjectState) -> ProjectState:
            logger.warning(
                "STUB worker running for phase '%s' — real agent not yet implemented.",
                phase_name,
            )
            state["current_phase"] = phase_name  # type: ignore[typeddict-item]
            notes = dict(state.get("agent_notes", {}))
            notes[phase_name] = (
                f"[STUB] Phase '{phase_name}' agent is not yet implemented. "
                "Returning state unchanged."
            )
            state["agent_notes"] = notes
            warnings = list(state.get("warnings", []))
            warnings.append(
                f"[{phase_name.upper()}] Agent not implemented — stub executed."
            )
            state["warnings"] = warnings
            return state

        _stub.__name__ = f"{phase_name}_stub"
        return _stub


# ─────────────────────────────────────────────────────────────────────────────
# human_feedback node
# ─────────────────────────────────────────────────────────────────────────────

def human_feedback_node(state: ProjectState) -> ProjectState:
    """
    Interrupt node for human-in-the-loop.

    When LangGraph resumes after a human interrupt it calls this node with
    the updated state (the runner is responsible for injecting the human's
    reply into `user_intent` before resuming).

    This node:
      1. Clears the awaiting_human flag.
      2. Appends the current user_intent to the messages list so the full
         conversation history is preserved.
      3. Returns state so the Director can re-evaluate.
    """
    # Clear the interrupt flag
    state["awaiting_human"] = False

    # Record the human's latest message in the conversation history
    user_intent = state.get("user_intent", "").strip()
    if user_intent:
        from langchain_core.messages import HumanMessage

        current_messages = list(state.get("messages", []))
        current_messages.append(HumanMessage(content=user_intent))
        state["messages"] = current_messages

    logger.debug("human_feedback_node: awaiting_human cleared, routing back to director")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_workflow() -> StateGraph:  # type: ignore[type-arg]
    """
    Construct and compile the full vidmachine LangGraph StateGraph.

    Graph topology
    ──────────────
    [START] → director
       director ──(route_from_director)──► ingest
                                        ► sourcing
                                        ► assembly
                                        ► export
                                        ► qa
                                        ► human_feedback
                                        ► END  (when next_phase == "done")

    Each worker ──(route_after_worker)──► director
    human_feedback ──────────────────────► director

    Returns the compiled graph (with MemorySaver checkpointer).
    """
    # Load worker agents (real implementations or stubs)
    ingest_agent = _load_worker("agents.ingest", "IngestAgent", "ingest")
    sourcing_agent = _load_worker("agents.sourcing", "SourcingAgent", "sourcing")
    assembly_agent = _load_worker("agents.assembly", "AssemblyAgent", "assembly")
    export_agent = _load_worker("agents.export", "ExportAgent", "export")
    qa_agent = _load_worker("agents.qa", "QAAgent", "qa")

    # ── Build graph ──────────────────────────────────────────────────────────
    graph = StateGraph(ProjectState)

    # Add all nodes
    graph.add_node("director", director_agent)
    graph.add_node("ingest", ingest_agent)
    graph.add_node("sourcing", sourcing_agent)
    graph.add_node("assembly", assembly_agent)
    graph.add_node("export", export_agent)
    graph.add_node("qa", qa_agent)
    graph.add_node("human_feedback", human_feedback_node)

    # Director is the entry point
    graph.set_entry_point("director")

    # Conditional edges out of the Director
    graph.add_conditional_edges(
        "director",
        route_from_director,
        {
            "ingest": "ingest",
            "sourcing": "sourcing",
            "assembly": "assembly",
            "export": "export",
            "qa": "qa",
            "human_feedback": "human_feedback",
            END: END,
        },
    )

    # All workers route back to Director
    for worker_node in ("ingest", "sourcing", "assembly", "export", "qa"):
        graph.add_conditional_edges(
            worker_node,
            route_after_worker,
            {"director": "director"},
        )

    # human_feedback always returns to Director (which re-evaluates intent)
    graph.add_edge("human_feedback", "director")

    # Compile with MemorySaver for human-in-the-loop checkpoint/resume support
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    logger.info("vidmachine workflow compiled successfully.")
    return compiled


# ─────────────────────────────────────────────────────────────────────────────
# Module-level compiled graph (imported by graph/__init__.py and the runner)
# ─────────────────────────────────────────────────────────────────────────────

compiled_graph = build_workflow()
