"""
LangGraph routing functions.

route_from_director  — conditional edge called after the Director node.
route_after_worker   — conditional edge called after every worker node.
"""

from __future__ import annotations

import logging

from langgraph.graph import END

from state.project_state import ProjectState

logger = logging.getLogger(__name__)


def route_from_director(state: ProjectState) -> str:
    """
    Conditional edge function called after the Director node.

    Priority order:
    1. If `awaiting_human` is True  → interrupt at "human_feedback" node.
    2. Map `next_phase` to the corresponding worker node name.
    3. Fallback to END if next_phase is None, "done", or unrecognised.

    Returns the name of the next node to activate, or END.
    """
    # Human-in-the-loop takes priority over any phase routing
    if state.get("awaiting_human", False):
        logger.debug("Router: awaiting_human=True → human_feedback")
        return "human_feedback"

    next_phase = state.get("next_phase")

    _phase_to_node: dict[str, str] = {
        "ingest": "ingest",
        "sourcing": "sourcing",
        "assembly": "assembly",
        "export": "export",
        "qa": "qa",
    }

    if next_phase in _phase_to_node:
        node = _phase_to_node[next_phase]
        logger.debug("Router: next_phase=%s → %s", next_phase, node)
        return node

    if next_phase == "done":
        logger.debug("Router: next_phase=done → END")
        return END  # type: ignore[return-value]

    # next_phase is None or an unexpected value
    logger.warning(
        "Router: unrecognised next_phase=%r, defaulting to END", next_phase
    )
    return END  # type: ignore[return-value]


def route_after_worker(state: ProjectState) -> str:
    """
    Conditional edge called after every worker node (ingest, sourcing,
    assembly, export, qa).

    All workers hand control back to the Director so it can assess the
    result, update the blueprint if needed, and decide the next phase.

    Returns "director" unconditionally.
    """
    logger.debug(
        "route_after_worker: phase=%s → director", state.get("current_phase")
    )
    return "director"
