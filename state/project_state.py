"""
LangGraph shared state — the single source of truth passed between all agents.

All agents receive a ProjectState, mutate the fields they own, and return it.
The Director reads the full state to decide routing.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


Phase = Literal[
    "init",
    "ingest",
    "sourcing",
    "assembly",
    "export",
    "qa",
    "done",
]


class ProjectState(TypedDict):
    # ── Identity ──────────────────────────────────────────────────────────────
    project_id: str
    project_dir: str                    # Absolute path to project folder

    # ── Conversation ──────────────────────────────────────────────────────────
    user_intent: str                    # Latest user message / instruction
    messages: Annotated[list[BaseMessage], add_messages]  # Full chat history

    # ── Blueprint ─────────────────────────────────────────────────────────────
    blueprint: dict                     # VideoBlueprint serialised as dict

    # ── Pipeline Control ──────────────────────────────────────────────────────
    current_phase: Phase                # Which phase is active
    next_phase: Phase | None            # Director sets this to route workflow
    phase_results: dict[str, Any]       # agent_name → structured result dict
    agent_notes: dict[str, str]         # agent_name → plain-English memo for peers

    # ── Assets ────────────────────────────────────────────────────────────────
    raw_assets: list[dict]              # AssetInfo dicts (not yet processed)
    processed_assets: list[dict]        # AssetInfo dicts (ready for assembly)

    # ── Status ────────────────────────────────────────────────────────────────
    errors: list[str]
    warnings: list[str]

    # ── Output ────────────────────────────────────────────────────────────────
    output_paths: dict[str, str]        # "draft" | "final_16x9" | "final_9x16" → abs path

    # ── Human-in-the-loop ─────────────────────────────────────────────────────
    awaiting_human: bool                # If True, LangGraph interrupts for user input


def make_initial_state(
    project_id: str | None = None,
    project_dir: str = "",
    user_intent: str = "",
) -> ProjectState:
    """Factory for a clean, empty ProjectState."""
    from state.blueprint import VideoBlueprint

    pid = project_id or uuid.uuid4().hex

    return ProjectState(
        project_id=pid,
        project_dir=project_dir,
        user_intent=user_intent,
        messages=[],
        blueprint=VideoBlueprint(project_id=pid).to_dict(),
        current_phase="init",
        next_phase=None,
        phase_results={},
        agent_notes={},
        raw_assets=[],
        processed_assets=[],
        errors=[],
        warnings=[],
        output_paths={},
        awaiting_human=False,
    )
