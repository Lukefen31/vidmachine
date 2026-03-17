"""
Director Agent — LangGraph supervisor.

Reads the full ProjectState, reasons about what to do next using Claude
(claude-sonnet-4-6) with structured tool calls, then sets `next_phase` and
mutates the `blueprint` accordingly.

The Director is the only agent that calls the Anthropic API directly.  All
other agents are domain workers that receive a fully-planned blueprint.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from agents.base import BaseAgent
from config import settings
from state.project_state import ProjectState

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas passed to Claude
# ─────────────────────────────────────────────────────────────────────────────

DIRECTOR_TOOLS: list[dict] = [
    {
        "name": "set_next_phase",
        "description": (
            "Set the next pipeline phase the workflow should activate.  "
            "Call this once per turn after you have decided what to do next.  "
            "Valid phases: ingest, sourcing, assembly, export, qa, done."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "phase": {
                    "type": "string",
                    "enum": ["ingest", "sourcing", "assembly", "export", "qa", "done"],
                    "description": "The phase to activate next.",
                },
                "reason": {
                    "type": "string",
                    "description": "One-sentence explanation of why this phase is next.",
                },
            },
            "required": ["phase", "reason"],
        },
    },
    {
        "name": "update_blueprint",
        "description": (
            "Update a single field in the VideoBlueprint using dot-notation.  "
            "Examples: 'output.variants', 'tracks.video[0].beat_aligned', "
            "'output.duration_target', 'color_grade.preset', 'bpm'.  "
            "Use JSON-serialisable values only.  "
            "For list indices use numeric notation: 'tracks.video.1.out_point'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "field_path": {
                    "type": "string",
                    "description": "Dot-notation path into the blueprint dict.",
                },
                "value": {
                    "description": "New value for the field (must be JSON-serialisable).",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this change is being made.",
                },
            },
            "required": ["field_path", "value", "reason"],
        },
    },
    {
        "name": "request_human_feedback",
        "description": (
            "Pause the pipeline and ask the human a clarifying question.  "
            "Use this when the user's intent is ambiguous or when a creative "
            "decision requires explicit sign-off."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to display to the human.",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "mark_done",
        "description": (
            "Mark the project as fully complete.  Call this only when all "
            "requested phases have finished and outputs have been verified."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what was produced.",
                },
            },
            "required": ["summary"],
        },
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the Director of an AI-powered video production pipeline called vidmachine.
Your job is to orchestrate a set of specialised worker agents by reading the
current project state and deciding what to do next.

## Pipeline phases (in typical order)
1. ingest      — Scan raw footage, analyse clips, extract metadata & motion scores.
2. sourcing    — Source additional assets: stock footage, music, voiceover TTS, SFX.
3. assembly    — Cut clips to the timeline, align to beats, apply effects & titles.
4. qa          — Quality-check the rendered draft; flag technical issues.
5. export      — Render final output in all requested variants (16:9, 9:16, etc.).
6. done        — All work is complete.

## Your responsibilities
- Parse the user's latest intent and previous agent notes.
- Decide which phase to run next (you may re-run phases after user edits).
- Update the blueprint when the user requests creative changes.
- Ask for human confirmation before irreversible actions or when intent is unclear.
- Be concise. Use tool calls to express decisions, not prose.

## Blueprint mutation rules
- Use `update_blueprint` with dot-notation paths.
- To set ALL video clips to beat-aligned use multiple calls or set the flag at
  the track level via a 'beat_aligned_default' annotation key.
- Output variants live at `output.variants` and must be a list such as
  ["16:9"] or ["16:9", "9:16"].
- Voiceover text lives in `tracks.voiceover` as AudioTrack entries with
  metadata.tts_text set.
- Target duration lives at `output.duration_target` (float, seconds).

## Common intent patterns
| User says                              | Actions                                              |
|----------------------------------------|------------------------------------------------------|
| "Create a 30s FPV compilation …"       | Set duration_target=30, set title, set_next_phase=ingest |
| "Make cuts on the beat"                | Update beat_aligned flags, set_next_phase=assembly   |
| "Add a voiceover saying X"             | Add voiceover entry with tts_text=X, set_next_phase=sourcing |
| "Export in 9:16"                       | Set output.variants=["16:9","9:16"], set_next_phase=export |
| "Make scene 2 shorter"                 | Update tracks.video.1.out_point, set_next_phase=assembly |
| Ambiguous request                      | request_human_feedback with clarifying question      |

Always call at least one tool.  Never respond with pure text only.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Blueprint helper — resolve / set a dot-notation path
# ─────────────────────────────────────────────────────────────────────────────

def _set_nested(obj: Any, path: str, value: Any) -> None:
    """
    Set a value inside a nested dict/list structure using dot-notation.
    Numeric segments are treated as list indices.

    Example:
        _set_nested(bp, "tracks.video.0.beat_aligned", True)
        _set_nested(bp, "output.variants", ["16:9", "9:16"])
    """
    parts = path.split(".")
    for part in parts[:-1]:
        if isinstance(obj, list):
            obj = obj[int(part)]
        elif isinstance(obj, dict):
            obj = obj.setdefault(part, {})
        else:
            raise KeyError(f"Cannot traverse into {type(obj)} at segment '{part}'")

    final = parts[-1]
    if isinstance(obj, list):
        obj[int(final)] = value
    elif isinstance(obj, dict):
        obj[final] = value
    else:
        raise KeyError(f"Cannot set attribute on {type(obj)} at segment '{final}'")


# ─────────────────────────────────────────────────────────────────────────────
# Director Agent
# ─────────────────────────────────────────────────────────────────────────────

class DirectorAgent(BaseAgent):
    """LangGraph supervisor — routes the workflow and mutates the blueprint."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.claude_model

    @property
    def name(self) -> str:
        return "director"

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, state: ProjectState) -> ProjectState:
        user_intent = state.get("user_intent", "").strip()
        agent_notes = state.get("agent_notes", {})
        blueprint = state.get("blueprint", {})
        current_phase = state.get("current_phase", "init")
        errors = state.get("errors", [])

        self.show_panel(
            "Reasoning",
            f"Phase: {current_phase} | Intent: {user_intent[:120]}",
        )

        # Build the context message for Claude
        context_parts: list[str] = [
            f"## Current phase\n{current_phase}",
            f"## User intent\n{user_intent or '(none)'}",
        ]

        if agent_notes:
            notes_block = "\n".join(
                f"- **{agent}**: {note}" for agent, note in agent_notes.items()
            )
            context_parts.append(f"## Agent notes from completed phases\n{notes_block}")

        if errors:
            context_parts.append(
                "## Errors reported by workers\n"
                + "\n".join(f"- {e}" for e in errors[-5:])  # last 5 only
            )

        # Summarise key blueprint fields so Claude doesn't need to read the whole thing
        bp_summary = self._summarise_blueprint(blueprint)
        context_parts.append(f"## Blueprint summary\n{bp_summary}")

        user_message = "\n\n".join(context_parts)

        # Call Claude with tool_use
        response = self._call_claude(user_message)

        # Process tool calls
        state = self._process_tool_calls(state, response)

        return state

    # ── Claude call ───────────────────────────────────────────────────────────

    def _call_claude(self, user_message: str) -> anthropic.types.Message:
        """Call Claude with the Director's tools and return the raw Message."""
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=DIRECTOR_TOOLS,  # type: ignore[arg-type]
                messages=[
                    {"role": "user", "content": user_message},
                ],
            )
            logger.debug(
                "Director Claude response: stop_reason=%s, tool_calls=%d",
                response.stop_reason,
                sum(1 for b in response.content if b.type == "tool_use"),
            )
            return response
        except anthropic.APIError as exc:
            logger.error("Anthropic API error in Director: %s", exc)
            raise

    # ── Tool dispatch ─────────────────────────────────────────────────────────

    def _process_tool_calls(
        self, state: ProjectState, response: anthropic.types.Message
    ) -> ProjectState:
        """Iterate over Claude's tool_use blocks and apply each action to state."""
        tool_blocks = [b for b in response.content if b.type == "tool_use"]

        if not tool_blocks:
            # Claude chose not to call any tool — log text response and go to done
            text_blocks = [b for b in response.content if b.type == "text"]
            text = " ".join(b.text for b in text_blocks).strip()
            self.log(
                f"No tool calls returned by Claude. Text response: {text[:200]}",
                level="warning",
            )
            self.write_note(
                state,
                f"Director could not determine next action. Claude said: {text[:400]}",
            )
            # Default to requesting human feedback so the pipeline doesn't stall
            state["awaiting_human"] = True
            state["agent_notes"]["director"] = (
                "I couldn't determine the next step automatically. "
                "Please clarify what you'd like to do next."
            )
            return state

        for block in tool_blocks:
            tool_name: str = block.name
            tool_input: dict = block.input  # type: ignore[assignment]

            self.log(f"Tool call: {tool_name}({json.dumps(tool_input)[:200]})")

            try:
                if tool_name == "set_next_phase":
                    state = self._tool_set_next_phase(state, tool_input)
                elif tool_name == "update_blueprint":
                    state = self._tool_update_blueprint(state, tool_input)
                elif tool_name == "request_human_feedback":
                    state = self._tool_request_human_feedback(state, tool_input)
                elif tool_name == "mark_done":
                    state = self._tool_mark_done(state, tool_input)
                else:
                    self.add_warning(state, f"Unknown tool called by Claude: {tool_name}")
            except Exception as exc:
                self.add_error(
                    state,
                    f"Tool '{tool_name}' raised {type(exc).__name__}: {exc}",
                )
                logger.exception("Error executing Director tool '%s'", tool_name)

        return state

    # ── Individual tool implementations ───────────────────────────────────────

    def _tool_set_next_phase(
        self, state: ProjectState, inp: dict
    ) -> ProjectState:
        phase: str = inp["phase"]
        reason: str = inp.get("reason", "")

        valid_phases = {"ingest", "sourcing", "assembly", "export", "qa", "done"}
        if phase not in valid_phases:
            self.add_error(state, f"set_next_phase: invalid phase '{phase}'")
            return state

        state["next_phase"] = phase  # type: ignore[typeddict-item]
        self.write_note(state, f"Routing to '{phase}': {reason}")
        self.log(f"next_phase => {phase} | {reason}")
        return state

    def _tool_update_blueprint(
        self, state: ProjectState, inp: dict
    ) -> ProjectState:
        field_path: str = inp["field_path"]
        value: Any = inp["value"]
        reason: str = inp.get("reason", "")

        blueprint: dict = state.get("blueprint", {})

        try:
            _set_nested(blueprint, field_path, value)
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            self.add_error(
                state,
                f"update_blueprint failed for path '{field_path}': {exc}",
            )
            return state

        # Touch updated_at
        from datetime import datetime

        blueprint["updated_at"] = datetime.utcnow().isoformat()
        state["blueprint"] = blueprint

        self.log(f"Blueprint updated: {field_path} = {repr(value)[:80]} | {reason}")
        return state

    def _tool_request_human_feedback(
        self, state: ProjectState, inp: dict
    ) -> ProjectState:
        question: str = inp["question"]
        state["awaiting_human"] = True
        state["agent_notes"]["director"] = f"[QUESTION FOR HUMAN] {question}"
        self.log(f"Awaiting human: {question}")
        # next_phase stays as-is; the router will detect awaiting_human first
        return state

    def _tool_mark_done(
        self, state: ProjectState, inp: dict
    ) -> ProjectState:
        summary: str = inp["summary"]
        state["next_phase"] = "done"  # type: ignore[typeddict-item]
        self.write_note(state, f"Project complete: {summary}")
        self.log(f"Marked done: {summary}")
        return state

    # ── Blueprint summariser ──────────────────────────────────────────────────

    def _summarise_blueprint(self, blueprint: dict) -> str:
        """Return a compact, human-readable summary of the key blueprint fields."""
        lines: list[str] = []

        title = blueprint.get("title", "Untitled")
        lines.append(f"Title: {title}")

        output = blueprint.get("output", {})
        lines.append(
            f"Output: {output.get('resolution')} @ {output.get('fps')}fps "
            f"| variants={output.get('variants')} "
            f"| duration_target={output.get('duration_target')}s"
        )

        tracks = blueprint.get("tracks", {})
        video_clips = tracks.get("video", [])
        audio_tracks = tracks.get("audio", [])
        voiceover_tracks = tracks.get("voiceover", [])
        lines.append(
            f"Tracks: {len(video_clips)} video clips, "
            f"{len(audio_tracks)} music tracks, "
            f"{len(voiceover_tracks)} voiceover tracks"
        )

        bpm = blueprint.get("bpm")
        beat_map_len = len(blueprint.get("beat_map", []))
        if bpm or beat_map_len:
            lines.append(f"Beat map: bpm={bpm}, {beat_map_len} markers")

        color_grade = blueprint.get("color_grade", {})
        if color_grade.get("preset", "none") != "none":
            lines.append(f"Color grade: {color_grade.get('preset')}")

        annotations = blueprint.get("agent_annotations", {})
        if annotations:
            lines.append(
                "Agent annotations: "
                + ", ".join(f"{k}={repr(v)[:40]}" for k, v in annotations.items())
            )

        return "\n".join(lines)


# Module-level singleton used by workflow.py
director_agent = DirectorAgent()
