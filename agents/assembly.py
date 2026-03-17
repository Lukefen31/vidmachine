"""
AssemblyAgent — sequences clips, burns overlays, mixes audio, and writes
draft.mp4 to {project_dir}/output/draft.mp4.

Responsibilities:
  1. Read blueprint from state["blueprint"]
  2. Validate that source files exist; warn on missing, never crash
  3. Optionally snap beat-aligned clip positions to the beat grid via
     librosa_tools.align_cuts_to_beats()
  4. Call moviepy_tools.assemble_draft() to produce the draft video
  5. Update state["blueprint"], state["output_paths"]["draft"],
     state["current_phase"], and write a rich agent_note for Export
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from agents.base import BaseAgent

if TYPE_CHECKING:
    from state.project_state import ProjectState

logger = logging.getLogger(__name__)


class AssemblyAgent(BaseAgent):
    """Assembles the draft video from the VideoBlueprint."""

    @property
    def name(self) -> str:
        return "assembly"

    # ------------------------------------------------------------------
    # Public entry point (called by LangGraph via __call__ on BaseAgent)
    # ------------------------------------------------------------------

    def run(self, state: "ProjectState") -> "ProjectState":
        """
        Assemble draft.mp4 from the current blueprint state.

        Idempotent: calling this again after the user edits the blueprint
        simply re-runs the assembly, overwriting the previous draft.
        """
        self.log("Starting assembly run")

        project_dir: str = state.get("project_dir", "")
        if not project_dir:
            self.add_error(state, "project_dir is not set in state")
            return state

        # ── 1. Deserialise blueprint ───────────────────────────────────────
        blueprint_dict: dict = state.get("blueprint", {})
        if not blueprint_dict:
            self.add_error(state, "blueprint is empty — nothing to assemble")
            return state

        try:
            from state.blueprint import VideoBlueprint  # noqa: PLC0415
            blueprint = VideoBlueprint.from_dict(blueprint_dict)
        except Exception as exc:  # noqa: BLE001
            self.add_error(state, f"Failed to parse blueprint: {exc}")
            logger.exception("AssemblyAgent: blueprint parse error")
            return state

        self.log(
            f"Blueprint loaded — {len(blueprint.tracks.video)} video clip(s), "
            f"{len(blueprint.tracks.audio)} music track(s), "
            f"{len(blueprint.tracks.voiceover)} voiceover track(s), "
            f"{len(blueprint.tracks.sfx)} sfx track(s), "
            f"{len(blueprint.tracks.text_overlays)} text overlay(s)"
        )

        # ── 2. Read context notes from upstream agents ────────────────────
        agent_notes: dict = state.get("agent_notes", {})
        ingest_note: str = agent_notes.get("ingest", "")
        sourcing_note: str = agent_notes.get("sourcing", "")
        if ingest_note:
            logger.debug("AssemblyAgent — ingest note: %s", ingest_note)
        if sourcing_note:
            logger.debug("AssemblyAgent — sourcing note: %s", sourcing_note)

        # ── 3. Validate source files ───────────────────────────────────────
        skipped_clips: list[str] = []
        valid_video_clips: list = []

        for clip in blueprint.tracks.video:
            src_path = Path(clip.source)

            # Resolve relative paths against project_dir
            if not src_path.is_absolute():
                src_path = Path(project_dir) / src_path
                clip.source = str(src_path)

            if not src_path.is_file():
                warning_msg = (
                    f"Source file missing, clip will be skipped: '{clip.source}'"
                )
                logger.warning("AssemblyAgent: %s", warning_msg)
                self.add_warning(state, warning_msg)
                skipped_clips.append(clip.source)
            else:
                valid_video_clips.append(clip)

        # Replace blueprint video track with only the valid clips
        blueprint.tracks.video = valid_video_clips

        # Validate audio source files (warn but don't remove — mix_audio_tracks
        # handles skipping gracefully)
        all_audio = (
            blueprint.tracks.audio
            + blueprint.tracks.voiceover
            + blueprint.tracks.sfx
        )
        for atrack in all_audio:
            src_path = Path(atrack.source)
            if not src_path.is_absolute():
                src_path = Path(project_dir) / src_path
                atrack.source = str(src_path)
            if not src_path.is_file():
                self.add_warning(
                    state,
                    f"Audio source missing (will be skipped): '{atrack.source}'",
                )

        if not blueprint.tracks.video:
            self.add_error(
                state,
                "No valid video clips remain after source validation — cannot assemble.",
            )
            return state

        # ── 4. Beat alignment ─────────────────────────────────────────────
        beat_aligned_flag = False
        beat_map: list[float] = blueprint.beat_map

        if beat_map and any(c.beat_aligned for c in blueprint.tracks.video):
            self.log("Beat alignment requested — snapping clip positions to beat grid")
            try:
                from tools.librosa_tools import align_cuts_to_beats  # noqa: PLC0415

                cut_times: list[float] = [
                    c.timeline_position
                    for c in blueprint.tracks.video
                    if c.beat_aligned
                ]
                snapped_times: list[float] = align_cuts_to_beats(cut_times, beat_map)

                beat_idx = 0
                for clip in blueprint.tracks.video:
                    if clip.beat_aligned:
                        original = clip.timeline_position
                        clip.timeline_position = snapped_times[beat_idx]
                        logger.debug(
                            "Beat-align: clip '%s'  %.3f → %.3f",
                            clip.id,
                            original,
                            clip.timeline_position,
                        )
                        beat_idx += 1

                beat_aligned_flag = True
                self.log(
                    f"Beat alignment complete — {beat_idx} clip(s) snapped to beat grid"
                )
            except ImportError:
                self.add_warning(
                    state,
                    "librosa_tools not available; beat alignment skipped.",
                )
                logger.warning(
                    "AssemblyAgent: librosa_tools.align_cuts_to_beats could not be imported"
                )
            except Exception as exc:  # noqa: BLE001
                self.add_warning(
                    state,
                    f"Beat alignment failed ({exc}); proceeding with original positions.",
                )
                logger.exception("AssemblyAgent: beat alignment error")

        # Persist any timeline adjustments back to state
        blueprint.touch()
        state["blueprint"] = blueprint.to_dict()

        # ── 5. Determine output path ──────────────────────────────────────
        output_dir = Path(project_dir) / "output"
        output_path = str(output_dir / "draft.mp4")

        self.log(f"Output path: {output_path}")

        # ── 6. Run assembly ────────────────────────────────────────────────
        try:
            from tools.moviepy_tools import assemble_draft  # noqa: PLC0415
        except ImportError as exc:
            self.add_error(
                state,
                f"moviepy_tools import failed — is moviepy installed? ({exc})",
            )
            return state

        self.log(
            f"Calling assemble_draft with "
            f"{len(blueprint.tracks.video)} clip(s) …"
        )

        assemble_result: dict = assemble_draft(
            blueprint.to_dict(),
            output_path=output_path,
        )

        # Merge any skipped clips from the assembler into our list
        assembler_skipped: list[str] = assemble_result.get("skipped_clips", [])
        all_skipped = list(dict.fromkeys(skipped_clips + assembler_skipped))

        # ── 7. Handle assembly failure ────────────────────────────────────
        if not assemble_result.get("success", False):
            error_msg = assemble_result.get("error", "Unknown assembly error")
            self.add_error(state, f"Assembly failed: {error_msg}")
            self.log(f"Assembly FAILED — {error_msg}", level="error")
            self.write_result(
                state,
                {
                    "success": False,
                    "error": error_msg,
                    "skipped_clips": all_skipped,
                },
            )
            return state

        # ── 8. Update state with results ──────────────────────────────────
        final_duration: float = assemble_result.get("duration", 0.0)

        # output_paths
        output_paths: dict = state.get("output_paths", {})
        output_paths["draft"] = output_path
        state["output_paths"] = output_paths

        # current phase
        state["current_phase"] = "assembly"

        # Structured result for Director / QA
        self.write_result(
            state,
            {
                "success": True,
                "output_path": output_path,
                "duration": final_duration,
                "beat_aligned": beat_aligned_flag,
                "skipped_clips": all_skipped,
            },
        )

        # ── 9. Write agent note for Export agent ──────────────────────────
        skipped_summary = (
            f"Skipped clips ({len(all_skipped)}): "
            + ", ".join(Path(p).name for p in all_skipped)
            if all_skipped
            else "None"
        )
        note = (
            f"Draft assembled at {output_path}. "
            f"Duration: {final_duration:.1f}s. "
            f"Beat-aligned: {'yes' if beat_aligned_flag else 'no'}. "
            f"Issues: {skipped_summary}."
        )
        self.write_note(state, note)

        # ── 10. Rich console summary ──────────────────────────────────────
        panel_body = (
            f"Output:        {output_path}\n"
            f"Duration:      {final_duration:.2f}s\n"
            f"Beat-aligned:  {'yes' if beat_aligned_flag else 'no'}\n"
            f"Clips used:    {len(blueprint.tracks.video)}\n"
            f"Clips skipped: {len(all_skipped)}"
        )
        if all_skipped:
            panel_body += "\nSkipped:\n" + "\n".join(
                f"  • {Path(p).name}" for p in all_skipped
            )
        self.show_panel("Assembly Complete", panel_body)

        self.log(
            f"Assembly complete — draft.mp4 written ({final_duration:.2f}s)"
        )
        return state
