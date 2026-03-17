"""
Export Agent — takes draft.mp4 from Assembly and produces polished final outputs.

Steps:
  A. Audio normalisation (loudnorm to -14 LUFS)
  B. Subtitle burn-in (optional, from blueprint.subtitle_track)
  C. 16:9 final render → output/final_16x9.mp4
  D. 9:16 auto-reframe via YOLO → output/final_9x16.mp4  (only if "9:16" in variants)
  E. Metadata collection via ffprobe
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from agents.base import BaseAgent
from state.project_state import ProjectState
from state.blueprint import VideoBlueprint, SubtitleEntry
from tools import ffmpeg_tools, yolo_tools

logger = logging.getLogger(__name__)


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert a float timestamp to SRT time format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_srt(subtitles: list[SubtitleEntry], srt_path: str) -> None:
    """Serialise a list of SubtitleEntry objects to an SRT file."""
    Path(srt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, sub in enumerate(sorted(subtitles, key=lambda s: s.start), start=1):
            f.write(f"{idx}\n")
            f.write(f"{_seconds_to_srt_time(sub.start)} --> {_seconds_to_srt_time(sub.end)}\n")
            f.write(f"{sub.text}\n\n")


class ExportAgent(BaseAgent):
    """
    Produces final deliverable video files from the assembled draft.

    Reads:
      state["output_paths"]["draft"]  — path to draft.mp4
      state["blueprint"]              — output config, subtitles, reframe, color_grade
      state["agent_notes"]["assembly"] — context from the Assembly agent

    Writes:
      state["output_paths"]["final_16x9"]   — 16:9 master
      state["output_paths"]["final_9x16"]   — 9:16 short-form (if variant requested)
      state["phase_results"]["export"]       — structured result with metadata
      state["agent_notes"]["export"]         — human-readable memo for QA agent
      state["current_phase"]                 — set to "export"
    """

    @property
    def name(self) -> str:
        return "export"

    def run(self, state: ProjectState) -> ProjectState:
        state["current_phase"] = "export"

        project_dir = state.get("project_dir", "")
        output_dir = str(Path(project_dir) / "output")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        blueprint_dict = state.get("blueprint", {})
        blueprint = VideoBlueprint.from_dict(blueprint_dict)
        assembly_note = state.get("agent_notes", {}).get("assembly", "")

        draft_path = state.get("output_paths", {}).get("draft", "")
        if not draft_path or not os.path.exists(draft_path):
            msg = f"Draft file not found at '{draft_path}'. Cannot proceed with export."
            self.add_error(state, msg)
            self.write_note(state, f"EXPORT FAILED: {msg}")
            return state

        self.log(f"Export starting — draft: {draft_path}")
        if assembly_note:
            self.log(f"Assembly note: {assembly_note[:200]}")

        errors: list[str] = []
        warnings: list[str] = []
        phase_result: dict = {
            "draft_path": draft_path,
            "steps": {},
            "outputs": {},
        }

        # ── Step A: Audio normalisation ────────────────────────────────────────
        normalised_path = str(Path(output_dir) / "normalised.mp4")
        self.log("Step A: Normalising audio to -14 LUFS")
        norm_result = ffmpeg_tools.normalize_audio(draft_path, normalised_path, target_lufs=-14.0)
        phase_result["steps"]["audio_normalisation"] = norm_result

        if not norm_result["success"]:
            warnings.append(f"Audio normalisation failed ({norm_result['error']}); using raw draft for downstream steps.")
            self.add_warning(state, warnings[-1])
            working_path = draft_path  # fall back to original
        else:
            working_path = normalised_path
            self.log(
                f"Audio normalised — measured LUFS: {norm_result.get('measured_lufs', 'N/A')}"
            )

        # ── Step B: Subtitle burn-in (optional) ───────────────────────────────
        if blueprint.subtitle_track:
            self.log(f"Step B: Burning {len(blueprint.subtitle_track)} subtitle entries")
            srt_path = str(Path(output_dir) / "subtitles.srt")
            try:
                _write_srt(blueprint.subtitle_track, srt_path)
                subtitled_path = str(Path(output_dir) / "with_subtitles.mp4")
                font_size = 24  # default; could be overridden via blueprint style
                sub_result = ffmpeg_tools.burn_subtitles(working_path, srt_path, subtitled_path, font_size=font_size)
                phase_result["steps"]["subtitle_burn"] = sub_result
                if sub_result["success"]:
                    working_path = subtitled_path
                    self.log("Subtitles burned in successfully.")
                else:
                    warnings.append(f"Subtitle burn failed ({sub_result['error']}); continuing without subtitles.")
                    self.add_warning(state, warnings[-1])
            except Exception as exc:
                warnings.append(f"Subtitle step error: {exc}")
                self.add_warning(state, warnings[-1])
                logger.exception("Subtitle burn-in error")
        else:
            self.log("Step B: No subtitle track — skipping burn-in.")
            phase_result["steps"]["subtitle_burn"] = {"skipped": True}

        # ── Step C: 16:9 Final Render ──────────────────────────────────────────
        self.log("Step C: Rendering 16:9 final output")
        final_16x9_path = str(Path(output_dir) / "final_16x9.mp4")
        out_cfg = blueprint.output
        render_result = ffmpeg_tools.render_final(
            input_path=working_path,
            output_path=final_16x9_path,
            resolution=tuple(out_cfg.resolution),  # type: ignore[arg-type]
            fps=out_cfg.fps,
            bitrate=out_cfg.bitrate,
            audio_bitrate=out_cfg.audio_bitrate,
            codec=out_cfg.codec,
            audio_codec=out_cfg.audio_codec,
        )
        phase_result["steps"]["render_16x9"] = render_result

        if render_result["success"]:
            state["output_paths"]["final_16x9"] = final_16x9_path
            self.log(
                f"16:9 render complete — {render_result['file_size_mb']:.1f} MB, "
                f"{render_result['duration']:.1f}s"
            )
        else:
            errors.append(f"16:9 render failed: {render_result['error']}")
            self.add_error(state, errors[-1])

        # ── Step D: 9:16 Auto-Reframe ─────────────────────────────────────────
        variants = out_cfg.variants or []
        if "9:16" in variants:
            self.log("Step D: Generating 9:16 auto-reframe via YOLO tracking")
            final_9x16_path = str(Path(output_dir) / "final_9x16.mp4")

            reframe_cfg = blueprint.reframe_9x16
            src_w = reframe_cfg.source_width
            src_h = reframe_cfg.source_height

            try:
                keyframes = yolo_tools.generate_reframe_keyframes(
                    draft_path,
                    source_width=src_w,
                    source_height=src_h,
                )
                self.log(f"YOLO produced {len(keyframes)} reframe keyframes")

                # Store keyframes back into the blueprint
                from state.blueprint import ReframeKeyframe
                blueprint.reframe_9x16.keyframes = [
                    ReframeKeyframe(**kf) for kf in keyframes
                ]
                state["blueprint"] = blueprint.to_dict()

                reframe_result = ffmpeg_tools.apply_reframe_keyframes(
                    input_path=working_path,
                    output_path=final_9x16_path,
                    keyframes=keyframes,
                    output_w=reframe_cfg.target_width,
                    output_h=reframe_cfg.target_height,
                    fps=out_cfg.fps,
                )
                phase_result["steps"]["reframe_9x16"] = reframe_result

                if reframe_result["success"]:
                    state["output_paths"]["final_9x16"] = final_9x16_path
                    self.log("9:16 reframe complete.")
                else:
                    errors.append(f"9:16 reframe render failed: {reframe_result['error']}")
                    self.add_error(state, errors[-1])

            except ImportError as exc:
                warnings.append(f"YOLO unavailable — skipping 9:16 reframe: {exc}")
                self.add_warning(state, warnings[-1])
                phase_result["steps"]["reframe_9x16"] = {"skipped": True, "reason": str(exc)}
            except Exception as exc:
                errors.append(f"9:16 reframe error: {exc}")
                self.add_error(state, errors[-1])
                logger.exception("9:16 reframe failed")
                phase_result["steps"]["reframe_9x16"] = {"success": False, "error": str(exc)}
        else:
            self.log("Step D: '9:16' not in output variants — skipping reframe.")
            phase_result["steps"]["reframe_9x16"] = {"skipped": True}

        # ── Step E: Metadata collection ────────────────────────────────────────
        self.log("Step E: Collecting output metadata via ffprobe")
        output_info: dict[str, dict] = {}

        for label, path_key in [("final_16x9", "final_16x9"), ("final_9x16", "final_9x16")]:
            out_path = state.get("output_paths", {}).get(path_key)
            if out_path and os.path.exists(out_path):
                info = ffmpeg_tools.get_video_info(out_path)
                output_info[label] = info
                self.log(
                    f"{label}: {info.get('width')}x{info.get('height')}, "
                    f"{info.get('duration', 0):.1f}s, "
                    f"{info.get('file_size_bytes', 0) / (1024*1024):.1f} MB"
                )
            else:
                output_info[label] = {}

        phase_result["outputs"] = output_info
        phase_result["errors"] = errors
        phase_result["warnings"] = warnings
        self.write_result(state, phase_result)

        # ── Agent note for QA ──────────────────────────────────────────────────
        note_lines = ["EXPORT AGENT SUMMARY", "=" * 40]

        note_lines.append(f"Draft input: {draft_path}")
        if assembly_note:
            note_lines.append(f"Assembly context: {assembly_note[:300]}")
        note_lines.append("")

        for label, info in output_info.items():
            if info:
                size_mb = info.get("file_size_bytes", 0) / (1024 * 1024)
                note_lines.append(
                    f"{label}: {info.get('width')}x{info.get('height')} | "
                    f"{info.get('duration', 0):.2f}s | "
                    f"{size_mb:.1f} MB | "
                    f"codec={info.get('codec')} audio={info.get('audio_codec')}"
                )
            else:
                note_lines.append(f"{label}: NOT PRODUCED")

        if errors:
            note_lines.append("")
            note_lines.append("ERRORS:")
            for e in errors:
                note_lines.append(f"  - {e}")

        if warnings:
            note_lines.append("")
            note_lines.append("WARNINGS:")
            for w in warnings:
                note_lines.append(f"  - {w}")

        if not errors:
            note_lines.append("")
            note_lines.append("All export steps completed successfully. Ready for QA.")

        self.write_note(state, "\n".join(note_lines))
        self.show_panel("Export Complete", "\n".join(note_lines))

        return state
