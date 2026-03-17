"""
QA Agent — audits exported videos for quality issues before the pipeline ends.

Checks performed:
  1. File existence       — all output_paths exist on disk
  2. Duration check       — output duration vs blueprint duration_target (warn if >10% off)
  3. Audio sync           — video and audio stream lengths match (via ffprobe)
  4. Resolution check     — 16:9 output has correct resolution; 9:16 has correct vertical res
  5. File size sanity     — warn if <1 MB (likely corrupt) or >2 GB (may need re-encode)
  6. Beat alignment audit — sample cut points from blueprint and check actual frame alignment

Results are stored in state["phase_results"]["qa"]:
  {passed: bool, checks: [{name, status, detail}], score: int 0-100}

Routing:
  - All pass: next_phase = "done"
  - Critical failure: next_phase = "assembly" or "export" depending on failure type
  - Warnings only: next_phase = "done" (with warnings)
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Literal

import os

from agents.base import BaseAgent
from state.project_state import ProjectState
from state.blueprint import VideoBlueprint
from tools import ffmpeg_tools

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Check status literals
# ---------------------------------------------------------------------------

CheckStatus = Literal["pass", "warn", "fail", "skip"]


def _check(name: str, status: CheckStatus, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


# ---------------------------------------------------------------------------
# QA Agent
# ---------------------------------------------------------------------------

class QAAgent(BaseAgent):
    """
    Audits the exported video files against the blueprint specification.

    Reads:
      state["output_paths"]            — paths to final outputs
      state["blueprint"]               — target spec (resolution, duration, fps, variants)
      state["phase_results"]["export"] — export metadata

    Writes:
      state["phase_results"]["qa"]     — {passed, checks, score}
      state["agent_notes"]["qa"]       — plain-English summary for Director
      state["next_phase"]              — "done" | "export" | "assembly"
    """

    # Thresholds
    DURATION_TOLERANCE = 0.10                    # 10% deviation triggers warning
    AUDIO_SYNC_TOLERANCE = 0.5                   # seconds — max acceptable A/V length mismatch
    MIN_FILE_SIZE_BYTES = 1 * 1024 * 1024        # 1 MB
    MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024 # 2 GB
    BEAT_ALIGN_TOLERANCE_FRAMES = 1              # allow ±1 frame

    @property
    def name(self) -> str:
        return "qa"

    def run(self, state: ProjectState) -> ProjectState:
        blueprint_dict = state.get("blueprint", {})
        blueprint = VideoBlueprint.from_dict(blueprint_dict)
        output_paths = state.get("output_paths", {})
        output_cfg = blueprint.output

        checks: list[dict] = []
        critical_failures: list[str] = []   # stage names that need re-run
        has_warnings = False

        # ── 1. File existence ──────────────────────────────────────────────────
        expected_outputs: dict[str, str] = {}
        if "16:9" in output_cfg.variants or not output_cfg.variants:
            expected_outputs["final_16x9"] = output_paths.get("final_16x9", "")
        if "9:16" in output_cfg.variants:
            expected_outputs["final_9x16"] = output_paths.get("final_9x16", "")

        for label, path in expected_outputs.items():
            if path and os.path.exists(path):
                checks.append(_check(
                    f"file_exists_{label}", "pass",
                    f"{label} exists at {path}"
                ))
            else:
                checks.append(_check(
                    f"file_exists_{label}", "fail",
                    f"{label} not found — expected at: {path or '(no path recorded)'}"
                ))
                critical_failures.append("export")

        # Collect video info for all existing outputs (used by later checks)
        video_infos: dict[str, dict] = {}
        for label, path in expected_outputs.items():
            if path and os.path.exists(path):
                info = ffmpeg_tools.get_video_info(path)
                video_infos[label] = info if info else {}

        # ── 2. Duration check ─────────────────────────────────────────────────
        target_duration: float | None = output_cfg.duration_target
        if target_duration is None:
            assembled = blueprint.total_video_duration()
            target_duration = assembled if assembled > 0 else None

        if target_duration and target_duration > 0:
            for label, info in video_infos.items():
                actual_dur = info.get("duration", 0.0)
                if actual_dur <= 0:
                    checks.append(_check(
                        f"duration_{label}", "warn",
                        f"Could not read duration for {label}"
                    ))
                    has_warnings = True
                    continue
                deviation = abs(actual_dur - target_duration) / target_duration
                if deviation > self.DURATION_TOLERANCE:
                    status: CheckStatus = "warn"
                    has_warnings = True
                    detail = (
                        f"{label} duration {actual_dur:.2f}s deviates {deviation*100:.1f}% "
                        f"from target {target_duration:.2f}s (>{self.DURATION_TOLERANCE*100:.0f}% threshold)"
                    )
                else:
                    status = "pass"
                    detail = (
                        f"{label} duration {actual_dur:.2f}s — "
                        f"within {deviation*100:.1f}% of target {target_duration:.2f}s"
                    )
                checks.append(_check(f"duration_{label}", status, detail))
        else:
            checks.append(_check(
                "duration", "skip",
                "No duration_target set in blueprint and no assembled clips; skipping duration check"
            ))

        # ── 3. Audio sync check ───────────────────────────────────────────────
        for label, info in video_infos.items():
            path = expected_outputs.get(label, "")
            if not path:
                continue
            try:
                ffprobe_bin = ffmpeg_tools._ffprobe_bin()
                raw = subprocess.run(
                    [
                        ffprobe_bin, "-v", "quiet",
                        "-print_format", "json",
                        "-show_streams",
                        path,
                    ],
                    capture_output=True, text=True, timeout=30,
                )
                if raw.returncode == 0:
                    streams = json.loads(raw.stdout).get("streams", [])
                    vid_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
                    aud_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
                    if vid_stream and aud_stream:
                        v_dur = float(vid_stream.get("duration") or 0)
                        a_dur = float(aud_stream.get("duration") or 0)
                        if v_dur > 0 and a_dur > 0:
                            diff = abs(v_dur - a_dur)
                            if diff > self.AUDIO_SYNC_TOLERANCE:
                                checks.append(_check(
                                    f"audio_sync_{label}", "warn",
                                    f"{label} A/V length mismatch: video={v_dur:.3f}s audio={a_dur:.3f}s "
                                    f"(diff={diff:.3f}s > {self.AUDIO_SYNC_TOLERANCE}s threshold)"
                                ))
                                has_warnings = True
                            else:
                                checks.append(_check(
                                    f"audio_sync_{label}", "pass",
                                    f"{label} audio/video durations match (diff={diff:.3f}s)"
                                ))
                        else:
                            checks.append(_check(
                                f"audio_sync_{label}", "skip",
                                f"Could not read per-stream durations for {label}"
                            ))
                    elif not aud_stream:
                        checks.append(_check(
                            f"audio_sync_{label}", "warn",
                            f"{label} has no audio stream detected"
                        ))
                        has_warnings = True
                    else:
                        checks.append(_check(
                            f"audio_sync_{label}", "skip",
                            f"Could not find both video and audio streams for {label}"
                        ))
                else:
                    checks.append(_check(
                        f"audio_sync_{label}", "skip",
                        f"ffprobe returned non-zero for {label}"
                    ))
            except Exception as exc:
                checks.append(_check(
                    f"audio_sync_{label}", "skip",
                    f"Audio sync check error for {label}: {exc}"
                ))

        # ── 4. Resolution check ───────────────────────────────────────────────
        target_w, target_h = output_cfg.resolution

        if "final_16x9" in video_infos:
            info = video_infos["final_16x9"]
            actual_w = info.get("width", 0)
            actual_h = info.get("height", 0)
            if actual_w == target_w and actual_h == target_h:
                checks.append(_check(
                    "resolution_16x9", "pass",
                    f"16:9 output is {actual_w}x{actual_h} (matches target {target_w}x{target_h})"
                ))
            elif actual_w > 0 and actual_h > 0:
                checks.append(_check(
                    "resolution_16x9", "warn",
                    f"16:9 output is {actual_w}x{actual_h}, expected {target_w}x{target_h}"
                ))
                has_warnings = True
            else:
                checks.append(_check(
                    "resolution_16x9", "fail",
                    "Could not read resolution for 16:9 output"
                ))
                critical_failures.append("export")

        if "final_9x16" in video_infos:
            info = video_infos["final_9x16"]
            actual_w = info.get("width", 0)
            actual_h = info.get("height", 0)
            reframe_cfg = blueprint.reframe_9x16
            expected_9x16_h = reframe_cfg.target_height

            if actual_h == expected_9x16_h:
                checks.append(_check(
                    "resolution_9x16", "pass",
                    f"9:16 output is {actual_w}x{actual_h} (vertical height {actual_h} matches target {expected_9x16_h})"
                ))
            elif actual_h > 0:
                checks.append(_check(
                    "resolution_9x16", "warn",
                    f"9:16 output is {actual_w}x{actual_h}, expected height {expected_9x16_h}"
                ))
                has_warnings = True
            else:
                checks.append(_check(
                    "resolution_9x16", "fail",
                    "Could not read resolution for 9:16 output"
                ))
                critical_failures.append("export")

        # ── 5. File size sanity ───────────────────────────────────────────────
        for label, info in video_infos.items():
            size = info.get("file_size_bytes", 0)
            if size == 0:
                checks.append(_check(
                    f"file_size_{label}", "skip",
                    f"Could not read file size for {label}"
                ))
            elif size < self.MIN_FILE_SIZE_BYTES:
                size_kb = size / 1024
                checks.append(_check(
                    f"file_size_{label}", "fail",
                    f"{label} is only {size_kb:.1f} KB — likely corrupt (threshold: 1 MB)"
                ))
                critical_failures.append("export")
            elif size > self.MAX_FILE_SIZE_BYTES:
                size_gb = size / (1024 ** 3)
                checks.append(_check(
                    f"file_size_{label}", "warn",
                    f"{label} is {size_gb:.2f} GB — may need re-encode (threshold: 2 GB)"
                ))
                has_warnings = True
            else:
                size_mb = size / (1024 * 1024)
                checks.append(_check(
                    f"file_size_{label}", "pass",
                    f"{label} file size is {size_mb:.1f} MB — within acceptable range"
                ))

        # ── 6. Beat alignment audit ───────────────────────────────────────────
        beat_map = blueprint.beat_map
        video_clips = blueprint.tracks.video
        fps = output_cfg.fps

        if beat_map and video_clips and fps > 0:
            frame_duration = 1.0 / fps
            tolerance_secs = self.BEAT_ALIGN_TOLERANCE_FRAMES * frame_duration

            # Sample up to 5 beat-aligned clips for spot-check
            beat_aligned_clips = [c for c in video_clips if c.beat_aligned]
            sample = beat_aligned_clips[:5]

            beat_mismatches = 0
            beat_details: list[str] = []

            for clip in sample:
                cut_time = clip.timeline_position
                nearest_beat = min(beat_map, key=lambda b: abs(b - cut_time))
                diff = abs(cut_time - nearest_beat)
                if diff <= tolerance_secs:
                    beat_details.append(
                        f"clip@{cut_time:.3f}s nearest_beat={nearest_beat:.3f}s diff={diff*1000:.1f}ms OK"
                    )
                else:
                    beat_mismatches += 1
                    beat_details.append(
                        f"clip@{cut_time:.3f}s nearest_beat={nearest_beat:.3f}s diff={diff*1000:.1f}ms MISS"
                    )

            if sample:
                if beat_mismatches == 0:
                    checks.append(_check(
                        "beat_alignment", "pass",
                        f"Beat alignment OK — {len(sample)} sampled cuts all within "
                        f"±{self.BEAT_ALIGN_TOLERANCE_FRAMES} frame(s). "
                        + " | ".join(beat_details)
                    ))
                else:
                    checks.append(_check(
                        "beat_alignment", "warn",
                        f"{beat_mismatches}/{len(sample)} sampled cuts are off-beat (>{tolerance_secs*1000:.1f}ms). "
                        + " | ".join(beat_details)
                    ))
                    has_warnings = True
            else:
                checks.append(_check(
                    "beat_alignment", "skip",
                    "No beat-aligned clips found in blueprint; skipping beat alignment audit"
                ))
        else:
            checks.append(_check(
                "beat_alignment", "skip",
                "No beat_map or video clips in blueprint; skipping beat alignment audit"
            ))

        # ── Score calculation ─────────────────────────────────────────────────
        # pass=full credit, warn=half credit, skip=neutral (full credit), fail=0
        status_weights: dict[str, float] = {"pass": 1.0, "warn": 0.5, "skip": 1.0, "fail": 0.0}
        scored_checks = [c for c in checks if c["status"] != "skip"]
        if scored_checks:
            score = int(
                100 * sum(status_weights[c["status"]] for c in scored_checks) / len(scored_checks)
            )
        else:
            score = 100  # nothing to score — assume ok

        # ── Overall pass/fail ─────────────────────────────────────────────────
        has_critical = len(critical_failures) > 0
        passed = not has_critical

        # ── Routing ───────────────────────────────────────────────────────────
        if has_critical:
            if "assembly" in critical_failures:
                state["next_phase"] = "assembly"
            else:
                state["next_phase"] = "export"
        else:
            state["next_phase"] = "done"

        # ── Write structured result ────────────────────────────────────────────
        qa_result = {
            "passed": passed,
            "checks": checks,
            "score": score,
            "critical_failures": list(set(critical_failures)),
            "has_warnings": has_warnings,
        }
        self.write_result(state, qa_result)

        # ── Human-readable note ───────────────────────────────────────────────
        note_lines = ["QA AGENT REPORT", "=" * 40]
        overall = "PASSED" if passed else "FAILED"
        note_lines.append(
            f"Overall: {overall}  |  Score: {score}/100  |  Next phase: {state['next_phase']}"
        )
        note_lines.append("")

        pass_count = sum(1 for c in checks if c["status"] == "pass")
        warn_count = sum(1 for c in checks if c["status"] == "warn")
        fail_count = sum(1 for c in checks if c["status"] == "fail")
        skip_count = sum(1 for c in checks if c["status"] == "skip")
        note_lines.append(
            f"Checks: {pass_count} passed | {warn_count} warned | {fail_count} failed | {skip_count} skipped"
        )
        note_lines.append("")

        for check in checks:
            icon = {"pass": "[OK]", "warn": "[WARN]", "fail": "[FAIL]", "skip": "[SKIP]"}[check["status"]]
            note_lines.append(f"  {icon} {check['name']}: {check['detail']}")

        note_lines.append("")
        if has_critical:
            note_lines.append(
                f"CRITICAL FAILURES detected — routing back to: {state['next_phase'].upper()}"
            )
            note_lines.append("Director should re-trigger the failed stage.")
        elif has_warnings:
            note_lines.append("Warnings present but no critical failures — marking pipeline as done.")
        else:
            note_lines.append("All checks passed cleanly. Pipeline complete.")

        self.write_note(state, "\n".join(note_lines))
        self.show_panel("QA Report", "\n".join(note_lines))

        return state
