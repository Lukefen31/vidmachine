"""
QA Agent — audits all exported outputs for quality issues.
Decides whether the pipeline is done or needs a re-run of a prior phase.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from agents.base import BaseAgent

if TYPE_CHECKING:
    from state.project_state import ProjectState


class QAAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "qa"

    def run(self, state: "ProjectState") -> "ProjectState":
        self.log("Starting quality audit")

        output_paths: dict = state.get("output_paths", {})
        blueprint: dict = state.get("blueprint", {})
        output_cfg: dict = blueprint.get("output", {})
        duration_target: float | None = output_cfg.get("duration_target")
        expected_res: list = output_cfg.get("resolution", [1920, 1080])
        variants: list = output_cfg.get("variants", ["16:9"])

        checks: list[dict] = []
        critical_failures: list[str] = []
        warnings: list[str] = []

        # ── 1. File existence ─────────────────────────────────────────────────
        for label, path in output_paths.items():
            if path and Path(path).exists():
                checks.append({"name": f"File exists ({label})", "status": "pass", "detail": path})
            else:
                checks.append({"name": f"File exists ({label})", "status": "fail", "detail": f"Not found: {path}"})
                critical_failures.append(f"Output file missing: {label}")

        if not output_paths:
            checks.append({"name": "Output files", "status": "fail", "detail": "No output paths recorded"})
            critical_failures.append("No output files were produced")

        # ── 2. File size sanity ───────────────────────────────────────────────
        for label, path in output_paths.items():
            if path and Path(path).exists():
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                if size_mb < 0.5:
                    checks.append({"name": f"File size ({label})", "status": "fail", "detail": f"{size_mb:.2f} MB — likely corrupt"})
                    critical_failures.append(f"Output {label} is suspiciously small ({size_mb:.2f} MB)")
                elif size_mb > 4096:
                    checks.append({"name": f"File size ({label})", "status": "warn", "detail": f"{size_mb:.0f} MB — very large"})
                    warnings.append(f"Output {label} is very large ({size_mb:.0f} MB)")
                else:
                    checks.append({"name": f"File size ({label})", "status": "pass", "detail": f"{size_mb:.1f} MB"})

        # ── 3. Duration + resolution via ffprobe ──────────────────────────────
        try:
            from tools.ffmpeg_tools import get_video_info
            ffprobe_available = True
        except ImportError:
            ffprobe_available = False
            warnings.append("ffprobe unavailable — skipping duration/resolution checks")

        if ffprobe_available:
            for label, path in output_paths.items():
                if not (path and Path(path).exists()):
                    continue

                info = get_video_info(path)
                if not info:
                    checks.append({"name": f"Probe ({label})", "status": "warn", "detail": "ffprobe returned no data"})
                    warnings.append(f"Could not probe {label}")
                    continue

                # Duration check
                actual_dur = info.get("duration", 0)
                if duration_target and actual_dur > 0:
                    ratio = abs(actual_dur - duration_target) / duration_target
                    if ratio > 0.15:
                        checks.append({
                            "name": f"Duration ({label})",
                            "status": "warn",
                            "detail": f"Expected ~{duration_target:.1f}s, got {actual_dur:.1f}s ({ratio*100:.0f}% off)",
                        })
                        warnings.append(f"Duration mismatch on {label}: {actual_dur:.1f}s vs target {duration_target:.1f}s")
                    else:
                        checks.append({"name": f"Duration ({label})", "status": "pass", "detail": f"{actual_dur:.1f}s"})

                # Resolution check for 16:9
                if label == "final_16x9":
                    w, h = info.get("width", 0), info.get("height", 0)
                    exp_w, exp_h = expected_res[0], expected_res[1]
                    if w != exp_w or h != exp_h:
                        checks.append({
                            "name": f"Resolution ({label})",
                            "status": "warn",
                            "detail": f"Expected {exp_w}x{exp_h}, got {w}x{h}",
                        })
                        warnings.append(f"Resolution mismatch on {label}")
                    else:
                        checks.append({"name": f"Resolution ({label})", "status": "pass", "detail": f"{w}x{h}"})

                # Resolution check for 9:16
                if label == "final_9x16":
                    w, h = info.get("width", 0), info.get("height", 0)
                    if w > h:
                        checks.append({
                            "name": f"Orientation ({label})",
                            "status": "fail",
                            "detail": f"Expected vertical (9:16) but got {w}x{h} (landscape)",
                        })
                        critical_failures.append(f"9:16 export is landscape, not vertical: {w}x{h}")
                    else:
                        checks.append({"name": f"Orientation ({label})", "status": "pass", "detail": f"{w}x{h} ✓ vertical"})

                # Audio/video stream sync
                audio_codec = info.get("audio_codec", "")
                if not audio_codec:
                    checks.append({"name": f"Audio stream ({label})", "status": "warn", "detail": "No audio stream detected"})
                    warnings.append(f"No audio stream in {label}")
                else:
                    checks.append({"name": f"Audio stream ({label})", "status": "pass", "detail": audio_codec})

        # ── 4. Variant coverage ───────────────────────────────────────────────
        if "9:16" in variants and "final_9x16" not in output_paths:
            checks.append({"name": "9:16 variant", "status": "warn", "detail": "Requested but not produced"})
            warnings.append("9:16 variant was requested but not found in output_paths")

        # ── Scoring ───────────────────────────────────────────────────────────
        n_pass = sum(1 for c in checks if c["status"] == "pass")
        n_warn = sum(1 for c in checks if c["status"] == "warn")
        n_fail = sum(1 for c in checks if c["status"] == "fail")
        total = len(checks) or 1
        score = int((n_pass / total) * 100 - (n_warn * 3) - (n_fail * 15))
        score = max(0, min(100, score))

        passed = len(critical_failures) == 0

        result = {
            "passed": passed,
            "score": score,
            "checks": checks,
            "critical_failures": critical_failures,
            "warnings": warnings,
            "n_pass": n_pass,
            "n_warn": n_warn,
            "n_fail": n_fail,
        }

        self.write_result(state, result)
        state["current_phase"] = "qa"

        # ── Routing ───────────────────────────────────────────────────────────
        if passed:
            state["next_phase"] = "done"
            note = (
                f"QA PASSED ✅ Score: {score}/100. "
                f"{n_pass} checks passed, {n_warn} warnings, {n_fail} failures. "
                f"Pipeline complete. Outputs: {list(output_paths.values())}"
            )
            self.log(f"All checks passed — score {score}/100")
        else:
            # Determine which phase to retry
            export_failures = [f for f in critical_failures if "output" in f.lower() or "corrupt" in f.lower()]
            if export_failures:
                state["next_phase"] = "export"
                retry_phase = "export"
            else:
                state["next_phase"] = "assembly"
                retry_phase = "assembly"

            note = (
                f"QA FAILED ❌ Score: {score}/100. "
                f"Critical failures: {critical_failures}. "
                f"Routing back to {retry_phase}."
            )
            self.log(f"Failures detected — routing to {retry_phase}", level="warning")
            for f in critical_failures:
                self.add_error(state, f)

        for w in warnings:
            self.add_warning(state, w)

        self.write_note(state, note)
        self.show_panel("QA Report", note)

        return state
