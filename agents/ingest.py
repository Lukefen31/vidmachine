"""
IngestAgent — first agent to run in the vidmachine pipeline.

Responsibilities
----------------
1. Scan ``<project_dir>/assets/raw/`` for video and audio files.
2. For every **video** file:
   - Read metadata via OpenCV (fps, resolution, duration).
   - Attempt Gyroflow stabilisation → write output to ``assets/processed/``.
   - Build an :class:`~state.asset_manifest.AssetInfo` dict and add it to
     both ``state["raw_assets"]`` and ``state["processed_assets"]``.
3. For every **audio** file:
   - Run librosa analysis (BPM, beat_map, energy_envelope).
   - Classify the track: music (>60 s, low speech confidence) vs.
     speech/vocals (run Whisper if detected).
   - Store the analysis on the AssetInfo dict.
4. Promote the best music track's beat data to
   ``state["blueprint"]["beat_map"]`` and ``state["blueprint"]["bpm"]``.
5. Write a comprehensive memo in ``state["agent_notes"]["ingest"]`` for
   downstream agents (Assembly, Director).
6. Set ``state["current_phase"] = "ingest"`` and
   ``state["next_phase"] = None`` (Director routes next).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.project_state import ProjectState

from agents.base import BaseAgent
from state.asset_manifest import AssetInfo
from state.blueprint import VideoBlueprint
from tools.gyroflow import stabilise_clip
from tools.librosa_tools import analyse_audio
from tools.whisper_tools import transcribe_and_save

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mov", ".avi", ".mkv", ".mts", ".m2ts"}
)
AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".wav", ".flac", ".aac", ".m4a"}
)

# A track must be at least this long (seconds) to be considered a music track
# rather than a short sound effect.
_MUSIC_MIN_DURATION: float = 60.0

# Minimum BPM confidence to trust a beat map
_MIN_TEMPO_CONFIDENCE: float = 0.25


# ---------------------------------------------------------------------------
# OpenCV helper — isolated so failures are clearly scoped
# ---------------------------------------------------------------------------

def _read_video_metadata(video_path: str) -> dict:
    """
    Extract fps, width, height and duration from *video_path* via OpenCV.

    Returns a dict with keys ``fps``, ``width``, ``height``, ``duration``
    (all float/int, or None on failure) plus ``error`` (str).
    """
    result = {
        "fps": None,
        "width": None,
        "height": None,
        "duration": None,
        "error": "",
    }

    try:
        import cv2  # type: ignore[import]
    except ImportError:
        result["error"] = (
            "opencv-python is not installed. "
            "Install it with: pip install opencv-python-headless"
        )
        logger.warning("_read_video_metadata: %s", result["error"])
        return result

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result["error"] = f"OpenCV could not open file: '{video_path}'"
        logger.warning("_read_video_metadata: %s", result["error"])
        return result

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        duration = (frame_count / fps) if (fps and fps > 0) else None

        result.update(
            fps=fps if fps > 0 else None,
            width=width if width > 0 else None,
            height=height if height > 0 else None,
            duration=duration,
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"OpenCV metadata read error: {exc}"
        logger.warning("_read_video_metadata: %s", result["error"])
    finally:
        cap.release()

    return result


# ---------------------------------------------------------------------------
# Speech-detection heuristic
# ---------------------------------------------------------------------------

def _looks_like_speech(audio_analysis: dict) -> bool:
    """
    Rough heuristic: if BPM is very low or tempo_confidence is near zero,
    and energy is relatively constant, it is more likely speech than music.

    This is intentionally conservative — we prefer false negatives (treating
    speech as music) over false positives (running Whisper on every track).
    """
    bpm = audio_analysis.get("bpm", 0.0)
    confidence = audio_analysis.get("tempo_confidence", 0.0)
    beat_map = audio_analysis.get("beat_map", [])

    # Very few beats detected relative to duration — likely not music
    duration = audio_analysis.get("duration", 1.0) or 1.0
    beat_density = len(beat_map) / duration  # beats per second

    # Typical speech: no clear rhythm, very low beat density
    # Typical music: 1–3 beats/s for 60–180 BPM
    if beat_density < 0.3 and confidence < _MIN_TEMPO_CONFIDENCE:
        return True

    # Suspiciously low BPM with low confidence is also speech-like
    if bpm < 40.0 and confidence < 0.4:
        return True

    return False


# ---------------------------------------------------------------------------
# IngestAgent
# ---------------------------------------------------------------------------

class IngestAgent(BaseAgent):
    """Scans raw assets, extracts metadata, stabilises video, analyses audio."""

    @property
    def name(self) -> str:
        return "ingest"

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self, state: "ProjectState") -> "ProjectState":
        project_dir = state.get("project_dir", "")
        if not project_dir:
            self.add_error(state, "project_dir is not set in state.")
            return state

        raw_dir = Path(project_dir) / "assets" / "raw"
        processed_dir = Path(project_dir) / "assets" / "processed"
        subtitles_dir = Path(project_dir) / "assets" / "subtitles"

        for folder in (raw_dir, processed_dir, subtitles_dir):
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self.add_error(state, f"Cannot create directory '{folder}': {exc}")
                return state

        if not raw_dir.is_dir():
            self.add_warning(
                state,
                f"Raw assets directory does not exist: '{raw_dir}'. "
                "Nothing to ingest.",
            )
            state["current_phase"] = "ingest"
            state["next_phase"] = None
            self._finalise(state, [], [], None)
            return state

        # ── Discover files ─────────────────────────────────────────────────────
        all_files = [
            p for p in raw_dir.iterdir()
            if p.is_file() and not p.name.startswith(".")
        ]

        video_files = [f for f in all_files if f.suffix.lower() in VIDEO_EXTENSIONS]
        audio_files = [f for f in all_files if f.suffix.lower() in AUDIO_EXTENSIONS]

        self.log(
            f"Found {len(video_files)} video file(s) and "
            f"{len(audio_files)} audio file(s) in '{raw_dir}'."
        )

        # ── Process video files ────────────────────────────────────────────────
        video_assets: list[dict] = []
        for vf in sorted(video_files):
            asset = self._process_video(vf, processed_dir, state)
            if asset:
                video_assets.append(asset)

        # ── Process audio files ────────────────────────────────────────────────
        audio_assets: list[dict] = []
        best_music_asset: dict | None = None

        for af in sorted(audio_files):
            asset = self._process_audio(af, subtitles_dir, state)
            if asset:
                audio_assets.append(asset)
                # Track the best music candidate for blueprint beat data
                if best_music_asset is None and asset.get("bpm", 0.0) > 0:
                    best_music_asset = asset
                elif best_music_asset is not None:
                    # Prefer the track with higher tempo confidence
                    existing_conf = best_music_asset.get("metadata", {}).get(
                        "tempo_confidence", 0.0
                    )
                    new_conf = asset.get("metadata", {}).get("tempo_confidence", 0.0)
                    if new_conf > existing_conf:
                        best_music_asset = asset

        # ── Update state ───────────────────────────────────────────────────────
        all_raw = (
            state.get("raw_assets") or []
        ) + video_assets + audio_assets

        processed_videos = [
            a for a in video_assets
            if a.get("working_path") and a.get("gyroflow_applied")
        ]
        # All audio assets go straight to processed (no processing step needed)
        all_processed = (
            state.get("processed_assets") or []
        ) + processed_videos + audio_assets

        state["raw_assets"] = all_raw
        state["processed_assets"] = all_processed
        state["current_phase"] = "ingest"
        state["next_phase"] = None

        self._finalise(state, video_assets, audio_assets, best_music_asset)
        return state

    # ── Video processing ──────────────────────────────────────────────────────

    def _process_video(
        self,
        video_path: Path,
        processed_dir: Path,
        state: "ProjectState",
    ) -> dict | None:
        """
        Process a single video file.

        1. Read metadata with OpenCV.
        2. Attempt Gyroflow stabilisation.
        3. Return a serialised AssetInfo dict.
        """
        self.log(f"Processing video: {video_path.name}")

        # Build raw AssetInfo
        asset = AssetInfo(
            asset_type="video_raw",
            original_path=str(video_path.resolve()),
            filename=video_path.name,
            file_size_bytes=self._safe_file_size(video_path),
            status="pending",
        )

        # ── Metadata via OpenCV ────────────────────────────────────────────────
        meta = _read_video_metadata(str(video_path))
        if meta["error"]:
            self.add_warning(
                state,
                f"Video metadata read failed for '{video_path.name}': {meta['error']}",
            )
        else:
            asset.fps = meta["fps"]
            asset.width = meta["width"]
            asset.height = meta["height"]
            asset.duration = meta["duration"]

        # ── Gyroflow stabilisation ─────────────────────────────────────────────
        stabilised_path = processed_dir / video_path.name
        asset.status = "stabilizing"

        stab_result = stabilise_clip(
            input_path=str(video_path),
            output_path=str(stabilised_path),
        )

        if stab_result["success"]:
            asset.working_path = stab_result["output_path"]
            asset.gyroflow_applied = True
            asset.gyroflow_project_path = stab_result.get("gyroflow_project_path", "")
            asset.status = "ready"
            self.log(f"Stabilised '{video_path.name}' → '{stabilised_path.name}'")
        else:
            # Stabilisation failed — use original as working path
            asset.working_path = str(video_path.resolve())
            asset.gyroflow_applied = False
            asset.error = stab_result.get("error", "")
            asset.status = "ready"  # still usable, just not stabilised
            self.add_warning(
                state,
                f"Gyroflow stabilisation skipped for '{video_path.name}': "
                f"{stab_result.get('error', 'unknown error')}",
            )

        asset.asset_type = "video_processed" if asset.gyroflow_applied else "video_raw"
        return asset.model_dump()

    # ── Audio processing ──────────────────────────────────────────────────────

    def _process_audio(
        self,
        audio_path: Path,
        subtitles_dir: Path,
        state: "ProjectState",
    ) -> dict | None:
        """
        Process a single audio file.

        1. Run librosa analysis.
        2. Classify as music or speech.
        3. Run Whisper if speech/vocals detected.
        4. Return a serialised AssetInfo dict.
        """
        self.log(f"Analysing audio: {audio_path.name}")

        asset = AssetInfo(
            asset_type="audio_music",  # default; overridden below
            original_path=str(audio_path.resolve()),
            working_path=str(audio_path.resolve()),
            filename=audio_path.name,
            file_size_bytes=self._safe_file_size(audio_path),
            status="analysing",
        )

        # ── librosa analysis ───────────────────────────────────────────────────
        try:
            audio_analysis = analyse_audio(str(audio_path))
        except Exception as exc:  # noqa: BLE001
            self.add_warning(
                state,
                f"librosa analysis failed for '{audio_path.name}': {exc}",
            )
            audio_analysis = {
                "bpm": 0.0,
                "beat_map": [],
                "energy_envelope": [],
                "duration": 0.0,
                "tempo_confidence": 0.0,
                "key": "unknown",
            }

        asset.bpm = audio_analysis.get("bpm") or 0.0
        asset.beat_map = audio_analysis.get("beat_map") or []
        asset.energy_envelope = audio_analysis.get("energy_envelope") or []
        asset.duration = audio_analysis.get("duration") or 0.0
        asset.metadata["tempo_confidence"] = audio_analysis.get("tempo_confidence", 0.0)
        asset.metadata["key"] = audio_analysis.get("key", "unknown")

        duration = asset.duration or 0.0
        is_music = (
            duration >= _MUSIC_MIN_DURATION
            and not _looks_like_speech(audio_analysis)
        )

        if is_music:
            asset.asset_type = "audio_music"
            asset.tags.append("music")
            self.log(
                f"  → music track: {asset.bpm:.1f} BPM, "
                f"key={audio_analysis.get('key', '?')}, "
                f"duration={duration:.1f}s"
            )
        else:
            # Treat as voiceover / speech — run Whisper
            asset.asset_type = "audio_voiceover"
            asset.tags.append("speech")
            self.log(
                f"  → speech/vocals detected in '{audio_path.name}'. "
                "Running Whisper transcription…"
            )
            asset.status = "transcribing"
            try:
                whisper_result = transcribe_and_save(
                    audio_path=str(audio_path),
                    output_dir=str(subtitles_dir),
                )
                asset.transcript = whisper_result.get("text", "")
                asset.srt_path = whisper_result.get("srt_path", "")
                asset.metadata["language"] = whisper_result.get("language", "")
                asset.metadata["whisper_segments"] = len(
                    whisper_result.get("segments", [])
                )
                self.log(
                    f"  → transcription complete: "
                    f"{len(whisper_result.get('segments', []))} segment(s), "
                    f"language={whisper_result.get('language', '?')}"
                )
            except Exception as exc:  # noqa: BLE001
                self.add_warning(
                    state,
                    f"Whisper transcription failed for '{audio_path.name}': {exc}",
                )

        asset.status = "ready"
        return asset.model_dump()

    # ── Finalise: blueprint + agent note ──────────────────────────────────────

    def _finalise(
        self,
        state: "ProjectState",
        video_assets: list[dict],
        audio_assets: list[dict],
        best_music: dict | None,
    ) -> None:
        """Update blueprint with beat data and write the agent memo."""

        blueprint = VideoBlueprint.from_dict(state["blueprint"])

        if best_music and best_music.get("beat_map"):
            blueprint.beat_map = best_music["beat_map"]
            blueprint.bpm = best_music.get("bpm")
            blueprint.touch()
            self.log(
                f"Blueprint updated with beat map from '{best_music.get('filename', '?')}': "
                f"{blueprint.bpm:.1f} BPM, {len(blueprint.beat_map)} beats."
            )

        state["blueprint"] = blueprint.to_dict()

        # ── Structured phase result ────────────────────────────────────────────
        stabilised_count = sum(
            1 for a in video_assets if a.get("gyroflow_applied")
        )
        music_tracks = [
            a for a in audio_assets if a.get("asset_type") == "audio_music"
        ]
        speech_tracks = [
            a for a in audio_assets if a.get("asset_type") == "audio_voiceover"
        ]

        self.write_result(state, {
            "video_files_found": len(video_assets),
            "audio_files_found": len(audio_assets),
            "videos_stabilised": stabilised_count,
            "music_tracks_found": len(music_tracks),
            "speech_tracks_found": len(speech_tracks),
            "beat_map_populated": bool(blueprint.beat_map),
            "bpm": blueprint.bpm,
        })

        # ── Agent memo for peers ───────────────────────────────────────────────
        note_lines: list[str] = [
            "=== INGEST AGENT REPORT ===",
            "",
            f"Video files found:      {len(video_assets)}",
            f"  Stabilised (Gyroflow): {stabilised_count}",
            f"  Unstabilised (raw):    {len(video_assets) - stabilised_count}",
            "",
            f"Audio files found:      {len(audio_assets)}",
            f"  Music tracks:          {len(music_tracks)}",
            f"  Speech/voiceover:      {len(speech_tracks)}",
            "",
        ]

        if blueprint.beat_map:
            note_lines += [
                "Beat Map:",
                f"  BPM: {blueprint.bpm:.2f}",
                f"  Beats: {len(blueprint.beat_map)} timestamps populated",
                f"  Source: {best_music.get('filename', 'unknown') if best_music else 'unknown'}",
                f"  Key: {best_music.get('metadata', {}).get('key', '?') if best_music else '?'}",
                "",
            ]
        else:
            note_lines += [
                "Beat Map: NOT populated — no suitable music track found.",
                "Assembly agent should use uniform cut spacing or source a music track.",
                "",
            ]

        # Per-video summary
        if video_assets:
            note_lines.append("Video asset details:")
            for va in video_assets:
                fps_str = f"{va.get('fps', 0):.2f}" if va.get("fps") else "?"
                dur_str = f"{va.get('duration', 0):.1f}s" if va.get("duration") else "?"
                res_str = (
                    f"{va.get('width', '?')}x{va.get('height', '?')}"
                    if va.get("width")
                    else "unknown res"
                )
                stab_str = "STABILISED" if va.get("gyroflow_applied") else "raw (no Gyroflow)"
                note_lines.append(
                    f"  - {va.get('filename', '?')} | {res_str} @ {fps_str}fps | "
                    f"{dur_str} | {stab_str}"
                )
            note_lines.append("")

        # Per-audio summary
        if audio_assets:
            note_lines.append("Audio asset details:")
            for aa in audio_assets:
                bpm_str = f"{aa.get('bpm', 0):.1f} BPM" if aa.get("bpm") else "no BPM"
                dur_str = f"{aa.get('duration', 0):.1f}s" if aa.get("duration") else "?"
                type_str = aa.get("asset_type", "?")
                key_str = aa.get("metadata", {}).get("key", "?")
                note_lines.append(
                    f"  - {aa.get('filename', '?')} | {type_str} | "
                    f"{dur_str} | {bpm_str} | key={key_str}"
                )
                if aa.get("srt_path"):
                    note_lines.append(f"    Transcript saved: {aa['srt_path']}")
            note_lines.append("")

        # Warnings summary
        warnings = [
            w for w in (state.get("warnings") or [])
            if w.startswith("[ingest]")
        ]
        if warnings:
            note_lines.append(f"Warnings ({len(warnings)}):")
            for w in warnings:
                note_lines.append(f"  ! {w}")
            note_lines.append("")

        note_lines += [
            "Next steps for Assembly agent:",
            "  - Use processed_assets for video clips (stabilised if Gyroflow ran).",
            "  - Use blueprint.beat_map for beat-aligned cut points.",
            "  - Check srt_path on audio assets for subtitle tracks.",
            "  - Inspect energy_envelope on music assets for drop/peak cut points.",
        ]

        self.write_note(state, "\n".join(note_lines))

        self.show_panel(
            "Ingest Complete",
            f"{len(video_assets)} video(s), {len(audio_assets)} audio file(s) processed.\n"
            f"BPM: {blueprint.bpm or 'N/A'} | "
            f"Beats: {len(blueprint.beat_map)} | "
            f"Stabilised: {stabilised_count}/{len(video_assets)}",
        )

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_file_size(path: Path) -> int:
        """Return file size in bytes, 0 on error."""
        try:
            return path.stat().st_size
        except OSError:
            return 0
