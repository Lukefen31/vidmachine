"""
SourcingAgent — acquires all assets required by the VideoBlueprint.

Responsibilities
----------------
1.  Read ``state["user_intent"]`` and ``state["blueprint"]`` to determine
    what needs to be sourced.
2.  For each voiceover track that has script text but no local source file,
    call ElevenLabs to generate an MP3.
3.  If the blueprint has no audio / music tracks, decide whether to:
    - Run MusicGen (when a specific style is mentioned in the intent), or
    - Search Pexels / Pixabay for stock music / ambient sound.
4.  For video tracks with empty source paths, search Pexels / Pixabay for
    stock footage matching the clip's metadata description or the user intent.
5.  Download everything to ``<project_dir>/assets/downloaded/``.
6.  Patch the blueprint's audio, voiceover, and video tracks with the real
    file paths.
7.  Add new ``AssetInfo`` dicts to ``state["processed_assets"]``.
8.  Read the ingest agent's note for additional context.
9.  Write a detailed note for the Assembly agent.
10. Set ``state["current_phase"] = "sourcing"``.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from agents.base import BaseAgent
from state.asset_manifest import AssetInfo
from state.blueprint import AudioTrack, VideoBlueprint, VideoClip
from state.project_state import ProjectState
from tools.elevenlabs_tools import estimate_duration, generate_voiceover
from tools.musicgen_tools import (
    generate_music,
    get_music_prompt_from_intent,
    is_musicgen_available,
)
from tools.stock_api import search_all_sources

logger = logging.getLogger(__name__)

# ── Music-related keywords to detect style intent ─────────────────────────────
_MUSIC_STYLE_KEYWORDS = [
    "music", "beat", "bpm", "soundtrack", "track", "score",
    "cinematic", "electronic", "ambient", "lofi", "lo-fi",
    "epic", "chill", "hip-hop", "orchestral",
]


class SourcingAgent(BaseAgent):
    """Asset-acquisition agent for vidmachine."""

    @property
    def name(self) -> str:
        return "sourcing"

    # ─────────────────────────────────────────────────────────────────────────
    # Entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state: ProjectState) -> ProjectState:
        state["current_phase"] = "sourcing"

        user_intent: str = state.get("user_intent", "")
        project_dir: str = state.get("project_dir", "")
        blueprint_dict: dict = state.get("blueprint", {})
        ingest_note: str = state.get("agent_notes", {}).get("ingest", "")

        if not project_dir:
            self.add_error(state, "project_dir is empty — cannot determine download location.")
            self._write_summary_note(state, [], [], [], error="project_dir not set")
            return state

        download_dir = str(Path(project_dir) / "assets" / "downloaded")
        Path(download_dir).mkdir(parents=True, exist_ok=True)

        # Build a quick lookup of already-processed asset paths to avoid
        # re-downloading files that are already on disk and registered.
        existing_paths: set[str] = {
            a.get("original_path", "") or a.get("working_path", "")
            for a in state.get("processed_assets", [])
        }

        # Parse the blueprint
        blueprint = VideoBlueprint.from_dict(blueprint_dict)

        sourcing_log: list[str] = []   # Plain-English log for the Assembly note
        new_assets: list[AssetInfo] = []
        errors: list[str] = []

        # Read ingest context
        if ingest_note:
            self.log(f"Ingest note: {ingest_note[:120]}{'...' if len(ingest_note) > 120 else ''}")

        # ── 1. Voiceover tracks ───────────────────────────────────────────────
        vo_results = self._source_voiceovers(
            blueprint, download_dir, existing_paths, sourcing_log, new_assets, errors
        )
        _ = vo_results  # mutates blueprint in-place

        # ── 2. Music / audio tracks ───────────────────────────────────────────
        self._source_music(
            blueprint, user_intent, download_dir, existing_paths,
            sourcing_log, new_assets, errors
        )

        # ── 3. Stock video footage ────────────────────────────────────────────
        self._source_stock_video(
            blueprint, user_intent, download_dir, existing_paths,
            sourcing_log, new_assets, errors
        )

        # ── 4. Persist mutations back to state ────────────────────────────────
        blueprint.touch()
        state["blueprint"] = blueprint.to_dict()

        # Merge new assets into processed_assets
        existing_asset_list: list[dict] = state.get("processed_assets", [])
        for asset in new_assets:
            existing_asset_list.append(asset.model_dump())
        state["processed_assets"] = existing_asset_list

        for err in errors:
            self.add_error(state, err)

        # ── 5. Write Assembly note ────────────────────────────────────────────
        self._write_summary_note(state, sourcing_log, new_assets, errors)

        self.write_result(state, {
            "assets_sourced": len(new_assets),
            "errors": errors,
            "download_dir": download_dir,
        })

        self.show_panel(
            "Complete",
            f"Sourced {len(new_assets)} asset(s). "
            f"Download dir: {download_dir}\n" +
            "\n".join(f"  • {line}" for line in sourcing_log[:15]),
        )

        return state

    # ─────────────────────────────────────────────────────────────────────────
    # Voiceover sourcing
    # ─────────────────────────────────────────────────────────────────────────

    def _source_voiceovers(
        self,
        blueprint: VideoBlueprint,
        download_dir: str,
        existing_paths: set[str],
        sourcing_log: list[str],
        new_assets: list[AssetInfo],
        errors: list[str],
    ) -> None:
        """Generate ElevenLabs MP3s for voiceover tracks that have no source file."""
        voiceover_tracks = blueprint.tracks.voiceover

        if not voiceover_tracks:
            self.log("No voiceover tracks in blueprint — skipping TTS.")
            return

        for idx, track in enumerate(voiceover_tracks):
            script_text: str = track.metadata.get("script", "")
            if not script_text:
                # Some agents might store text directly in the source field
                # as a sentinel value starting with "TTS:" or similar
                if track.source and track.source.upper().startswith("TTS:"):
                    script_text = track.source[4:].strip()
                elif track.source and not Path(track.source).exists():
                    # Source looks like raw text, not a path
                    if not os.sep in track.source and "." not in track.source[-5:]:
                        script_text = track.source

            if not script_text:
                self.log(f"Voiceover track {idx}: already has source or no script text — skipping.")
                continue

            # Already sourced?
            if track.source and Path(track.source).exists() and track.source in existing_paths:
                self.log(f"Voiceover track {idx}: file already exists at {track.source}")
                continue

            # Build output path
            vo_filename = f"voiceover_{idx:02d}_{uuid.uuid4().hex[:6]}.mp3"
            vo_path = str(Path(download_dir) / vo_filename)

            self.log(f"Generating voiceover {idx}: {script_text[:60]}...")
            result = generate_voiceover(text=script_text, output_path=vo_path)

            if result["success"]:
                track.source = result["output_path"]
                duration_est = result["duration_estimate"]
                if track.out_point == 0.0:
                    track.out_point = duration_est

                asset = AssetInfo(
                    asset_type="audio_voiceover",
                    original_path=result["output_path"],
                    working_path=result["output_path"],
                    filename=vo_filename,
                    duration=duration_est,
                    status="ready",
                    tags=["voiceover", "tts", "elevenlabs"],
                    metadata={"script": script_text, "track_index": idx},
                )
                new_assets.append(asset)
                existing_paths.add(result["output_path"])
                sourcing_log.append(
                    f"Voiceover {idx}: generated {vo_filename} (~{duration_est:.1f}s)"
                )
                self.log(f"Voiceover {idx} written to {result['output_path']}")
            else:
                msg = f"Voiceover {idx} generation failed: {result['error']}"
                errors.append(msg)
                self.log(msg, level="warning")
                sourcing_log.append(f"Voiceover {idx}: FAILED — {result['error']}")

    # ─────────────────────────────────────────────────────────────────────────
    # Music sourcing
    # ─────────────────────────────────────────────────────────────────────────

    def _source_music(
        self,
        blueprint: VideoBlueprint,
        user_intent: str,
        download_dir: str,
        existing_paths: set[str],
        sourcing_log: list[str],
        new_assets: list[AssetInfo],
        errors: list[str],
    ) -> None:
        """
        Source background music.

        Priority:
        1. If blueprint already has a music track with a valid source, skip.
        2. If user intent mentions a music style and MusicGen is available, generate.
        3. Otherwise search Pexels / Pixabay for stock music / ambient audio.
        """
        # Check if there are existing music tracks with real sources
        existing_music = [
            t for t in blueprint.tracks.audio
            if t.track_type == "music" and t.source and Path(t.source).exists()
        ]
        if existing_music:
            self.log(f"Blueprint already has {len(existing_music)} music track(s) with valid source — skipping.")
            return

        # Check if there's a music track stub (no source yet)
        stub_music_tracks = [
            (i, t) for i, t in enumerate(blueprint.tracks.audio)
            if t.track_type == "music" and (not t.source or not Path(t.source).exists())
        ]

        # Decide whether to use MusicGen
        intent_lower = user_intent.lower()
        music_style_in_intent = any(kw in intent_lower for kw in _MUSIC_STYLE_KEYWORDS)
        use_musicgen = music_style_in_intent and is_musicgen_available()

        target_duration = blueprint.output.duration_target or blueprint.total_video_duration() or 30.0

        if use_musicgen:
            self.log("User intent mentions music style + MusicGen available — generating AI music.")
            music_prompt = get_music_prompt_from_intent(user_intent)
            music_filename = f"music_gen_{uuid.uuid4().hex[:6]}.wav"
            music_path = str(Path(download_dir) / music_filename)

            result = generate_music(
                prompt=music_prompt,
                duration=target_duration,
                output_path=music_path,
            )

            if result["success"]:
                self._attach_music_to_blueprint(
                    blueprint, stub_music_tracks, result["output_path"],
                    result["duration"], "music"
                )
                asset = AssetInfo(
                    asset_type="audio_generated",
                    original_path=result["output_path"],
                    working_path=result["output_path"],
                    filename=music_filename,
                    duration=result["duration"],
                    status="ready",
                    tags=["music", "musicgen", "generated"],
                    metadata={"prompt": result["prompt_used"]},
                )
                new_assets.append(asset)
                existing_paths.add(result["output_path"])
                sourcing_log.append(
                    f"Music: generated via MusicGen ({result['duration']:.1f}s) — "
                    f"prompt: {result['prompt_used'][:60]}"
                )
                self.log(f"MusicGen music written to {result['output_path']}")
                return
            else:
                msg = f"MusicGen failed: {result['error']} — falling back to stock music search."
                self.log(msg, level="warning")
                sourcing_log.append(f"Music: MusicGen failed ({result['error']}), trying stock search.")

        # ── Stock music search fallback ───────────────────────────────────────
        self.log("Searching Pexels/Pixabay for stock music / ambient audio.")
        music_query = _derive_music_query(user_intent)
        downloaded = search_all_sources(
            query=music_query,
            output_dir=download_dir,
            max_results=3,
        )

        if downloaded:
            best = downloaded[0]
            local_path = best["local_path"]
            duration_val = best.get("duration") or target_duration

            self._attach_music_to_blueprint(
                blueprint, stub_music_tracks, local_path, float(duration_val), "music"
            )
            asset = AssetInfo(
                asset_type="audio_music",
                original_path=local_path,
                working_path=local_path,
                filename=Path(local_path).name,
                duration=float(duration_val),
                file_size_bytes=best.get("file_size_bytes", 0),
                status="ready",
                tags=["music", "stock", best.get("source", "unknown")],
                metadata={
                    "source": best.get("source"),
                    "photographer": best.get("photographer"),
                    "query": music_query,
                },
            )
            new_assets.append(asset)
            existing_paths.add(local_path)
            sourcing_log.append(
                f"Music: downloaded stock track from {best.get('source')} "
                f"({duration_val:.1f}s) — {Path(local_path).name}"
            )
        else:
            msg = "Could not source any background music — blueprint will have no music track."
            self.log(msg, level="warning")
            sourcing_log.append(f"Music: no results found for query {music_query!r}")

    # ─────────────────────────────────────────────────────────────────────────
    # Stock video sourcing
    # ─────────────────────────────────────────────────────────────────────────

    def _source_stock_video(
        self,
        blueprint: VideoBlueprint,
        user_intent: str,
        download_dir: str,
        existing_paths: set[str],
        sourcing_log: list[str],
        new_assets: list[AssetInfo],
        errors: list[str],
    ) -> None:
        """
        Download stock footage for video tracks that have empty or missing
        source paths.
        """
        empty_video_tracks = [
            (i, clip) for i, clip in enumerate(blueprint.tracks.video)
            if not clip.source or not Path(clip.source).exists()
        ]

        if not empty_video_tracks:
            self.log("All video tracks already have source files — skipping stock video search.")
            return

        self.log(f"{len(empty_video_tracks)} video track(s) need stock footage.")

        for track_idx, clip in empty_video_tracks:
            # Derive search query from clip metadata description, then fall
            # back to the user intent
            clip_description: str = (
                clip.metadata.get("description")
                or clip.metadata.get("query")
                or clip.metadata.get("subject")
                or ""
            )
            search_query = clip_description or user_intent or "cinematic footage"
            search_query = search_query[:100]  # API query length cap

            # Target duration for the clip (add a buffer so we have room to trim)
            clip_duration_needed = clip.duration if clip.duration > 0 else 10.0
            min_duration = max(3, int(clip_duration_needed) - 2)
            max_duration = int(clip_duration_needed) + 30

            self.log(
                f"Track {track_idx}: searching for {search_query!r} "
                f"(need ~{clip_duration_needed:.1f}s)"
            )

            downloaded = search_all_sources(
                query=search_query,
                output_dir=download_dir,
                max_results=3,
            )

            if not downloaded:
                # Widen the search to just the user intent
                if clip_description and clip_description != user_intent:
                    self.log(
                        f"Track {track_idx}: no results for description {search_query!r}, "
                        "retrying with user intent."
                    )
                    downloaded = search_all_sources(
                        query=user_intent or "cinematic footage",
                        output_dir=download_dir,
                        max_results=3,
                    )

            if downloaded:
                best = downloaded[0]
                local_path = best["local_path"]

                # Patch the clip source and timing
                clip.source = local_path
                actual_duration = float(best.get("duration") or clip_duration_needed)
                if clip.out_point == 0.0 or clip.out_point > actual_duration:
                    clip.in_point = 0.0
                    clip.out_point = min(actual_duration, clip_duration_needed or actual_duration)

                asset = AssetInfo(
                    asset_type="video_stock",
                    original_path=local_path,
                    working_path=local_path,
                    filename=Path(local_path).name,
                    duration=actual_duration,
                    width=best.get("width"),
                    height=best.get("height"),
                    file_size_bytes=best.get("file_size_bytes", 0),
                    status="ready",
                    tags=["stock", "video", best.get("source", "unknown")],
                    metadata={
                        "source": best.get("source"),
                        "photographer": best.get("photographer"),
                        "query": search_query,
                        "track_index": track_idx,
                    },
                )
                new_assets.append(asset)
                existing_paths.add(local_path)
                sourcing_log.append(
                    f"Video track {track_idx}: downloaded {Path(local_path).name} "
                    f"from {best.get('source')} "
                    f"({actual_duration:.1f}s, {best.get('width')}x{best.get('height')})"
                )
                self.log(f"Track {track_idx}: sourced {local_path}")
            else:
                msg = (
                    f"Video track {track_idx}: no stock footage found for "
                    f"{search_query!r} — track will have empty source."
                )
                errors.append(msg)
                self.log(msg, level="warning")
                sourcing_log.append(f"Video track {track_idx}: NO FOOTAGE FOUND for {search_query!r}")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _attach_music_to_blueprint(
        blueprint: VideoBlueprint,
        stub_tracks: list[tuple[int, Any]],
        file_path: str,
        duration: float,
        track_type: str,
    ) -> None:
        """
        Either update the first stub music track or append a new AudioTrack.
        """
        if stub_tracks:
            idx, track = stub_tracks[0]
            track.source = file_path
            if track.out_point == 0.0:
                track.out_point = duration
        else:
            new_track = AudioTrack(
                source=file_path,
                in_point=0.0,
                out_point=duration,
                timeline_position=0.0,
                volume=0.8,
                fade_in=2.0,
                fade_out=3.0,
                track_type=track_type,  # type: ignore[arg-type]
            )
            blueprint.tracks.audio.append(new_track)

    def _write_summary_note(
        self,
        state: ProjectState,
        sourcing_log: list[str],
        new_assets: list[AssetInfo],
        errors: list[str],
        error: str = "",
    ) -> None:
        """Compose and write the agent note for the Assembly agent."""
        lines: list[str] = [
            f"SourcingAgent completed. {len(new_assets)} asset(s) acquired.",
        ]

        if error:
            lines.append(f"FATAL: {error}")
        else:
            voiceovers = [a for a in new_assets if a.asset_type == "audio_voiceover"]
            music = [a for a in new_assets if a.asset_type in ("audio_music", "audio_generated")]
            videos = [a for a in new_assets if a.asset_type == "video_stock"]

            if voiceovers:
                lines.append(
                    f"Voiceovers generated: {len(voiceovers)} file(s). "
                    "Blueprint voiceover tracks have been updated with real paths."
                )
            if music:
                source_types = ", ".join({a.asset_type for a in music})
                lines.append(
                    f"Music track(s) sourced: {len(music)} ({source_types}). "
                    "Blueprint audio tracks updated."
                )
            if videos:
                lines.append(
                    f"Stock video clips downloaded: {len(videos)}. "
                    "Blueprint video tracks updated with file paths and trimmed durations."
                )

            if sourcing_log:
                lines.append("Details:")
                lines.extend(f"  {entry}" for entry in sourcing_log)

        if errors:
            lines.append(f"Errors encountered ({len(errors)}):")
            lines.extend(f"  ! {e}" for e in errors)

        lines.append(
            "Assembly: all blueprint tracks with valid source paths are ready to cut. "
            "Tracks with empty sources were unfulfilled — handle gracefully."
        )

        self.write_note(state, "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _derive_music_query(user_intent: str) -> str:
    """
    Build a stock-music search query from the user intent.

    Keeps it concise enough for API search but descriptive enough to find
    relevant results.
    """
    if not user_intent:
        return "ambient background music"

    intent_lower = user_intent.lower()

    # Map broad intent themes to search-friendly music queries
    mapping: list[tuple[list[str], str]] = [
        (["fpv", "drone"], "cinematic action music"),
        (["travel", "vlog"], "uplifting travel music"),
        (["workout", "gym", "fitness"], "energetic workout music"),
        (["wedding"], "romantic wedding music"),
        (["nature", "wildlife"], "ambient nature music"),
        (["corporate", "startup", "tech"], "corporate background music"),
        (["gaming", "esports"], "gaming electronic music"),
        (["cinematic", "film", "trailer"], "cinematic epic music"),
        (["relax", "calm", "meditation"], "relaxing ambient music"),
    ]

    for keywords, query in mapping:
        if any(kw in intent_lower for kw in keywords):
            return query

    return "background music"
