"""
MoviePy helpers for video assembly.

These functions translate VideoBlueprint data into MoviePy composites and
write a draft MP4.  All public functions are intentionally pure-ish (no
global state) so they can be tested in isolation.

Heavy imports (moviepy, numpy) are kept inside this module; nothing at the
top of the package graph needs to pay that import cost at startup.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy MoviePy import helper
# ---------------------------------------------------------------------------

def _moviepy():
    """Return the moviepy.editor module, raising ImportError with a friendly
    message if moviepy is not installed."""
    try:
        import moviepy.editor as mpe  # noqa: PLC0415
        return mpe
    except ImportError as exc:
        raise ImportError(
            "moviepy is required for video assembly. "
            "Install it with:  pip install moviepy"
        ) from exc


# ---------------------------------------------------------------------------
# load_clip
# ---------------------------------------------------------------------------

def load_clip(source: str, in_point: float, out_point: float):
    """
    Load a subclip from *source* between *in_point* and *out_point* (seconds).

    Returns a MoviePy VideoFileClip subclip.
    Raises FileNotFoundError if the source path does not exist.
    Raises ValueError if in_point >= out_point.
    """
    mpe = _moviepy()

    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source file not found: '{source}'")

    if in_point >= out_point:
        raise ValueError(
            f"in_point ({in_point}) must be less than out_point ({out_point}) "
            f"for source '{source}'"
        )

    logger.debug(
        "load_clip: '%s'  %.3f → %.3f", source, in_point, out_point
    )
    clip = mpe.VideoFileClip(str(source_path))
    return clip.subclip(in_point, out_point)


# ---------------------------------------------------------------------------
# apply_transition_out
# ---------------------------------------------------------------------------

def apply_transition_out(clip, transition: dict):
    """
    Apply a transition effect to the *end* of *clip*.

    Supported transition types (keyed from transition["type"]):
        - "cut"      — no-op, clip returned unchanged
        - "dissolve" — crossfadeout for transition["duration"] seconds
        - "fade"     — fadeout to black for transition["duration"] seconds
        - "flash"    — ramp brightness to white and back over the last
                       transition["duration"] seconds
        - "wipe"     — not supported at this stage; falls back to cut with
                       a warning

    Args:
        clip:        A MoviePy VideoFileClip (or any VideoClip subclass).
        transition:  dict with keys "type" (str) and "duration" (float).

    Returns:
        Modified (or original) VideoClip.
    """
    mpe = _moviepy()

    t_type: str = transition.get("type", "cut")
    t_dur: float = float(transition.get("duration", 0.0))

    if t_type == "cut" or t_dur <= 0.0:
        return clip

    clip_dur = clip.duration

    if t_type == "dissolve":
        logger.debug("apply_transition_out: crossfadeout %.2fs", t_dur)
        return clip.crossfadeout(min(t_dur, clip_dur))

    if t_type == "fade":
        logger.debug("apply_transition_out: fadeout %.2fs", t_dur)
        return clip.fadeout(min(t_dur, clip_dur))

    if t_type == "flash":
        logger.debug("apply_transition_out: flash %.2fs", t_dur)
        # Build a brightness-spike effect: ramp up to white then back down
        # over the last t_dur seconds of the clip.
        import numpy as np  # noqa: PLC0415

        effect_start = max(0.0, clip_dur - t_dur)

        def _flash_filter(get_frame, t):
            frame = get_frame(t)
            if t >= effect_start:
                progress = (t - effect_start) / t_dur  # 0 → 1
                # Sine ramp: peaks at mid-point, then falls back
                brightness = float(np.sin(progress * np.pi))
                white = np.ones_like(frame, dtype=np.float64) * 255.0
                blended = frame.astype(np.float64) * (1.0 - brightness) + white * brightness
                return np.clip(blended, 0, 255).astype(np.uint8)
            return frame

        return clip.fl(_flash_filter, apply_to=["mask"])

    if t_type == "wipe":
        logger.warning(
            "apply_transition_out: 'wipe' is not implemented; falling back to cut."
        )
        return clip

    logger.warning(
        "apply_transition_out: unknown transition type '%s'; treating as cut.",
        t_type,
    )
    return clip


# ---------------------------------------------------------------------------
# apply_transition_in
# ---------------------------------------------------------------------------

def apply_transition_in(clip, transition: dict):
    """
    Apply a transition effect to the *start* of *clip*.

    Supported types mirror apply_transition_out (crossfadein / fadein / flash).
    """
    mpe = _moviepy()

    t_type: str = transition.get("type", "cut")
    t_dur: float = float(transition.get("duration", 0.0))

    if t_type == "cut" or t_dur <= 0.0:
        return clip

    clip_dur = clip.duration

    if t_type == "dissolve":
        logger.debug("apply_transition_in: crossfadein %.2fs", t_dur)
        return clip.crossfadein(min(t_dur, clip_dur))

    if t_type == "fade":
        logger.debug("apply_transition_in: fadein %.2fs", t_dur)
        return clip.fadein(min(t_dur, clip_dur))

    if t_type == "flash":
        logger.debug("apply_transition_in: flash-in %.2fs", t_dur)
        import numpy as np  # noqa: PLC0415

        effect_end = min(t_dur, clip_dur)

        def _flash_in_filter(get_frame, t):
            frame = get_frame(t)
            if t <= effect_end:
                progress = t / effect_end  # 0 → 1
                brightness = float(1.0 - progress)  # fade from white
                white = np.ones_like(frame, dtype=np.float64) * 255.0
                blended = frame.astype(np.float64) * (1.0 - brightness) + white * brightness
                return np.clip(blended, 0, 255).astype(np.uint8)
            return frame

        return clip.fl(_flash_in_filter, apply_to=["mask"])

    if t_type == "wipe":
        logger.warning(
            "apply_transition_in: 'wipe' is not implemented; falling back to cut."
        )
        return clip

    logger.warning(
        "apply_transition_in: unknown transition type '%s'; treating as cut.",
        t_type,
    )
    return clip


# ---------------------------------------------------------------------------
# add_text_overlay
# ---------------------------------------------------------------------------

def add_text_overlay(clip, overlay: dict):
    """
    Burn a text overlay onto *clip*.

    The *overlay* dict mirrors the TextOverlay model:
        text       (str)              — displayed string
        start      (float)            — timeline seconds (relative to clip start)
        end        (float)            — timeline seconds (relative to clip start)
        position   (tuple[float,float]) — normalised (x, y), e.g. (0.5, 0.9)
        font_size  (int)              — point size
        color      (str)              — e.g. "white", "#FF0000"
        font       (str)              — font name (must be available to ImageMagick)

    Returns a CompositeVideoClip with the text burned in.
    """
    mpe = _moviepy()

    text: str = overlay.get("text", "")
    if not text.strip():
        return clip

    start: float = float(overlay.get("start", 0.0))
    end: float = float(overlay.get("end", clip.duration))
    position: tuple = overlay.get("position", (0.5, 0.9))
    font_size: int = int(overlay.get("font_size", 48))
    color: str = overlay.get("color", "white")
    font: str = overlay.get("font", "Arial")

    # Clamp to clip bounds
    start = max(0.0, min(start, clip.duration))
    end = max(start, min(end, clip.duration))
    text_duration = end - start

    if text_duration <= 0:
        logger.warning("add_text_overlay: zero-duration overlay '%s' skipped.", text)
        return clip

    # Convert normalised position to MoviePy ('center', relative) format.
    # MoviePy accepts ("center", "center") or pixel offsets; we use relative
    # fractions via lambda.
    def _pos(clip_size):
        w, h = clip_size
        return (int(position[0] * w), int(position[1] * h))

    try:
        txt_clip = (
            mpe.TextClip(
                text,
                fontsize=font_size,
                color=color,
                font=font,
                method="caption",
                size=(clip.w, None),
            )
            .set_position(lambda s: _pos((clip.w, clip.h)))
            .set_start(start)
            .set_duration(text_duration)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "add_text_overlay: failed to create TextClip for '%s': %s — skipping.",
            text,
            exc,
        )
        return clip

    composite = mpe.CompositeVideoClip([clip, txt_clip])
    composite = composite.set_duration(clip.duration)
    return composite


# ---------------------------------------------------------------------------
# sequence_clips
# ---------------------------------------------------------------------------

def sequence_clips(video_clips: list[dict], fps: int = 30):
    """
    Sequence a list of VideoClip dicts (from blueprint.tracks.video) into a
    single MoviePy composite.

    Each dict is expected to have (at minimum):
        source            str
        in_point          float
        out_point         float
        timeline_position float
        transition_in     dict  (type, duration)
        transition_out    dict  (type, duration)
        volume            float  (0.0 = natural audio, -1 = strip audio)

    Clips are placed at their timeline_position.  Missing source files are
    skipped with a WARNING (never crash).

    Returns a CompositeVideoClip, or raises RuntimeError if no clips could
    be loaded.
    """
    mpe = _moviepy()

    if not video_clips:
        raise RuntimeError("sequence_clips: video_clips list is empty.")

    placed_clips = []
    total_duration = 0.0

    for clip_dict in video_clips:
        source = clip_dict.get("source", "")
        in_point = float(clip_dict.get("in_point", 0.0))
        out_point = float(clip_dict.get("out_point", 0.0))
        timeline_pos = float(clip_dict.get("timeline_position", 0.0))
        t_in = clip_dict.get("transition_in", {"type": "cut", "duration": 0.0})
        t_out = clip_dict.get("transition_out", {"type": "cut", "duration": 0.0})
        volume = float(clip_dict.get("volume", 0.0))

        # Validate source
        if not source or not Path(source).is_file():
            logger.warning(
                "sequence_clips: source not found, skipping clip — '%s'", source
            )
            continue

        if in_point >= out_point:
            logger.warning(
                "sequence_clips: invalid in/out points (%.3f / %.3f) for '%s', skipping.",
                in_point,
                out_point,
                source,
            )
            continue

        try:
            clip = load_clip(source, in_point, out_point)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "sequence_clips: could not load '%s': %s — skipping.", source, exc
            )
            continue

        # Apply transitions
        clip = apply_transition_in(clip, t_in)
        clip = apply_transition_out(clip, t_out)

        # Handle audio volume on the clip's natural audio track
        if volume == -1.0:
            clip = clip.without_audio()
        elif volume == 0.0:
            # Natural audio — leave as-is
            pass
        else:
            if clip.audio is not None:
                clip = clip.set_audio(clip.audio.volumex(volume))

        clip = clip.set_start(timeline_pos)
        placed_clips.append(clip)

        clip_end = timeline_pos + clip.duration
        if clip_end > total_duration:
            total_duration = clip_end

    if not placed_clips:
        raise RuntimeError(
            "sequence_clips: all clips were skipped — no valid source files found."
        )

    logger.info(
        "sequence_clips: compositing %d clip(s) over %.2fs timeline.",
        len(placed_clips),
        total_duration,
    )

    composite = mpe.CompositeVideoClip(placed_clips, size=placed_clips[0].size)
    composite = composite.set_duration(total_duration).set_fps(fps)
    return composite


# ---------------------------------------------------------------------------
# mix_audio_tracks
# ---------------------------------------------------------------------------

def mix_audio_tracks(audio_tracks: list[dict], duration: float):
    """
    Mix multiple audio track dicts into a single composite AudioFileClip.

    Each dict mirrors the AudioTrack model:
        source            str
        in_point          float
        out_point         float
        timeline_position float
        volume            float   (1.0 = unity gain)
        fade_in           float   (seconds)
        fade_out          float   (seconds)
        track_type        str     (ignored here; used by caller for routing)

    Missing source files are skipped with a WARNING.

    Returns a CompositeAudioClip or None if all tracks were skipped.
    """
    mpe = _moviepy()

    if not audio_tracks:
        return None

    mixed: list = []

    for track_dict in audio_tracks:
        source = track_dict.get("source", "")
        in_point = float(track_dict.get("in_point", 0.0))
        out_point = float(track_dict.get("out_point", 0.0))
        timeline_pos = float(track_dict.get("timeline_position", 0.0))
        volume = float(track_dict.get("volume", 1.0))
        fade_in_dur = float(track_dict.get("fade_in", 0.5))
        fade_out_dur = float(track_dict.get("fade_out", 1.0))

        if not source or not Path(source).is_file():
            logger.warning(
                "mix_audio_tracks: source not found, skipping — '%s'", source
            )
            continue

        try:
            audio = mpe.AudioFileClip(str(source))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "mix_audio_tracks: could not load '%s': %s — skipping.", source, exc
            )
            continue

        # Subclip if in/out points are specified
        if out_point > in_point:
            audio = audio.subclip(in_point, min(out_point, audio.duration))

        # Volume
        if volume != 1.0:
            audio = audio.volumex(volume)

        # Fade in / out
        if fade_in_dur > 0:
            audio = audio.audio_fadein(min(fade_in_dur, audio.duration))
        if fade_out_dur > 0:
            audio = audio.audio_fadeout(min(fade_out_dur, audio.duration))

        # Place on timeline
        audio = audio.set_start(timeline_pos)

        mixed.append(audio)

    if not mixed:
        logger.warning("mix_audio_tracks: no valid audio tracks; returning None.")
        return None

    composite = mpe.CompositeAudioClip(mixed)
    composite = composite.set_duration(duration)
    return composite


# ---------------------------------------------------------------------------
# assemble_draft
# ---------------------------------------------------------------------------

def assemble_draft(blueprint_dict: dict, output_path: str) -> dict:
    """
    Main assembly function.  Takes the full VideoBlueprint as a dict, assembles
    video + audio, and writes draft.mp4 to *output_path*.

    Steps:
        1. Sequence all video clips from blueprint["tracks"]["video"]
        2. Burn all text overlays from blueprint["tracks"]["text_overlays"]
        3. Mix audio tracks from music + voiceover + sfx sub-lists
        4. Replace clip audio with mixed composite
        5. Write output using clip.write_videofile() with libx264 / aac

    Returns:
        dict with keys: success (bool), output_path (str), duration (float),
                        error (str), skipped_clips (list[str])
    """
    mpe = _moviepy()

    result: dict = {
        "success": False,
        "output_path": output_path,
        "duration": 0.0,
        "error": "",
        "skipped_clips": [],
    }

    # ── Pull data from blueprint ──────────────────────────────────────────────
    tracks: dict = blueprint_dict.get("tracks", {})
    video_clips: list[dict] = tracks.get("video", [])
    text_overlays: list[dict] = tracks.get("text_overlays", [])

    music_tracks: list[dict] = tracks.get("audio", [])
    voiceover_tracks: list[dict] = tracks.get("voiceover", [])
    sfx_tracks: list[dict] = tracks.get("sfx", [])
    all_audio_tracks: list[dict] = music_tracks + voiceover_tracks + sfx_tracks

    output_cfg: dict = blueprint_dict.get("output", {})
    fps: int = int(output_cfg.get("fps", 30))
    codec: str = output_cfg.get("codec", "libx264")
    audio_codec: str = output_cfg.get("audio_codec", "aac")
    bitrate: str = output_cfg.get("bitrate", "8000k")
    audio_bitrate: str = output_cfg.get("audio_bitrate", "192k")

    # Track which clips were skipped for the agent note
    skipped: list[str] = []
    for clip_dict in video_clips:
        src = clip_dict.get("source", "")
        if src and not Path(src).is_file():
            skipped.append(src)

    result["skipped_clips"] = skipped

    # ── Step 1: Sequence video ────────────────────────────────────────────────
    if not video_clips:
        result["error"] = "blueprint has no video clips"
        logger.error("assemble_draft: %s", result["error"])
        return result

    try:
        video = sequence_clips(video_clips, fps=fps)
    except RuntimeError as exc:
        result["error"] = str(exc)
        logger.error("assemble_draft: sequence_clips failed — %s", exc)
        return result

    total_duration: float = video.duration

    # ── Step 2: Burn text overlays ────────────────────────────────────────────
    for overlay in text_overlays:
        try:
            video = add_text_overlay(video, overlay)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "assemble_draft: text overlay '%s' failed: %s — skipping.",
                overlay.get("text", ""),
                exc,
            )

    # ── Step 3: Mix audio ─────────────────────────────────────────────────────
    mixed_audio = None
    if all_audio_tracks:
        try:
            mixed_audio = mix_audio_tracks(all_audio_tracks, total_duration)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "assemble_draft: audio mixing failed: %s — proceeding without mixed audio.",
                exc,
            )

    # ── Step 4: Attach audio to composite ────────────────────────────────────
    if mixed_audio is not None:
        video = video.set_audio(mixed_audio)
    # If no external audio was mixed, the natural clip audio (if any) is
    # preserved by default from sequence_clips.

    # ── Step 5: Ensure output directory exists ────────────────────────────────
    output_file = Path(output_path)
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        result["error"] = f"Cannot create output directory '{output_file.parent}': {exc}"
        logger.error("assemble_draft: %s", result["error"])
        return result

    # ── Step 6: Write output ──────────────────────────────────────────────────
    logger.info(
        "assemble_draft: writing draft to '%s' (%.2fs, fps=%d, codec=%s)",
        output_path,
        total_duration,
        fps,
        codec,
    )

    try:
        video.write_videofile(
            str(output_file),
            fps=fps,
            codec=codec,
            audio_codec=audio_codec,
            bitrate=bitrate,
            audio_bitrate=audio_bitrate,
            temp_audiofile=str(
                output_file.parent / f"_tmp_audio_{output_file.stem}.aac"
            ),
            remove_temp=True,
            logger="bar",
            threads=os.cpu_count() or 4,
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"write_videofile failed: {exc}"
        logger.exception("assemble_draft: write_videofile raised an exception")
        return result
    finally:
        # Always close clips to free file handles / memory
        try:
            video.close()
        except Exception:  # noqa: BLE001
            pass
        if mixed_audio is not None:
            try:
                mixed_audio.close()
            except Exception:  # noqa: BLE001
                pass

    if not output_file.is_file():
        result["error"] = (
            f"write_videofile reported success but output not found at '{output_path}'"
        )
        logger.error("assemble_draft: %s", result["error"])
        return result

    result["success"] = True
    result["duration"] = round(total_duration, 3)
    logger.info(
        "assemble_draft: draft written successfully — %.2fs at '%s'",
        total_duration,
        output_path,
    )
    return result
