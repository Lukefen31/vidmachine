"""
FFmpeg tool wrappers for final render, audio normalisation, subtitle burn-in,
format conversion, and video inspection.

All functions use ffmpeg-python for filter graph construction and subprocess
for ffprobe queries. The FFmpeg binary path is read from config.settings.ffmpeg_path.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
from pathlib import Path
from typing import Optional

import ffmpeg

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ffmpeg_bin() -> str:
    return settings.ffmpeg_path


def _ffprobe_bin() -> str:
    # Derive ffprobe path from ffmpeg path (same directory, sibling binary)
    ffmpeg_p = Path(settings.ffmpeg_path)
    if ffmpeg_p.parent != Path("."):
        return str(ffmpeg_p.parent / "ffprobe")
    return "ffprobe"


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_final(
    input_path: str,
    output_path: str,
    resolution: tuple[int, int] = (1920, 1080),
    fps: int = 30,
    bitrate: str = "8000k",
    audio_bitrate: str = "192k",
    codec: str = "libx264",
    audio_codec: str = "aac",
) -> dict:
    """
    Final quality render using ffmpeg-python.

    Returns dict: {success, output_path, duration, file_size_mb, error}
    """
    _ensure_dir(output_path)
    try:
        width, height = resolution
        stream = (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                vcodec=codec,
                acodec=audio_codec,
                video_bitrate=bitrate,
                audio_bitrate=audio_bitrate,
                r=fps,
                vf=f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
                movflags="+faststart",
                **{"b:v": bitrate, "b:a": audio_bitrate},
            )
            .overwrite_output()
        )
        stream.run(cmd=_ffmpeg_bin(), capture_stdout=True, capture_stderr=True)

        info = get_video_info(output_path)
        duration = info.get("duration", 0.0)
        file_size_mb = info.get("file_size_bytes", 0) / (1024 * 1024)

        return {
            "success": True,
            "output_path": output_path,
            "duration": duration,
            "file_size_mb": round(file_size_mb, 2),
            "error": None,
        }
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        logger.error("render_final failed: %s", stderr)
        return {"success": False, "output_path": output_path, "duration": 0.0, "file_size_mb": 0.0, "error": stderr}
    except Exception as exc:
        logger.exception("render_final unexpected error")
        return {"success": False, "output_path": output_path, "duration": 0.0, "file_size_mb": 0.0, "error": str(exc)}


def normalize_audio(
    input_path: str,
    output_path: str,
    target_lufs: float = -14.0,
) -> dict:
    """
    Normalise audio loudness to target LUFS using FFmpeg loudnorm filter.

    Two-pass loudnorm: first pass measures actual loudness, second pass applies
    correction so the output hits exactly target_lufs.

    Returns dict: {success, output_path, measured_lufs, error}
    """
    _ensure_dir(output_path)
    measured_lufs: Optional[float] = None

    try:
        # ── Pass 1: measure ────────────────────────────────────────────────────
        probe_result = subprocess.run(
            [
                _ffmpeg_bin(), "-i", input_path,
                "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json",
                "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
        )
        # loudnorm stats are written to stderr
        stderr_text = probe_result.stderr
        json_start = stderr_text.rfind("{")
        json_end = stderr_text.rfind("}") + 1
        loudnorm_stats: dict = {}
        if json_start != -1 and json_end > json_start:
            try:
                loudnorm_stats = json.loads(stderr_text[json_start:json_end])
                measured_lufs = float(loudnorm_stats.get("input_i", target_lufs))
            except json.JSONDecodeError:
                logger.warning("Could not parse loudnorm JSON from ffmpeg stderr")

        # ── Pass 2: apply with measured stats ─────────────────────────────────
        if loudnorm_stats:
            measured_i = loudnorm_stats.get("input_i", str(target_lufs))
            measured_lra = loudnorm_stats.get("input_lra", "11.0")
            measured_tp = loudnorm_stats.get("input_tp", "-1.5")
            measured_offset = loudnorm_stats.get("target_offset", "0.0")
            af_filter = (
                f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:"
                f"measured_I={measured_i}:measured_LRA={measured_lra}:"
                f"measured_TP={measured_tp}:measured_offset={measured_offset}:"
                f"linear=true:print_format=summary"
            )
        else:
            # Fallback: single-pass loudnorm
            af_filter = f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"

        (
            ffmpeg
            .input(input_path)
            .output(output_path, af=af_filter, acodec="aac", audio_bitrate="192k")
            .overwrite_output()
            .run(cmd=_ffmpeg_bin(), capture_stdout=True, capture_stderr=True)
        )

        return {
            "success": True,
            "output_path": output_path,
            "measured_lufs": measured_lufs,
            "error": None,
        }
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        logger.error("normalize_audio failed: %s", stderr)
        return {"success": False, "output_path": output_path, "measured_lufs": measured_lufs, "error": stderr}
    except Exception as exc:
        logger.exception("normalize_audio unexpected error")
        return {"success": False, "output_path": output_path, "measured_lufs": measured_lufs, "error": str(exc)}


def burn_subtitles(
    input_path: str,
    srt_path: str,
    output_path: str,
    font_size: int = 24,
) -> dict:
    """
    Burn SRT subtitles into video using FFmpeg subtitles filter.

    Returns dict: {success, output_path, error}
    """
    _ensure_dir(output_path)
    try:
        # Escape colons/backslashes in the srt path for the subtitles filter
        safe_srt = srt_path.replace("\\", "/").replace(":", "\\:")
        vf_filter = f"subtitles='{safe_srt}':force_style='FontSize={font_size},Outline=1,Shadow=1'"

        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                vf=vf_filter,
                vcodec="libx264",
                acodec="copy",
                video_bitrate="8000k",
                movflags="+faststart",
            )
            .overwrite_output()
            .run(cmd=_ffmpeg_bin(), capture_stdout=True, capture_stderr=True)
        )
        return {"success": True, "output_path": output_path, "error": None}
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        logger.error("burn_subtitles failed: %s", stderr)
        return {"success": False, "output_path": output_path, "error": stderr}
    except Exception as exc:
        logger.exception("burn_subtitles unexpected error")
        return {"success": False, "output_path": output_path, "error": str(exc)}


def crop_and_scale(
    input_path: str,
    output_path: str,
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    output_w: int,
    output_h: int,
) -> dict:
    """
    Apply a static crop and scale. Used for 9:16 reframe of single frames.

    Returns dict: {success, output_path, error}
    """
    _ensure_dir(output_path)
    try:
        vf_filter = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={output_w}:{output_h}"
        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                vf=vf_filter,
                vcodec="libx264",
                acodec="copy",
                video_bitrate="8000k",
                movflags="+faststart",
            )
            .overwrite_output()
            .run(cmd=_ffmpeg_bin(), capture_stdout=True, capture_stderr=True)
        )
        return {"success": True, "output_path": output_path, "error": None}
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        logger.error("crop_and_scale failed: %s", stderr)
        return {"success": False, "output_path": output_path, "error": stderr}
    except Exception as exc:
        logger.exception("crop_and_scale unexpected error")
        return {"success": False, "output_path": output_path, "error": str(exc)}


def apply_reframe_keyframes(
    input_path: str,
    output_path: str,
    keyframes: list[dict],
    output_w: int = 608,
    output_h: int = 1080,
    fps: int = 30,
) -> dict:
    """
    Apply dynamic pan/zoom 9:16 reframe using keyframe crop coordinates.

    keyframes: list of {t, x, y, w, h} dicts (sorted by t).

    For 0 or 1 keyframes, falls back to a static centre crop.
    For 2+ keyframes, builds an FFmpeg sendcmd filter that linearly interpolates
    crop parameters between keyframe timestamps. The sendcmd approach is more
    portable than per-expression crop filters.

    Returns dict: {success, output_path, error}
    """
    _ensure_dir(output_path)

    if not keyframes or len(keyframes) <= 1:
        # Static centre crop fallback
        info = get_video_info(input_path)
        src_w = info.get("width", 1920)
        src_h = info.get("height", 1080)
        crop_h = src_h
        crop_w = int(crop_h * output_w / output_h)
        crop_x = (src_w - crop_w) // 2
        crop_y = 0
        if keyframes and len(keyframes) == 1:
            kf = keyframes[0]
            crop_x = int(kf.get("x", crop_x))
            crop_y = int(kf.get("y", crop_y))
            crop_w = int(kf.get("w", crop_w))
            crop_h = int(kf.get("h", crop_h))
        return crop_and_scale(input_path, output_path, crop_x, crop_y, crop_w, crop_h, output_w, output_h)

    # Sort by time
    kfs = sorted(keyframes, key=lambda k: k["t"])

    # Build a per-frame expression using FFmpeg's crop filter with if/between expressions.
    # We interpolate linearly between consecutive keyframes using FFmpeg's lerp via
    # conditional expressions built from the timeline.
    #
    # Strategy: build a nested if(between(t,...)) expression for each segment.
    def _lerp_expr(t0: float, v0: float, t1: float, v1: float) -> str:
        """Linear interpolation expression between two timestamps."""
        if math.isclose(t0, t1, abs_tol=1e-6):
            return str(int(v0))
        slope = (v1 - v0) / (t1 - t0)
        return f"({v0}+({slope}*(t-{t0})))"

    def _build_piecewise(param: str) -> str:
        """Build a piecewise linear expression over all keyframe segments."""
        n = len(kfs)
        # Before first keyframe: hold first value
        expr = str(int(kfs[0][param]))
        # After last keyframe: hold last value — wrap from the outside
        last_expr = str(int(kfs[-1][param]))

        # Build from last segment inward so inner if clauses are innermost
        # Format: if(lt(t, t0), v0, if(between(t,t0,t1), lerp(...), if(..., ..., vN)))
        inner = last_expr
        for i in range(n - 1, 0, -1):
            t0 = kfs[i - 1]["t"]
            t1 = kfs[i]["t"]
            v0 = float(kfs[i - 1][param])
            v1 = float(kfs[i][param])
            seg_expr = _lerp_expr(t0, v0, t1, v1)
            inner = f"if(between(t,{t0},{t1}),{seg_expr},{inner})"

        # Prepend: if t < first keyframe time, use first value
        first_t = kfs[0]["t"]
        expr = f"if(lt(t,{first_t}),{int(kfs[0]['x'] if param == 'x' else (kfs[0]['y'] if param == 'y' else (kfs[0]['w'] if param == 'w' else kfs[0]['h'])))},{inner})"
        return expr

    x_expr = _build_piecewise("x")
    y_expr = _build_piecewise("y")
    w_expr = _build_piecewise("w")
    h_expr = _build_piecewise("h")

    vf_filter = (
        f"crop=w='{w_expr}':h='{h_expr}':x='{x_expr}':y='{y_expr}',"
        f"scale={output_w}:{output_h}"
    )

    try:
        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                vf=vf_filter,
                vcodec="libx264",
                acodec="copy",
                video_bitrate="8000k",
                r=fps,
                movflags="+faststart",
            )
            .overwrite_output()
            .run(cmd=_ffmpeg_bin(), capture_stdout=True, capture_stderr=True)
        )
        return {"success": True, "output_path": output_path, "error": None}
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        logger.error("apply_reframe_keyframes failed: %s", stderr)
        return {"success": False, "output_path": output_path, "error": stderr}
    except Exception as exc:
        logger.exception("apply_reframe_keyframes unexpected error")
        return {"success": False, "output_path": output_path, "error": str(exc)}


def get_video_info(path: str) -> dict:
    """
    Use ffprobe to get: duration, width, height, fps, codec, audio_codec, file_size_bytes.

    Returns a dict with these keys. Returns an empty dict on error.
    """
    try:
        cmd = [
            _ffprobe_bin(),
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning("ffprobe returned non-zero for %s: %s", path, result.stderr)
            return {}

        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        fmt = data.get("format", {})

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        # Duration: prefer format-level, fall back to stream
        duration = 0.0
        if fmt.get("duration"):
            duration = float(fmt["duration"])
        elif video_stream and video_stream.get("duration"):
            duration = float(video_stream["duration"])

        # FPS: parse "num/den" fraction
        fps = 0.0
        if video_stream:
            r_frame_rate = video_stream.get("r_frame_rate", "0/1")
            try:
                num, den = r_frame_rate.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 0.0
            except (ValueError, ZeroDivisionError):
                fps = 0.0

        return {
            "duration": duration,
            "width": video_stream.get("width", 0) if video_stream else 0,
            "height": video_stream.get("height", 0) if video_stream else 0,
            "fps": round(fps, 3),
            "codec": video_stream.get("codec_name", "") if video_stream else "",
            "audio_codec": audio_stream.get("codec_name", "") if audio_stream else "",
            "file_size_bytes": int(fmt.get("size", 0)),
        }
    except subprocess.TimeoutExpired:
        logger.error("ffprobe timed out for %s", path)
        return {}
    except (json.JSONDecodeError, FileNotFoundError, OSError) as exc:
        logger.error("get_video_info error for %s: %s", path, exc)
        return {}


def add_watermark(
    input_path: str,
    watermark_path: str,
    output_path: str,
    position: str = "bottom_right",
) -> dict:
    """
    Add image watermark overlay.

    position: top_left | top_right | bottom_left | bottom_right
    Returns dict: {success, output_path, error}
    """
    _ensure_dir(output_path)

    padding = 20
    position_map = {
        "top_left":     f"x={padding}:y={padding}",
        "top_right":    f"x=main_w-overlay_w-{padding}:y={padding}",
        "bottom_left":  f"x={padding}:y=main_h-overlay_h-{padding}",
        "bottom_right": f"x=main_w-overlay_w-{padding}:y=main_h-overlay_h-{padding}",
    }
    overlay_pos = position_map.get(position, position_map["bottom_right"])

    try:
        main = ffmpeg.input(input_path)
        wm = ffmpeg.input(watermark_path)
        (
            ffmpeg
            .filter([main.video, wm], "overlay", **dict(item.split("=") for item in overlay_pos.split(":")))
            .output(main.audio, output_path, vcodec="libx264", acodec="copy", movflags="+faststart")
            .overwrite_output()
            .run(cmd=_ffmpeg_bin(), capture_stdout=True, capture_stderr=True)
        )
        return {"success": True, "output_path": output_path, "error": None}
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        logger.error("add_watermark failed: %s", stderr)
        return {"success": False, "output_path": output_path, "error": stderr}
    except Exception as exc:
        logger.exception("add_watermark unexpected error")
        return {"success": False, "output_path": output_path, "error": str(exc)}
