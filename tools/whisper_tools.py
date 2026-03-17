"""
OpenAI Whisper integration — transcription and subtitle generation.

Whisper is run locally (no API key needed). Model size trades accuracy for
speed:  tiny < base < small < medium < large.

Usage:
    from tools.whisper_tools import transcribe, transcribe_and_save, to_srt
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import — whisper is a heavy dependency.
# ---------------------------------------------------------------------------

try:
    import whisper as _whisper
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False
    logger.warning(
        "openai-whisper is not installed. Transcription functions will be unavailable. "
        "Install it with: pip install openai-whisper"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_whisper(fn_name: str) -> bool:
    """Return False and log a clear message if whisper is not installed."""
    if not _WHISPER_AVAILABLE:
        logger.error(
            "%s: openai-whisper is not installed. "
            "Run `pip install openai-whisper` to enable transcription.",
            fn_name,
        )
        return False
    return True


def _empty_result() -> dict:
    return {
        "text": "",
        "segments": [],
        "language": "",
        "word_timestamps": [],
    }


def _format_srt_timestamp(seconds: float) -> str:
    """Convert a float number of seconds to an SRT timestamp string HH:MM:SS,mmm."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    # Clamp milliseconds that rounded up to 1000
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe(audio_path: str, model_size: str = "base") -> dict:
    """
    Transcribe audio using a locally-loaded Whisper model.

    Args:
        audio_path:  Path to the audio or video file to transcribe.
        model_size:  Whisper model variant — one of:
                     ``"tiny"``, ``"base"``, ``"small"``, ``"medium"``, ``"large"``.
                     Larger models are more accurate but slower.

    Returns:
        A dict with the following keys:

        - ``text`` (str):                   Full plain-text transcript.
        - ``segments`` (list[dict]):        Each segment has keys
                                             ``start`` (float), ``end`` (float),
                                             ``text`` (str).
        - ``language`` (str):               Detected language code (e.g. ``"en"``).
        - ``word_timestamps`` (list[dict]): Per-word timing if available.
                                             Each entry has keys
                                             ``word`` (str), ``start`` (float),
                                             ``end`` (float).
                                             Empty list if word-level data is absent.

        Returns an empty result dict on any error.
    """
    if not _require_whisper("transcribe"):
        return _empty_result()

    if not os.path.isfile(audio_path):
        logger.error("transcribe: file not found: '%s'", audio_path)
        return _empty_result()

    try:
        logger.info("transcribe: loading Whisper model '%s'…", model_size)
        model = _whisper.load_model(model_size)
    except Exception as exc:  # noqa: BLE001
        logger.error("transcribe: failed to load model '%s': %s", model_size, exc)
        return _empty_result()

    try:
        logger.info("transcribe: transcribing '%s'…", audio_path)
        result = model.transcribe(
            audio_path,
            word_timestamps=True,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("transcribe: transcription failed for '%s': %s", audio_path, exc)
        return _empty_result()

    # ── Normalise segments ─────────────────────────────────────────────────────
    raw_segments: list[dict] = result.get("segments", [])
    segments: list[dict] = []
    word_timestamps: list[dict] = []

    for seg in raw_segments:
        segments.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": str(seg.get("text", "")).strip(),
        })

        # Word-level timestamps are nested under each segment as "words"
        for w in seg.get("words", []):
            word_timestamps.append({
                "word": str(w.get("word", "")).strip(),
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
            })

    return {
        "text": str(result.get("text", "")).strip(),
        "segments": segments,
        "language": str(result.get("language", "")),
        "word_timestamps": word_timestamps,
    }


def to_srt(segments: list[dict]) -> str:
    """
    Convert a list of Whisper segment dicts to an SRT subtitle string.

    Args:
        segments: List of dicts, each with ``start`` (float), ``end`` (float),
                  and ``text`` (str) keys — as returned by :func:`transcribe`.

    Returns:
        A complete SRT-formatted string ready to be written to a ``.srt`` file.
        Returns an empty string if *segments* is empty.
    """
    if not segments:
        return ""

    lines: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        start_ts = _format_srt_timestamp(float(seg.get("start", 0.0)))
        end_ts = _format_srt_timestamp(float(seg.get("end", 0.0)))
        text = str(seg.get("text", "")).strip()

        lines.append(str(idx))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")  # blank line between entries

    return "\n".join(lines)


def save_srt(segments: list[dict], output_path: str) -> str:
    """
    Convert *segments* to SRT format and write the result to *output_path*.

    Args:
        segments:    Segment dicts from :func:`transcribe`.
        output_path: Destination ``.srt`` file path.  Parent directories are
                     created automatically.

    Returns:
        The absolute path string of the saved file on success.
        Returns the original *output_path* string on failure (file may not exist).
    """
    srt_content = to_srt(segments)
    output_file = Path(output_path)

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(srt_content, encoding="utf-8")
        logger.info("save_srt: wrote %d segments to '%s'.", len(segments), output_path)
        return str(output_file.resolve())
    except OSError as exc:
        logger.error("save_srt: could not write '%s': %s", output_path, exc)
        return output_path


def transcribe_and_save(
    audio_path: str,
    output_dir: str,
    model_size: str = "base",
) -> dict:
    """
    Full pipeline: transcribe an audio file and save the SRT subtitle file.

    The SRT file is saved alongside the transcript as
    ``<output_dir>/<stem>.srt`` where *stem* is the filename without extension.

    Args:
        audio_path:  Source audio or video file.
        output_dir:  Directory where the ``.srt`` file will be saved.
        model_size:  Whisper model variant (see :func:`transcribe`).

    Returns:
        The dict returned by :func:`transcribe`, augmented with an additional
        ``srt_path`` key (str) — the absolute path of the saved SRT file.
        ``srt_path`` is ``""`` if transcription failed or segments were empty.
    """
    result = transcribe(audio_path, model_size=model_size)

    srt_path = ""
    if result["segments"]:
        stem = Path(audio_path).stem
        srt_filename = f"{stem}.srt"
        srt_dest = str(Path(output_dir) / srt_filename)
        srt_path = save_srt(result["segments"], srt_dest)
    else:
        logger.warning(
            "transcribe_and_save: no segments returned for '%s' — SRT not saved.",
            audio_path,
        )

    result["srt_path"] = srt_path
    return result
