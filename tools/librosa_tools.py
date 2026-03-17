"""
Audio analysis tools built on librosa.

These functions power beat-aligned editing: they analyse a music track and
return the exact timestamps of every beat, the overall BPM, a normalised
energy envelope, and the best candidate cut points.

All public functions degrade gracefully — on failure they log a warning and
return empty / zero values so the rest of the pipeline can continue.

Usage:
    from tools.librosa_tools import analyse_audio, align_cuts_to_beats
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import — librosa is a heavy dependency; fail loudly but gracefully.
# ---------------------------------------------------------------------------

try:
    import librosa
    import numpy as np
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False
    logger.warning(
        "librosa is not installed. Audio analysis functions will return empty results. "
        "Install it with: pip install librosa"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_librosa(fn_name: str) -> bool:
    """Log a warning and return False if librosa is unavailable."""
    if not _LIBROSA_AVAILABLE:
        logger.warning(
            "%s: librosa is not available — returning empty result.", fn_name
        )
        return False
    return True


def _load_audio(audio_path: str, fn_name: str):
    """
    Load an audio file with librosa.

    Returns (y, sr) on success, (None, None) on failure.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        return y, sr
    except FileNotFoundError:
        logger.warning("%s: audio file not found: '%s'", fn_name, audio_path)
        return None, None
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s: failed to load '%s': %s", fn_name, audio_path, exc)
        return None, None


def _detect_key(y, sr) -> str:
    """
    Attempt to detect the musical key using a chromagram.

    Returns a string like "C major" or "A# minor".
    Falls back to "unknown" on any error.
    """
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)

        # Krumhansl–Schmuckler key profiles (major / minor)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                   2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                   2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        note_names = ["C", "C#", "D", "D#", "E", "F",
                      "F#", "G", "G#", "A", "A#", "B"]

        best_score = -float("inf")
        best_key = "unknown"

        for i in range(12):
            rotated_chroma = np.roll(chroma_mean, -i)

            # Normalise both vectors before correlation
            c = rotated_chroma / (np.linalg.norm(rotated_chroma) + 1e-9)

            for profile, quality in ((major_profile, "major"), (minor_profile, "minor")):
                p = profile / (np.linalg.norm(profile) + 1e-9)
                score = float(np.dot(c, p))
                if score > best_score:
                    best_score = score
                    best_key = f"{note_names[i]} {quality}"

        return best_key
    except Exception as exc:  # noqa: BLE001
        logger.debug("Key detection failed: %s", exc)
        return "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_audio(audio_path: str) -> dict:
    """
    Full audio analysis of a single file.

    Args:
        audio_path: Path to the audio file (.mp3, .wav, .flac, etc.).

    Returns:
        A dict with the following keys:

        - ``bpm`` (float):               Estimated beats-per-minute.
        - ``beat_map`` (list[float]):     Beat timestamps in seconds.
        - ``energy_envelope`` (list[float]): Normalised RMS energy sampled at 10 Hz
                                          (values in [0.0, 1.0]).
        - ``duration`` (float):          Total duration in seconds.
        - ``tempo_confidence`` (float):  Confidence score for the BPM estimate
                                          (0.0–1.0, approximated from beat strength).
        - ``key`` (str):                 Detected musical key (e.g. "A minor").

        All numeric values are 0.0 / empty list on failure.
    """
    empty: dict = {
        "bpm": 0.0,
        "beat_map": [],
        "energy_envelope": [],
        "duration": 0.0,
        "tempo_confidence": 0.0,
        "key": "unknown",
    }

    if not _require_librosa("analyse_audio"):
        return empty

    y, sr = _load_audio(audio_path, "analyse_audio")
    if y is None:
        return empty

    try:
        duration = float(librosa.get_duration(y=y, sr=sr))

        # ── BPM & beat map ────────────────────────────────────────────────────
        tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        bpm = float(np.atleast_1d(tempo_arr)[0])
        beat_times: list[float] = librosa.frames_to_time(
            beat_frames, sr=sr
        ).tolist()

        # ── Tempo confidence: ratio of strong beats to total beats ────────────
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        if beat_frames.size > 0 and onset_env.size > 0:
            beat_strengths = onset_env[
                np.clip(beat_frames, 0, len(onset_env) - 1)
            ]
            max_strength = onset_env.max()
            tempo_confidence = float(
                np.mean(beat_strengths) / (max_strength + 1e-9)
            )
        else:
            tempo_confidence = 0.0

        # ── Energy envelope at 10 Hz ─────────────────────────────────────────
        # hop_length chosen so that sr / hop_length ≈ 10 Hz
        hop_length = int(sr / 10)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_max = rms.max()
        if rms_max > 0:
            energy_envelope: list[float] = (rms / rms_max).tolist()
        else:
            energy_envelope = rms.tolist()

        # ── Musical key ───────────────────────────────────────────────────────
        key = _detect_key(y, sr)

        return {
            "bpm": bpm,
            "beat_map": beat_times,
            "energy_envelope": energy_envelope,
            "duration": duration,
            "tempo_confidence": tempo_confidence,
            "key": key,
        }

    except Exception as exc:  # noqa: BLE001
        logger.warning("analyse_audio: analysis failed for '%s': %s", audio_path, exc)
        return empty


def get_beat_map(audio_path: str) -> list[float]:
    """
    Return a list of beat timestamps (in seconds) for the given audio file.

    This is a convenience wrapper around :func:`analyse_audio`.
    Returns an empty list on failure.
    """
    return analyse_audio(audio_path)["beat_map"]


def get_bpm(audio_path: str) -> float:
    """
    Return the estimated BPM for the given audio file.

    Returns 0.0 on failure.
    """
    return analyse_audio(audio_path)["bpm"]


def find_energy_peaks(audio_path: str, n_peaks: int = 10) -> list[float]:
    """
    Return the timestamps (in seconds) of the *n_peaks* highest-energy moments.

    These timestamps are good candidates for edit cut points because they
    coincide with musical impact frames (drops, hits, accents).

    Args:
        audio_path: Path to the audio file.
        n_peaks:    Number of peak timestamps to return.

    Returns:
        Sorted list of timestamps in seconds.  Empty list on failure.
    """
    if not _require_librosa("find_energy_peaks"):
        return []

    y, sr = _load_audio(audio_path, "find_energy_peaks")
    if y is None:
        return []

    try:
        hop_length = int(sr / 10)  # 10 Hz sample rate for the envelope
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # We want the top-n indices — use argpartition for efficiency.
        n_peaks = min(n_peaks, len(rms))
        top_indices = np.argpartition(rms, -n_peaks)[-n_peaks:]
        top_indices_sorted = top_indices[np.argsort(rms[top_indices])[::-1]]

        # Convert frame indices → seconds
        times = librosa.frames_to_time(top_indices_sorted, sr=sr, hop_length=hop_length)
        return sorted(float(t) for t in times)

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "find_energy_peaks: failed for '%s': %s", audio_path, exc
        )
        return []


def align_cuts_to_beats(
    cut_times: list[float],
    beat_map: list[float],
) -> list[float]:
    """
    Snap each cut time to the nearest beat in *beat_map*.

    Useful for making rough cut points musically precise without re-running
    the full analysis.

    Args:
        cut_times: Proposed cut points in seconds (timeline or source time).
        beat_map:  Beat timestamps in seconds, as returned by :func:`get_beat_map`.

    Returns:
        A new list of beat-aligned cut times, in the same order as *cut_times*.
        If *beat_map* is empty, the original *cut_times* are returned unchanged.
    """
    if not beat_map:
        logger.warning(
            "align_cuts_to_beats: beat_map is empty — returning original cut_times."
        )
        return list(cut_times)

    if not cut_times:
        return []

    beats = sorted(beat_map)

    aligned: list[float] = []
    for cut in cut_times:
        # Binary-search for the closest beat
        lo, hi = 0, len(beats) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if beats[mid] < cut:
                lo = mid + 1
            else:
                hi = mid

        # lo is now the index of the first beat >= cut
        if lo == 0:
            nearest = beats[0]
        elif lo >= len(beats):
            nearest = beats[-1]
        else:
            before = beats[lo - 1]
            after = beats[lo]
            nearest = before if (cut - before) <= (after - cut) else after

        aligned.append(nearest)

    return aligned
