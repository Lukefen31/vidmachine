"""
ElevenLabs text-to-speech voiceover generation tools.

Falls back to a mock response when ELEVENLABS_API_KEY is not set,
so the pipeline can be developed and tested without a live key.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def estimate_duration(text: str, words_per_minute: int = 150) -> float:
    """
    Estimate the spoken duration of *text* in seconds.

    Uses a simple word-count heuristic at *words_per_minute* WPM
    (default 150 WPM ≈ conversational speech).
    """
    if not text or not text.strip():
        return 0.0
    word_count = len(text.split())
    return round((word_count / words_per_minute) * 60.0, 2)


def _get_client():
    """
    Return an initialised ElevenLabs client, or None if the SDK is unavailable
    or the API key is missing.
    """
    if not settings.elevenlabs_api_key:
        return None
    try:
        from elevenlabs.client import ElevenLabs  # type: ignore
        return ElevenLabs(api_key=settings.elevenlabs_api_key)
    except ImportError:
        logger.warning(
            "elevenlabs SDK not installed. "
            "Run `pip install elevenlabs` to enable TTS."
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_voiceover(
    text: str,
    output_path: str,
    voice_id: Optional[str] = None,
    model_id: str = "eleven_multilingual_v2",
    stability: float = 0.5,
    similarity_boost: float = 0.75,
) -> dict:
    """
    Generate a voiceover MP3 using the ElevenLabs API.

    Parameters
    ----------
    text:
        The script to synthesise.
    output_path:
        Absolute path where the MP3 will be written.
    voice_id:
        ElevenLabs voice ID. Defaults to ``config.settings.elevenlabs_voice_id``.
    model_id:
        ElevenLabs model identifier (default ``eleven_multilingual_v2``).
    stability / similarity_boost:
        Voice settings (0.0–1.0).

    Returns
    -------
    dict
        ``{success: bool, output_path: str, duration_estimate: float, error: str}``
    """
    resolved_voice_id = voice_id or settings.elevenlabs_voice_id
    duration_estimate = estimate_duration(text)

    # ── Guard: no API key / SDK not available ─────────────────────────────────
    client = _get_client()
    if client is None:
        if not settings.elevenlabs_api_key:
            logger.warning(
                "ELEVENLABS_API_KEY is not set. Returning mock voiceover response. "
                "Set the key in your .env to enable real TTS."
            )
        return {
            "success": False,
            "output_path": "",
            "duration_estimate": duration_estimate,
            "error": "ElevenLabs API key not configured or SDK not installed.",
        }

    # ── Validate text ─────────────────────────────────────────────────────────
    if not text or not text.strip():
        return {
            "success": False,
            "output_path": "",
            "duration_estimate": 0.0,
            "error": "No text provided for voiceover generation.",
        }

    # ── Ensure output directory exists ────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Generate via SDK ──────────────────────────────────────────────────────
    try:
        from elevenlabs import VoiceSettings  # type: ignore

        logger.info(
            "Generating voiceover: voice=%s model=%s text_len=%d chars",
            resolved_voice_id,
            model_id,
            len(text),
        )

        audio_iterator = client.text_to_speech.convert(
            text=text,
            voice_id=resolved_voice_id,
            model_id=model_id,
            voice_settings=VoiceSettings(
                stability=stability,
                similarity_boost=similarity_boost,
            ),
        )

        # The SDK returns an iterator of bytes chunks; write them all out.
        with open(output_path, "wb") as fh:
            for chunk in audio_iterator:
                if chunk:
                    fh.write(chunk)

        file_size = Path(output_path).stat().st_size
        logger.info(
            "Voiceover written to %s (%d bytes, ~%.1fs)",
            output_path,
            file_size,
            duration_estimate,
        )

        return {
            "success": True,
            "output_path": output_path,
            "duration_estimate": duration_estimate,
            "error": "",
        }

    except Exception as exc:
        logger.error("ElevenLabs generation failed: %s", exc)
        return {
            "success": False,
            "output_path": "",
            "duration_estimate": duration_estimate,
            "error": str(exc),
        }


def list_voices() -> list[dict]:
    """
    Return available ElevenLabs voices as a list of dicts.

    Each dict has keys: ``voice_id``, ``name``, ``category``.
    Returns an empty list if the API key is missing or the SDK is unavailable.
    """
    client = _get_client()
    if client is None:
        logger.warning("Cannot list voices — ElevenLabs client unavailable.")
        return []

    try:
        response = client.voices.get_all()
        voices = []
        for v in response.voices:
            voices.append({
                "voice_id": v.voice_id,
                "name": v.name,
                "category": getattr(v, "category", "unknown"),
            })
        logger.info("Retrieved %d ElevenLabs voices", len(voices))
        return voices
    except Exception as exc:
        logger.error("Failed to list ElevenLabs voices: %s", exc)
        return []
