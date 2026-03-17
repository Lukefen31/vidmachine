"""
AI music generation using Meta's MusicGen via HuggingFace Transformers.

Falls back gracefully when torch / transformers are not installed, logging a
clear message and returning success=False so the pipeline can continue.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Availability check
# ─────────────────────────────────────────────────────────────────────────────

def is_musicgen_available() -> bool:
    """
    Return True if torch, transformers, and scipy are installed and the
    configured MusicGen model can be referenced.

    Does NOT attempt to load the model weights (that happens lazily).
    """
    try:
        import torch  # noqa: F401
        from transformers import AutoProcessor, MusicgenForConditionalGeneration  # noqa: F401
        import scipy.io.wavfile  # noqa: F401
        return True
    except ImportError as exc:
        logger.debug("MusicGen dependencies not available: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Intent → MusicGen prompt mapping
# ─────────────────────────────────────────────────────────────────────────────

# Maps keyword fragments found in user intent → descriptive MusicGen prompt.
_INTENT_PROMPT_MAP: list[tuple[list[str], str]] = [
    (
        ["fpv", "drone", "aerial"],
        "epic cinematic electronic music, driving beat, 128bpm, adrenaline, soaring",
    ),
    (
        ["travel", "vlog", "adventure", "explore"],
        "uplifting acoustic guitar and piano, positive vibes, travel adventure, 100bpm",
    ),
    (
        ["workout", "gym", "fitness", "training"],
        "energetic hip-hop beats, heavy bass, motivational, 140bpm",
    ),
    (
        ["wedding", "romantic", "love"],
        "soft romantic orchestral music, strings and piano, emotional, 70bpm",
    ),
    (
        ["nature", "landscape", "wildlife"],
        "ambient nature soundscape, peaceful orchestral swells, 60bpm",
    ),
    (
        ["tech", "startup", "corporate", "product"],
        "modern corporate background music, inspiring and clean, 100bpm",
    ),
    (
        ["gaming", "game", "esports"],
        "intense electronic gaming music, synth, fast tempo, 160bpm",
    ),
    (
        ["cinematic", "film", "movie", "trailer"],
        "epic cinematic trailer music, orchestral, dramatic build, 90bpm",
    ),
    (
        ["calm", "meditation", "relax", "sleep", "lofi"],
        "lo-fi chill hip-hop, mellow beats, relaxing, 75bpm",
    ),
    (
        ["comedy", "funny", "fun"],
        "playful upbeat ukulele and percussion, lighthearted and fun, 120bpm",
    ),
]

_DEFAULT_PROMPT = "ambient background music, neutral tone, gentle melody, 90bpm"


def get_music_prompt_from_intent(intent: str) -> str:
    """
    Convert a user intent string into a descriptive MusicGen prompt.

    Performs a case-insensitive keyword scan. The first matching rule wins.
    Falls back to a generic ambient prompt when no keywords match.

    Examples
    --------
    >>> get_music_prompt_from_intent("epic FPV compilation")
    'epic cinematic electronic music, driving beat, 128bpm, adrenaline, soaring'
    >>> get_music_prompt_from_intent("wedding highlight reel")
    'soft romantic orchestral music, strings and piano, emotional, 70bpm'
    """
    if not intent:
        return _DEFAULT_PROMPT

    lower_intent = intent.lower()
    for keywords, prompt in _INTENT_PROMPT_MAP:
        if any(kw in lower_intent for kw in keywords):
            logger.debug("Intent %r matched keyword group → %r", intent, prompt)
            return prompt

    logger.debug("No keyword match for intent %r — using default prompt", intent)
    return _DEFAULT_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# Music generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_music(
    prompt: str,
    duration: float = 30.0,
    output_path: str = "generated_music.wav",
    model_id: Optional[str] = None,
) -> dict:
    """
    Generate background music from a text *prompt* using MusicGen.

    Parameters
    ----------
    prompt:
        Descriptive text prompt for the music style.
    duration:
        Target duration in seconds (approximate — MusicGen works in tokens).
    output_path:
        Absolute path where the WAV file will be saved.
    model_id:
        HuggingFace model identifier.
        Defaults to ``config.settings.musicgen_model``
        (typically ``"facebook/musicgen-small"``).

    Returns
    -------
    dict
        ``{success, output_path, duration, prompt_used, error}``

    Notes
    -----
    - Automatically selects CUDA if available, otherwise CPU.
    - The model is loaded fresh for each call to keep memory usage predictable
      in a pipeline context. For repeated calls in the same process you can
      cache the model externally.
    """
    resolved_model = model_id or settings.musicgen_model

    # ── Dependency guard ──────────────────────────────────────────────────────
    if not is_musicgen_available():
        msg = (
            "MusicGen dependencies are not installed. "
            "Run: pip install torch transformers scipy "
            "to enable AI music generation."
        )
        logger.warning(msg)
        return {
            "success": False,
            "output_path": "",
            "duration": 0.0,
            "prompt_used": prompt,
            "error": msg,
        }

    import torch
    import numpy as np
    import scipy.io.wavfile as wav_writer
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "MusicGen: loading model %s on %s (duration=%.1fs)",
        resolved_model,
        device,
        duration,
    )

    # ── Ensure output dir ─────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        processor = AutoProcessor.from_pretrained(resolved_model)
        model = MusicgenForConditionalGeneration.from_pretrained(resolved_model)
        model = model.to(device)
        model.eval()

        # MusicGen generates ~50 tokens/second of audio (model-dependent).
        # We calculate the token count needed to approximate the target duration.
        # The config attribute is `audio_encoder.chunk_length_s` or we derive from
        # `max_new_tokens`.  The simplest cross-model approach is:
        #   tokens_per_second ≈ 50 for musicgen-small/medium, 51.2 for large
        tokens_per_second = getattr(
            model.config, "audio_encoder", None
        )
        # Fall back to the commonly documented 50 tokens/sec if not available
        if tokens_per_second is None or not hasattr(tokens_per_second, "frame_rate"):
            tokens_per_second = 50
        else:
            tokens_per_second = tokens_per_second.frame_rate

        max_new_tokens = int(duration * tokens_per_second)
        logger.info("MusicGen: max_new_tokens=%d for %.1fs", max_new_tokens, duration)

        # ── Encode prompt ─────────────────────────────────────────────────────
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ── Generate audio ────────────────────────────────────────────────────
        with torch.inference_mode():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                guidance_scale=3.0,
            )

        # audio_values shape: (batch, channels, samples)
        audio_np = audio_values[0].cpu().numpy()  # (channels, samples) or (samples,)

        # MusicGen outputs mono; squeeze to 1-D if needed
        if audio_np.ndim > 1:
            audio_np = audio_np[0]  # take first channel

        # Retrieve actual sampling rate from the model config
        sampling_rate: int = model.config.audio_encoder.sampling_rate

        # Convert float32 → int16 for standard WAV
        audio_int16 = (audio_np * 32767).astype(np.int16)

        wav_writer.write(output_path, sampling_rate, audio_int16)

        actual_duration = len(audio_np) / sampling_rate
        file_size = Path(output_path).stat().st_size

        logger.info(
            "MusicGen: wrote %s (%.1fs, %d bytes)",
            Path(output_path).name,
            actual_duration,
            file_size,
        )

        return {
            "success": True,
            "output_path": output_path,
            "duration": round(actual_duration, 2),
            "prompt_used": prompt,
            "error": "",
        }

    except Exception as exc:
        logger.error("MusicGen generation failed: %s", exc, exc_info=True)
        return {
            "success": False,
            "output_path": "",
            "duration": 0.0,
            "prompt_used": prompt,
            "error": str(exc),
        }
