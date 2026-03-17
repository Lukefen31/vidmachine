"""
Asset tracking models — records every file the pipeline touches.
"""

from __future__ import annotations

import uuid
from typing import Literal, Optional
from pydantic import BaseModel, Field


AssetType = Literal[
    "video_raw",
    "video_processed",
    "video_stock",
    "audio_music",
    "audio_voiceover",
    "audio_sfx",
    "audio_generated",
    "image",
    "subtitle",
]

ProcessingStatus = Literal[
    "pending",
    "stabilizing",
    "analysing",
    "transcribing",
    "ready",
    "error",
]


class AssetInfo(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    asset_type: AssetType
    original_path: str
    working_path: str = ""          # Path after any processing (stabilise, transcode, etc.)
    filename: str = ""
    duration: Optional[float] = None   # Seconds (video/audio)
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    sample_rate: Optional[int] = None
    file_size_bytes: int = 0
    status: ProcessingStatus = "pending"
    error: str = ""
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    # Beat/music specific
    bpm: Optional[float] = None
    beat_map: list[float] = Field(default_factory=list)
    energy_envelope: list[float] = Field(default_factory=list)

    # Transcription
    transcript: str = ""
    srt_path: str = ""

    # Gyroflow
    gyroflow_applied: bool = False
    gyroflow_project_path: str = ""
