"""
VideoBlueprint — the central JSON spine that all agents read and write.
Every clip, audio track, beat marker, subtitle, and effect lives here.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Transitions
# ─────────────────────────────────────────────────────────────────────────────

class Transition(BaseModel):
    type: Literal["cut", "dissolve", "fade", "wipe", "flash"] = "cut"
    duration: float = 0.0  # seconds


# ─────────────────────────────────────────────────────────────────────────────
# Video Clip
# ─────────────────────────────────────────────────────────────────────────────

class VideoClip(BaseModel):
    id: str = Field(default_factory=lambda: f"clip_{uuid.uuid4().hex[:8]}")
    source: str                          # Absolute or project-relative path
    in_point: float = 0.0               # Seconds into source file
    out_point: float = 0.0              # Seconds into source file (exclusive)
    timeline_position: float = 0.0      # Position on output timeline (seconds)
    effects: list[str] = Field(default_factory=list)  # e.g. ["stabilize", "grade"]
    transition_in: Transition = Field(default_factory=Transition)
    transition_out: Transition = Field(default_factory=Transition)
    beat_aligned: bool = False
    volume: float = 0.0                  # 0.0 = muted (natural sound), -1 = strip audio
    metadata: dict = Field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.out_point - self.in_point)


# ─────────────────────────────────────────────────────────────────────────────
# Audio Track
# ─────────────────────────────────────────────────────────────────────────────

class AudioTrack(BaseModel):
    id: str = Field(default_factory=lambda: f"audio_{uuid.uuid4().hex[:8]}")
    source: str
    in_point: float = 0.0
    out_point: float = 0.0
    timeline_position: float = 0.0
    volume: float = 1.0
    fade_in: float = 0.5
    fade_out: float = 1.0
    track_type: Literal["music", "voiceover", "sfx"] = "music"
    metadata: dict = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Subtitle Entry
# ─────────────────────────────────────────────────────────────────────────────

class SubtitleEntry(BaseModel):
    id: str = Field(default_factory=lambda: f"sub_{uuid.uuid4().hex[:8]}")
    start: float       # Timeline seconds
    end: float
    text: str
    style: dict = Field(default_factory=dict)  # font, size, color, position


# ─────────────────────────────────────────────────────────────────────────────
# Text / Title Overlay
# ─────────────────────────────────────────────────────────────────────────────

class TextOverlay(BaseModel):
    id: str = Field(default_factory=lambda: f"text_{uuid.uuid4().hex[:8]}")
    text: str
    start: float
    end: float
    position: tuple[float, float] = (0.5, 0.9)  # Normalized (x, y)
    font_size: int = 48
    color: str = "white"
    font: str = "Arial"


# ─────────────────────────────────────────────────────────────────────────────
# 9:16 Reframe Keyframe
# ─────────────────────────────────────────────────────────────────────────────

class ReframeKeyframe(BaseModel):
    t: float           # Timeline seconds
    x: int             # Crop left edge (pixels in source frame)
    y: int             # Crop top edge
    w: int             # Crop width
    h: int             # Crop height


class Reframe9x16(BaseModel):
    method: Literal["center_crop", "yolo_tracking", "manual"] = "yolo_tracking"
    source_width: int = 1920
    source_height: int = 1080
    target_width: int = 608   # 1080 * (9/16) rounded
    target_height: int = 1080
    keyframes: list[ReframeKeyframe] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Output Config
# ─────────────────────────────────────────────────────────────────────────────

class OutputConfig(BaseModel):
    resolution: tuple[int, int] = (1920, 1080)
    fps: int = 30
    format: str = "mp4"
    codec: str = "libx264"
    audio_codec: str = "aac"
    duration_target: Optional[float] = None  # None = match assembled length
    variants: list[Literal["16:9", "9:16"]] = Field(default_factory=lambda: ["16:9"])
    bitrate: str = "8000k"
    audio_bitrate: str = "192k"


# ─────────────────────────────────────────────────────────────────────────────
# Color Grade
# ─────────────────────────────────────────────────────────────────────────────

class ColorGrade(BaseModel):
    preset: Literal["none", "cinematic", "warm", "cold", "high_contrast", "log"] = "none"
    lut_path: Optional[str] = None
    brightness: float = 0.0   # -1.0 to 1.0
    contrast: float = 0.0
    saturation: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Tracks container
# ─────────────────────────────────────────────────────────────────────────────

class Tracks(BaseModel):
    video: list[VideoClip] = Field(default_factory=list)
    audio: list[AudioTrack] = Field(default_factory=list)
    voiceover: list[AudioTrack] = Field(default_factory=list)
    sfx: list[AudioTrack] = Field(default_factory=list)
    text_overlays: list[TextOverlay] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# VideoBlueprint — root model
# ─────────────────────────────────────────────────────────────────────────────

class VideoBlueprint(BaseModel):
    project_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    title: str = "Untitled Project"
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    output: OutputConfig = Field(default_factory=OutputConfig)
    tracks: Tracks = Field(default_factory=Tracks)
    color_grade: ColorGrade = Field(default_factory=ColorGrade)

    beat_map: list[float] = Field(default_factory=list)   # Beat timestamps (seconds)
    bpm: Optional[float] = None
    subtitle_track: list[SubtitleEntry] = Field(default_factory=list)
    reframe_9x16: Reframe9x16 = Field(default_factory=Reframe9x16)

    # Freeform notes written by agents for traceability
    agent_annotations: dict[str, str] = Field(default_factory=dict)

    def total_video_duration(self) -> float:
        """Calculated duration of assembled video track."""
        if not self.tracks.video:
            return 0.0
        last = max(
            c.timeline_position + c.duration for c in self.tracks.video
        )
        return last

    def touch(self) -> None:
        self.updated_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "VideoBlueprint":
        return cls.model_validate(data)
