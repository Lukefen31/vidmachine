"""
Central configuration — all settings loaded from .env via pydantic-settings.
Import `settings` anywhere in the project.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"

    # ── Stock Media ───────────────────────────────────────────────────────────
    pexels_api_key: str = ""
    pixabay_api_key: str = ""

    # ── Audio Generation ──────────────────────────────────────────────────────
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "EXAVITQu4vr4xnSDxMaL"

    # ── Local Tool Paths ──────────────────────────────────────────────────────
    gyroflow_path: str = "gyroflow"
    ffmpeg_path: str = "ffmpeg"

    # ── Project Storage ───────────────────────────────────────────────────────
    projects_dir: Path = Path("projects")

    # ── MusicGen ──────────────────────────────────────────────────────────────
    musicgen_model: str = "facebook/musicgen-small"

    def project_path(self, project_id: str) -> Path:
        """Return the root directory for a given project."""
        return self.projects_dir / project_id

    def ensure_projects_dir(self) -> None:
        self.projects_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
