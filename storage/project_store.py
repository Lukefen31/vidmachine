"""
ProjectStore — save and load ProjectState to/from disk as JSON.

Each project lives at:
  projects/{project_id}/
    ├── blueprint.json    ← VideoBlueprint (written separately for easy editing)
    └── state.json        ← Full ProjectState snapshot (messages excluded for size)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from config import settings
from state.blueprint import VideoBlueprint


class ProjectStore:

    def __init__(self, projects_dir: Optional[Path] = None):
        self.projects_dir = projects_dir or settings.projects_dir
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    # ── Project directory helpers ─────────────────────────────────────────────

    def project_dir(self, project_id: str) -> Path:
        return self.projects_dir / project_id

    def ensure_project_dirs(self, project_id: str) -> Path:
        root = self.project_dir(project_id)
        for sub in ["assets/raw", "assets/processed", "assets/downloaded", "output"]:
            (root / sub).mkdir(parents=True, exist_ok=True)
        return root

    # ── State persistence ─────────────────────────────────────────────────────

    def save_state(self, state: dict) -> None:
        """Persist the ProjectState dict (messages stripped for size)."""
        project_id = state["project_id"]
        root = self.ensure_project_dirs(project_id)

        # Strip LangGraph BaseMessage objects — not JSON serialisable
        slim = {k: v for k, v in state.items() if k != "messages"}

        state_path = root / "state.json"
        state_path.write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")

        # Always keep blueprint in sync as its own file
        if state.get("blueprint"):
            blueprint_path = root / "blueprint.json"
            blueprint_path.write_text(
                json.dumps(state["blueprint"], indent=2, default=str), encoding="utf-8"
            )

    def load_state(self, project_id: str) -> Optional[dict]:
        """Load a previously saved state dict. Returns None if not found."""
        state_path = self.project_dir(project_id) / "state.json"
        if not state_path.exists():
            return None
        return json.loads(state_path.read_text(encoding="utf-8"))

    # ── Blueprint helpers ─────────────────────────────────────────────────────

    def save_blueprint(self, project_id: str, blueprint: VideoBlueprint) -> None:
        root = self.ensure_project_dirs(project_id)
        (root / "blueprint.json").write_text(
            blueprint.model_dump_json(indent=2), encoding="utf-8"
        )

    def load_blueprint(self, project_id: str) -> Optional[VideoBlueprint]:
        path = self.project_dir(project_id) / "blueprint.json"
        if not path.exists():
            return None
        return VideoBlueprint.model_validate_json(path.read_text(encoding="utf-8"))

    # ── Project listing ───────────────────────────────────────────────────────

    def list_projects(self) -> list[dict]:
        """Return a list of {project_id, title, updated_at} for the UI."""
        projects = []
        for entry in sorted(self.projects_dir.iterdir(), key=os.path.getmtime, reverse=True):
            if not entry.is_dir():
                continue
            blueprint_path = entry / "blueprint.json"
            if blueprint_path.exists():
                try:
                    data = json.loads(blueprint_path.read_text(encoding="utf-8"))
                    projects.append({
                        "project_id": data.get("project_id", entry.name),
                        "title": data.get("title", "Untitled"),
                        "updated_at": data.get("updated_at", ""),
                    })
                except Exception:
                    projects.append({"project_id": entry.name, "title": entry.name, "updated_at": ""})
        return projects

    def delete_project(self, project_id: str) -> bool:
        import shutil
        root = self.project_dir(project_id)
        if root.exists():
            shutil.rmtree(root)
            return True
        return False


# Module-level singleton
store = ProjectStore()
