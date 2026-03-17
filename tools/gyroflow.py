"""
Gyroflow CLI wrapper — drone/FPV footage stabilisation.

Gyroflow is an open-source video stabilisation tool that uses gyroscope data
(embedded in the video file or from an external .gyroflow project) to
remove unwanted camera shake with sub-frame precision.

Usage:
    from tools.gyroflow import stabilise_clip, check_gyroflow_available
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

# Timeout for a single gyroflow stabilisation run (seconds).
# Long clips can take a while; 30 minutes is a safe upper bound.
_GYROFLOW_TIMEOUT = 1800


def check_gyroflow_available() -> bool:
    """
    Return True if the Gyroflow CLI is reachable.

    Checks `settings.gyroflow_path` first (explicit config), then falls back
    to searching PATH with shutil.which so a plain ``"gyroflow"`` default
    also works when gyroflow is installed system-wide.
    """
    configured_path = settings.gyroflow_path

    # Explicit absolute/relative path in config — does the binary exist?
    if configured_path != "gyroflow":
        return os.path.isfile(configured_path) and os.access(configured_path, os.X_OK)

    # Default: look on PATH
    return shutil.which("gyroflow") is not None


def list_gyroflow_projects(directory: str) -> list[str]:
    """
    Find all .gyroflow project files inside *directory* (non-recursive).

    Args:
        directory: Folder to search.

    Returns:
        Sorted list of absolute path strings to .gyroflow files.
        Returns an empty list if the directory does not exist or is unreadable.
    """
    folder = Path(directory)
    if not folder.is_dir():
        logger.warning("list_gyroflow_projects: '%s' is not a directory.", directory)
        return []

    try:
        matches = sorted(str(p.resolve()) for p in folder.glob("*.gyroflow"))
        logger.debug("Found %d .gyroflow project(s) in '%s'.", len(matches), directory)
        return matches
    except OSError as exc:
        logger.warning("list_gyroflow_projects: could not read '%s': %s", directory, exc)
        return []


def stabilise_clip(
    input_path: str,
    output_path: str,
    gyroflow_project: str | None = None,
) -> dict:
    """
    Run the Gyroflow CLI to stabilise a video clip.

    Args:
        input_path:       Source video file (absolute or relative path).
        output_path:      Destination path for the stabilised output video.
        gyroflow_project: Optional path to a .gyroflow project file.
                          When None, Gyroflow attempts auto-detection from the
                          video's embedded gyro data or a sidecar file.

    Returns:
        A dict with the following keys:

        - ``success`` (bool):              Whether stabilisation completed without error.
        - ``output_path`` (str):           Absolute path of the stabilised file (empty on failure).
        - ``gyroflow_project_path`` (str): The .gyroflow project file used (empty if none / unknown).
        - ``error`` (str):                 Human-readable error description (empty on success).
    """
    _result_template: dict = {
        "success": False,
        "output_path": "",
        "gyroflow_project_path": gyroflow_project or "",
        "error": "",
    }

    # ── Pre-flight checks ──────────────────────────────────────────────────────

    if not check_gyroflow_available():
        msg = (
            "Gyroflow is not installed or not reachable at "
            f"'{settings.gyroflow_path}'. "
            "Install Gyroflow and set GYROFLOW_PATH in .env, or add it to PATH."
        )
        logger.warning(msg)
        return {**_result_template, "error": msg}

    input_file = Path(input_path)
    if not input_file.is_file():
        msg = f"Input file does not exist: '{input_path}'"
        logger.error(msg)
        return {**_result_template, "error": msg}

    if gyroflow_project is not None:
        gf_project_file = Path(gyroflow_project)
        if not gf_project_file.is_file():
            msg = f"Gyroflow project file does not exist: '{gyroflow_project}'"
            logger.error(msg)
            return {**_result_template, "error": msg}

    # Ensure the output directory exists.
    output_file = Path(output_path)
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        msg = f"Cannot create output directory '{output_file.parent}': {exc}"
        logger.error(msg)
        return {**_result_template, "error": msg}

    # ── Build CLI command ──────────────────────────────────────────────────────
    #
    # Gyroflow CLI reference (as of 1.5.x):
    #   gyroflow <input_video> [--gyroflow-project <file>] --output <output_video>
    #
    # The --export flag tells it to render and exit (non-interactive).

    cmd: list[str] = [settings.gyroflow_path]

    if gyroflow_project is not None:
        cmd += ["--gyroflow-project", str(Path(gyroflow_project).resolve())]

    cmd += [
        str(input_file.resolve()),
        "--output",
        str(output_file.resolve()),
        "--export",          # render and exit non-interactively
    ]

    logger.info("Running Gyroflow: %s", " ".join(cmd))

    # ── Execute ────────────────────────────────────────────────────────────────

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_GYROFLOW_TIMEOUT,
            text=True,
        )
    except FileNotFoundError:
        msg = (
            f"Gyroflow executable not found at '{settings.gyroflow_path}'. "
            "Ensure it is installed and the path is correct."
        )
        logger.error(msg)
        return {**_result_template, "error": msg}
    except subprocess.TimeoutExpired:
        msg = (
            f"Gyroflow timed out after {_GYROFLOW_TIMEOUT}s "
            f"processing '{input_path}'."
        )
        logger.error(msg)
        return {**_result_template, "error": msg}
    except OSError as exc:
        msg = f"OS error launching Gyroflow: {exc}"
        logger.error(msg)
        return {**_result_template, "error": msg}

    # ── Evaluate result ────────────────────────────────────────────────────────

    if proc.returncode != 0:
        stderr_snippet = (proc.stderr or "").strip()[-500:]  # keep it readable
        msg = (
            f"Gyroflow exited with code {proc.returncode}. "
            f"stderr: {stderr_snippet}"
        )
        logger.error(msg)
        return {**_result_template, "error": msg}

    if not output_file.is_file():
        msg = (
            f"Gyroflow reported success (exit 0) but output file was not "
            f"created at '{output_path}'."
        )
        logger.error(msg)
        return {**_result_template, "error": msg}

    logger.info(
        "Gyroflow stabilisation complete: '%s' → '%s'",
        input_path,
        output_path,
    )

    return {
        "success": True,
        "output_path": str(output_file.resolve()),
        "gyroflow_project_path": gyroflow_project or "",
        "error": "",
    }
