"""
Asset Browser — thumbnail grid of all project assets.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st


def render_asset_browser(state: dict | None, project_dir: str = "") -> None:
    """
    Render a grid of thumbnails for all raw and processed assets.
    """
    st.subheader("Asset Browser")

    if not state:
        st.caption("No project loaded.")
        return

    raw_assets: list[dict] = state.get("raw_assets", [])
    processed_assets: list[dict] = state.get("processed_assets", [])
    all_assets = processed_assets or raw_assets

    if not all_assets:
        st.caption("No assets ingested yet. Drop your footage into the project folder.")
        _show_drop_hint(project_dir)
        return

    # Filter controls
    col1, col2 = st.columns([2, 1])
    with col1:
        search = st.text_input("Filter assets", placeholder="Search by filename...", label_visibility="collapsed")
    with col2:
        filter_type = st.selectbox(
            "Type",
            ["All", "Video", "Audio"],
            label_visibility="collapsed",
        )

    # Filter assets
    filtered = all_assets
    if search:
        filtered = [a for a in filtered if search.lower() in a.get("filename", "").lower()]
    if filter_type == "Video":
        filtered = [a for a in filtered if "video" in a.get("asset_type", "")]
    elif filter_type == "Audio":
        filtered = [a for a in filtered if "audio" in a.get("asset_type", "")]

    if not filtered:
        st.caption("No assets match filter.")
        return

    # Grid display
    cols_per_row = 3
    for row_start in range(0, len(filtered), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, asset in enumerate(filtered[row_start: row_start + cols_per_row]):
            with cols[col_idx]:
                _render_asset_card(asset)


def _render_asset_card(asset: dict) -> None:
    asset_type = asset.get("asset_type", "unknown")
    filename = asset.get("filename") or Path(asset.get("original_path", "unknown")).name
    status = asset.get("status", "pending")
    duration = asset.get("duration")
    bpm = asset.get("bpm")
    gyroflow = asset.get("gyroflow_applied", False)

    status_color = {
        "ready": "🟢",
        "pending": "🟡",
        "error": "🔴",
        "stabilizing": "🔵",
        "analysing": "🔵",
        "transcribing": "🔵",
    }.get(status, "⚪")

    is_video = "video" in asset_type
    icon = "🎬" if is_video else "🎵"

    with st.container():
        st.markdown(
            f"""
            <div style="
                border: 1px solid #333;
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 8px;
                background: #1e1e1e;
            ">
                <div style="font-size: 2rem; text-align: center;">{icon}</div>
                <div style="font-size: 0.75rem; color: #ccc; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{filename}">
                    {filename}
                </div>
                <div style="font-size: 0.7rem; color: #888; margin-top: 4px;">
                    {status_color} {status}
                    {f" · {duration:.1f}s" if duration else ""}
                    {f" · {bpm:.0f} BPM" if bpm else ""}
                    {"· 🌀 Stabilised" if gyroflow else ""}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _show_drop_hint(project_dir: str) -> None:
    if project_dir:
        raw_dir = str(Path(project_dir) / "assets" / "raw")
        st.markdown(
            f"""
            **Drop your files here:**
            ```
            {raw_dir}
            ```
            Supported formats: `.mp4 .mov .avi .mkv .mts .mp3 .wav .flac .aac`
            """,
        )
