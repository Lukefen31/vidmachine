"""
Timeline visualiser — shows video clips as blocks on a horizontal bar,
with beat markers beneath.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


def render_timeline(blueprint: dict | None) -> None:
    """
    Render a visual timeline of the current blueprint.
    Uses HTML/CSS blocks for clip visualisation.
    """
    st.subheader("Timeline")

    if not blueprint:
        st.caption("No blueprint loaded.")
        return

    video_clips: list[dict] = blueprint.get("tracks", {}).get("video", [])
    audio_tracks: list[dict] = blueprint.get("tracks", {}).get("audio", [])
    beat_map: list[float] = blueprint.get("beat_map", [])
    total_duration = _calc_total_duration(video_clips, audio_tracks)

    if total_duration == 0:
        st.caption("Blueprint is empty — no clips added yet.")
        return

    # Build timeline HTML
    html = _build_timeline_html(
        video_clips=video_clips,
        audio_tracks=audio_tracks,
        beat_map=beat_map,
        total_duration=total_duration,
    )
    components.html(html, height=180, scrolling=False)

    # Stats row
    cols = st.columns(4)
    cols[0].metric("Total Duration", f"{total_duration:.1f}s")
    cols[1].metric("Video Clips", len(video_clips))
    cols[2].metric("Beat Markers", len(beat_map))
    bpm = blueprint.get("bpm")
    cols[3].metric("BPM", f"{bpm:.0f}" if bpm else "—")


def _calc_total_duration(video_clips: list[dict], audio_tracks: list[dict]) -> float:
    durations = []
    for c in video_clips:
        end = c.get("timeline_position", 0) + (c.get("out_point", 0) - c.get("in_point", 0))
        durations.append(end)
    for a in audio_tracks:
        end = a.get("timeline_position", 0) + (a.get("out_point", 0) - a.get("in_point", 0))
        durations.append(end)
    return max(durations) if durations else 0.0


def _build_timeline_html(
    video_clips: list[dict],
    audio_tracks: list[dict],
    beat_map: list[float],
    total_duration: float,
) -> str:
    width_px = 800
    clip_row_h = 40
    audio_row_h = 24
    beat_row_h = 16
    padding = 8

    clip_colors = [
        "#4a9eff", "#ff6b6b", "#48cfad", "#ffce54",
        "#ac92ec", "#fc6e51", "#a0d468", "#4fc1e9",
    ]

    def pct(t: float) -> float:
        return (t / total_duration) * 100 if total_duration > 0 else 0

    # Build clip blocks
    clip_blocks = ""
    for i, clip in enumerate(video_clips):
        pos = clip.get("timeline_position", 0)
        dur = clip.get("out_point", 0) - clip.get("in_point", 0)
        left = pct(pos)
        w = pct(dur)
        color = clip_colors[i % len(clip_colors)]
        name = clip.get("id", f"clip_{i}")
        beat_tag = "♪" if clip.get("beat_aligned") else ""
        clip_blocks += f"""
        <div title="{name} ({dur:.1f}s)" style="
            position: absolute;
            left: {left:.2f}%;
            width: {max(w, 0.5):.2f}%;
            height: {clip_row_h}px;
            background: {color};
            border-radius: 4px;
            border: 1px solid rgba(255,255,255,0.2);
            overflow: hidden;
            display: flex;
            align-items: center;
            padding-left: 4px;
            box-sizing: border-box;
        ">
            <span style="font-size: 10px; color: white; white-space: nowrap; overflow: hidden;">{beat_tag} {name}</span>
        </div>
        """

    # Build audio blocks
    audio_blocks = ""
    for i, track in enumerate(audio_tracks):
        pos = track.get("timeline_position", 0)
        dur = track.get("out_point", 0) - track.get("in_point", 0)
        left = pct(pos)
        w = pct(dur)
        audio_blocks += f"""
        <div title="{track.get('id','audio')} ({dur:.1f}s)" style="
            position: absolute;
            left: {left:.2f}%;
            width: {max(w, 0.5):.2f}%;
            height: {audio_row_h}px;
            background: #2ecc71;
            border-radius: 3px;
            opacity: 0.8;
        "></div>
        """

    # Build beat markers
    beat_markers = ""
    for bt in beat_map[:200]:  # cap at 200 for performance
        left = pct(bt)
        beat_markers += f"""
        <div style="
            position: absolute;
            left: {left:.2f}%;
            width: 1px;
            height: {beat_row_h}px;
            background: rgba(255, 220, 0, 0.7);
        "></div>
        """

    # Time ruler ticks (every 5s)
    ruler = ""
    tick_interval = 5
    for t in range(0, int(total_duration) + 1, tick_interval):
        left = pct(t)
        ruler += f"""
        <div style="position: absolute; left: {left:.2f}%; bottom: 0; font-size: 9px; color: #888; transform: translateX(-50%);">
            {t}s
        </div>
        """

    total_height = clip_row_h + audio_row_h + beat_row_h + 24 + padding * 3

    html = f"""
    <style>
        body {{ margin: 0; background: #111; font-family: sans-serif; }}
        .track-label {{ font-size: 10px; color: #666; margin-bottom: 2px; }}
    </style>
    <div style="background: #111; padding: {padding}px; border-radius: 8px; width: 100%; box-sizing: border-box;">
        <div class="track-label">VIDEO</div>
        <div style="position: relative; width: 100%; height: {clip_row_h}px; background: #1a1a1a; border-radius: 4px; margin-bottom: 4px; overflow: hidden;">
            {clip_blocks}
        </div>
        <div class="track-label">AUDIO</div>
        <div style="position: relative; width: 100%; height: {audio_row_h}px; background: #1a1a1a; border-radius: 3px; margin-bottom: 4px; overflow: hidden;">
            {audio_blocks}
        </div>
        <div class="track-label">BEATS</div>
        <div style="position: relative; width: 100%; height: {beat_row_h}px; background: #111; overflow: hidden;">
            {beat_markers}
        </div>
        <div style="position: relative; width: 100%; height: 20px;">
            {ruler}
        </div>
    </div>
    """
    return html
