# Agentic Video Production Pipeline — Implementation Plan

## Stack Decisions
| Layer | Choice | Reason |
|---|---|---|
| LLM | Claude (claude-sonnet-4-6) | Tool use, structured JSON output, long context |
| Agent Framework | LangGraph | Graph state machine, native human-in-the-loop, streaming |
| Video Assembly | MoviePy + ffmpeg-python | MoviePy for sequencing logic, FFmpeg for final render/post |
| UI | Streamlit | Visual timeline, file upload, chat sidebar |
| CV / Reframing | YOLOv8 (Ultralytics) | Subject detection for 9:16 auto-crop |
| Beat Detection | Librosa | BPM, beat grid, energy envelope |
| Transcription | Whisper (openai-whisper) | Subtitle generation, lyric context |
| Stabilization | Gyroflow CLI | Drone / FPV footage stabilization |
| TTS | ElevenLabs API | Voiceover generation |
| Music Gen | MusicGen (HuggingFace) | AI background music from prompt |
| Stock Media | Pexels + Pixabay APIs | Free commercial-use clips & images |
| Settings | pydantic-settings + .env | All API keys and paths in one place |

---

## Project Root
```
C:\Users\Luke Fenech\Desktop\vidmachine\
```

---

## Full Directory Structure

```
vidmachine/
├── main.py                        # Entry point: `python main.py ui` or `python main.py run`
├── requirements.txt
├── .env.example
├── config.py                      # Pydantic-settings — all API keys, paths, model names
│
├── state/
│   ├── __init__.py
│   ├── project_state.py           # LangGraph TypedDict — the shared brain
│   ├── blueprint.py               # VideoBlueprint + nested Pydantic models
│   └── asset_manifest.py          # AssetInfo, AssetType, processing status tracking
│
├── agents/
│   ├── __init__.py
│   ├── base.py                    # BaseAgent ABC — run(), report(), log()
│   ├── director.py                # Supervisor: parses user intent, routes, manages memory
│   ├── ingest.py                  # Scan files, Gyroflow stabilize, Librosa beat map, Whisper transcribe
│   ├── sourcing.py                # Pexels/Pixabay download, ElevenLabs voiceover, MusicGen
│   ├── assembly.py                # MoviePy sequencing from blueprint → draft.mp4
│   ├── export.py                  # YOLOv8 reframe coords, FFmpeg final render, normalization
│   └── qa.py                      # Frame-level checks, audio sync audit, feedback to Director
│
├── tools/
│   ├── __init__.py
│   ├── gyroflow.py                # subprocess wrapper for Gyroflow CLI
│   ├── librosa_tools.py           # beat_map(), bpm(), energy_envelope()
│   ├── whisper_tools.py           # transcribe() → SRT + word-level timestamps
│   ├── stock_api.py               # PexelsClient, PixabayClient, download_clip()
│   ├── elevenlabs_tools.py        # generate_voiceover(text, voice_id) → mp3
│   ├── musicgen_tools.py          # generate_music(prompt, duration) → wav
│   ├── moviepy_tools.py           # sequence_clips(), add_text_overlay(), crossfade()
│   ├── ffmpeg_tools.py            # render_final(), normalize_audio(), burn_subtitles()
│   └── yolo_tools.py              # detect_subject(), compute_reframe_coords()
│
├── graph/
│   ├── __init__.py
│   ├── workflow.py                # LangGraph StateGraph — nodes, edges, compile()
│   └── router.py                  # route_from_director() conditional edge logic
│
├── ui/
│   ├── __init__.py
│   ├── app.py                     # Streamlit app root
│   └── components/
│       ├── __init__.py
│       ├── chat.py                # Chat sidebar — streams agent messages in real time
│       ├── timeline.py            # Visual timeline bar (clip blocks + beat markers)
│       ├── asset_browser.py       # Thumbnail grid of ingested / downloaded assets
│       └── progress.py            # Agent pipeline status cards with progress bars
│
├── storage/
│   ├── __init__.py
│   └── project_store.py           # save/load ProjectState ↔ JSON, project listing
│
└── projects/                      # Auto-created at runtime
    └── {project_id}/
        ├── blueprint.json         # The live editing blueprint
        ├── state.json             # Persisted LangGraph state snapshot
        ├── assets/
        │   ├── raw/               # Original uploaded footage
        │   ├── processed/         # Stabilized, trimmed clips
        │   └── downloaded/        # Stock media, generated audio
        └── output/
            ├── draft.mp4          # Intermediate MoviePy assembly
            ├── final_16x9.mp4     # FFmpeg polished render
            └── final_9x16.mp4     # Auto-reframed vertical export
```

---

## Core Data Model: VideoBlueprint (JSON spine shared by all agents)

```json
{
  "project_id": "uuid4",
  "title": "My FPV Reel",
  "created_at": "ISO8601",
  "output": {
    "resolution": [1920, 1080],
    "fps": 30,
    "format": "mp4",
    "duration_target": 60.0,
    "variants": ["16:9", "9:16"]
  },
  "tracks": {
    "video": [
      {
        "id": "clip_01",
        "source": "projects/abc/assets/processed/clip_01.mp4",
        "in_point": 2.0,
        "out_point": 8.5,
        "timeline_position": 0.0,
        "effects": ["stabilize"],
        "transition_in": {"type": "cut", "duration": 0},
        "transition_out": {"type": "dissolve", "duration": 0.3},
        "beat_aligned": true,
        "metadata": {"source_fps": 60, "gyroflow_applied": true}
      }
    ],
    "audio": [
      {
        "id": "audio_01",
        "source": "projects/abc/assets/downloaded/music.mp3",
        "in_point": 0.0,
        "out_point": 60.0,
        "volume": 0.85,
        "fade_in": 1.0,
        "fade_out": 2.0
      }
    ],
    "voiceover": [],
    "sfx": []
  },
  "beat_map": [0.0, 0.48, 0.96, 1.44],
  "subtitle_track": [],
  "color_grade": {"preset": "cinematic", "lut_path": null},
  "reframe_9x16": {
    "method": "yolo_tracking",
    "keyframes": [
      {"t": 0.0, "x": 480, "y": 0, "w": 540, "h": 1080}
    ]
  }
}
```

---

## LangGraph Shared State

```python
class ProjectState(TypedDict):
    project_id: str
    project_dir: str
    user_intent: str                   # Latest user message
    messages: List[BaseMessage]        # Full conversation history (LangGraph messages)
    blueprint: dict                    # VideoBlueprint as dict
    current_phase: str                 # init|ingest|sourcing|assembly|export|qa|done
    phase_results: dict[str, Any]      # Each agent writes its results here
    agent_notes: dict[str, str]        # Agent→Agent plain-English memos
    raw_assets: list[dict]
    processed_assets: list[dict]
    errors: list[str]
    warnings: list[str]
    output_paths: dict[str, str]       # "draft"→path, "final_16x9"→path, "final_9x16"→path
    awaiting_human: bool               # Triggers human-in-the-loop interrupt
```

---

## LangGraph Workflow Graph

```
[START]
   │
   ▼
[Director Agent] ◄──────────────────────────────────────┐
   │                                                     │
   ├─(phase=ingest)──────► [Ingest Agent]   ─────────────┤
   ├─(phase=sourcing)────► [Sourcing Agent] ─────────────┤
   ├─(phase=assembly)────► [Assembly Agent] ─────────────┤
   ├─(phase=export)──────► [Export Agent]   ─────────────┤
   ├─(phase=qa)──────────► [QA Agent]       ─────────────┤
   ├─(awaiting_human)────► [INTERRUPT → Streamlit chat]  ┤
   └─(phase=done)────────► [END]
```

All worker agents return state to Director. Director decides next phase or surfaces output to user.

---

## Streamlit UI Layout

```
┌─────────────────────────────────────────────────────────┐
│  AGENTIC VIDEO STUDIO                         [New Proj] │
├──────────────┬──────────────────────────────────────────┤
│  CHAT        │  PROJECT WORKSPACE                        │
│              │  ┌────────────────────────────────────┐  │
│  Director:   │  │  TIMELINE                          │  │
│  Analysing   │  │  [==clip1==][=clip2=][===clip3===] │  │
│  your FPV    │  │  ♪  ♪   ♪   ♪    ♪  ♪   ♪   ♪    │  │
│  footage...  │  └────────────────────────────────────┘  │
│              │  ┌──────────────┐  ┌──────────────────┐  │
│  You:        │  │ AGENT STATUS │  │  ASSET BROWSER   │  │
│  "Make cuts  │  │ ● Ingest  ✓  │  │  [▪][▪][▪][▪]   │  │
│   on the     │  │ ● Assembly ▶ │  │  [▪][▪][▪][▪]   │  │
│   beat"      │  │ ● Export  …  │  │                  │  │
│              │  └──────────────┘  └──────────────────┘  │
│  [Send ↵]    │                                           │
└──────────────┴──────────────────────────────────────────┘
```

---

## Inter-Agent Communication Protocol

Each agent:
1. Reads `ProjectState` (all fields visible to all agents)
2. Executes its tools
3. Writes structured results to `state["phase_results"][agent_name]`
4. Writes a plain-English memo to `state["agent_notes"][agent_name]` for other agents
5. Updates `state["current_phase"]` to signal completion to Director
6. Sets `state["awaiting_human"] = True` if user input is needed

Example ingest memo read by Assembly Agent:
```
"Found 3 FPV clips (4K@60fps). BPM=128, 64 beat hits across 30s.
Gyroflow stabilisation applied to all 3 clips.
Whisper found no lyrics. Beat map stored in blueprint.
Assembly: align clip cuts to beat_map timestamps."
```

---

## Build Phases

### Phase 1 — Foundation (1 subagent, sequential)
`requirements.txt`, `.env.example`, `config.py`, all `state/` models,
`agents/base.py`, `storage/project_store.py`, all `__init__.py` stubs

### Phase 2 — Agents + Tools (5 parallel subagents)
| Subagent | Owns |
|---|---|
| A: Director + Graph | `agents/director.py`, `graph/workflow.py`, `graph/router.py` |
| B: Ingest | `agents/ingest.py`, `tools/gyroflow.py`, `tools/librosa_tools.py`, `tools/whisper_tools.py` |
| C: Sourcing | `agents/sourcing.py`, `tools/stock_api.py`, `tools/elevenlabs_tools.py`, `tools/musicgen_tools.py` |
| D: Assembly | `agents/assembly.py`, `tools/moviepy_tools.py` |
| E: Export + QA | `agents/export.py`, `agents/qa.py`, `tools/ffmpeg_tools.py`, `tools/yolo_tools.py` |

### Phase 3 — UI + Entry Point (sequential)
`ui/components/` (4 files), `ui/app.py`, `main.py`

---

## Key Dependencies
```
anthropic>=0.40.0
langgraph>=0.2.0
langchain-anthropic>=0.3.0
langchain-core>=0.3.0
streamlit>=1.40.0
moviepy>=1.0.3
ffmpeg-python>=0.2.0
librosa>=0.10.0
openai-whisper>=20240930
ultralytics>=8.0.0
requests>=2.31.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
elevenlabs>=1.0.0
transformers>=4.40.0
torch>=2.0.0
rich>=13.0.0
pillow>=10.0.0
opencv-python>=4.8.0
```
