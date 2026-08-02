# Codebase Documentation — Episodic Memory Pipeline

## Overview

This project is a **benchmarking pipeline** to assess **Vision Language Models' (VLMs) episodic memory capabilities** in long-form **egocentric dashcam video**. The input videos are dashcam recordings from vehicles driving through forests, rural roads, and tunnels, ranging from 6 minutes to 1 hour in duration. These videos have very low information density (few events), which intentionally stresses the model's ability to recall details over long time spans.

### Benchmark Categories (as defined in `AI.md`)

| # | Category | Description |
|---|----------|-------------|
| 1 | **Sparse Event Localisation** | Detecting rare, outlier events without hallucination. Includes temporal perception, existence verification, noteworthy object/moment queries, and deceptive questioning. |
| 2 | **Attribute Perception & Temporal Order** | Testing if the model truly remembers event order or just guesses from training distribution. |
| 3 | **Spatial Reasoning** | Physically grounded reasoning (e.g., distance estimation) in environments with scarce reference points. |
| 4 | **Counting** | Episodic-memory-based counting across long time spans where objects cannot all be seen in a single frame. |

---

## Project Status Summary

### What Is Currently Implemented

1. **Full-Video Chunked Description Pipeline** (`main.py`) — Processes an entire video in chunks of frames, generating scene descriptions via a VLM (Qwen2-VL) and detecting changes between chunks via a text LLM comparator (DeepSeek-V4). This serves as the general foundation for the broader benchmark (all four categories).

2. **Video Parser** (`sparse_event_pipeline/video_parser.py`) — Splits a video into overlapping time sections, extracts sampled JPEG frames into per-section subfolders, and writes a manifest JSON. Uses Decord for video reading and PIL for frame resizing/saving. No ffmpeg dependency.

3. **AI Event Parser** (`sparse_event_pipeline/AIParser.py`) — Sends batches of frame images to a VLM via the opencode.ai API gateway (Anthropic-compatible endpoint) and parses the model's JSON response for rare/noteworthy events. Uses asyncio with a semaphore for concurrent API calls.

4. **HPC Job Script** (`run_vllm.pbs`) — PBS batch script for running the vLLM pipeline on NCI's Gadi cluster with A100 GPUs.

### What Is Not Yet Implemented

- The dedicated benchmarks for categories 2–4 (attribute perception, spatial reasoning, counting) — only the foundation pipeline exists.
- The `run_vllm.pbs` script references a non-existent file `test_vlm.py`.
- The `AI.md` design brief mentions plans for a **User Interface**, a **Hybrid Search Module** (historical data fetching), and a **Reporting frontend** — none of these exist yet.
- No test suite, CI/CD, Dockerfile, or Makefile exists.

---

## Full Directory Tree

```
episodic-memory-pipeline/
├── AI.md
├── .gitignore
├── main.py
├── README.md
├── requirements.txt
├── run_vllm.pbs
├── CODEBASE_DOCUMENTATION.md
├── sparse_event_pipeline/
│   ├── AIParser.py
│   ├── run.py
│   └── video_parser.py
├── vids/
│   └── 118014-714270866_medium.mp4
└── .venv/                         (virtual environment, git-ignored)
```

---

## File-by-File Documentation

---

### `AI.md`

**Purpose:** Project design brief / AI context document. Serves as the specification for the entire project.

**What it defines:**
- Project identity and goal: benchmarking VLMs on episodic memory using long-form egocentric dashcam video.
- The four benchmark categories (listed above).
- Input video characteristics: dashcam footage from forest, rural road, and tunnel driving. 6 minutes to 1 hour duration, minimal events, low information density.
- High-level pipeline structure: a master pipeline composed of sub-pipelines for each benchmark category, with a user interface for executing specific benchmarks and a reporting frontend.
- A "Hybrid Search" module for fetching historical data (anecdotes, images, code repositories, weather reports) to enrich reasoning.

**Status:** Reference document. Not all planned components have been implemented.

---

### `.gitignore`

**Purpose:** Version control exclusion rules.

**Contents:**
- `/vids/` — Excludes all video files from git (videos are large binary files).
- `/.venv/` — Excludes the Python virtual environment.

---

### `main.py`

**Purpose:** Core episodic memory benchmarking pipeline. Processes a video end-to-end by chunking it into segments, describing each segment with a VLM, and comparing consecutive descriptions with a text LLM to detect changes.

**Key Dependencies:** `vllm` (LLM, SamplingParams), `decord` (VideoReader), `PIL.Image`, `json`, `os`.

**Functions:**

| Function | Purpose |
|----------|---------|
| `format_timestamp(seconds)` | Converts a float seconds value to `MM:SS` string format |
| `sample_frames_from_indices(vr, start_idx, end_idx, num_samples=16)` | Uniformly samples a fixed number of frames (16) from a Decord VideoReader between two frame indices. Returns a list of PIL Images. |
| `run_video_pipeline(video_path, full_history_json_path, change_analysis_json_path, vlm_model_name, text_llm_model_name, frames_per_chunk=300)` | **The main entry point.** See detailed flow below. |

**Pipeline Flow (`run_video_pipeline`):**

1. Opens the video via Decord, computes total frames and FPS.
2. Initializes two separate vLLM models:
   - **VLM** (default: `Qwen/Qwen2-VL-7B-Instruct`) — for describing video chunks.
   - **Text LLM** (default: `deepseek-ai/DeepSeek-V4-Flash`) — for comparing consecutive scene descriptions.
3. Iterates through the video in chunks of `frames_per_chunk` (default: 300 frames).
4. For each chunk:
   - Samples 16 frames uniformly from the chunk.
   - Sends frames to the VLM with prompt: *"Describe what is taking place in this scene in detail."*
   - If a previous description exists, asks the text LLM to compare the two scenes and output JSON: `{something_new, what_changed, question, answer}`.
   - If a change is detected (`something_new: true`), appends a change record (including the generated QA pair) to `change_events_only`.
   - Logs every chunk's description and change status to `full_pipeline_history`.
5. Stream-saves both JSON files (`full_pipeline_history.json` and `change_analysis.json`) to disk after **every chunk** (crash-resilient).
6. Returns both data buffers after processing the entire video.

**Bottom-of-file invocation:** Hardcoded test call with `"test.mp4"`, outputting to `./`. Uses `"Qwen/Qwen2-VL-7B-Instruct"` and `"deepseek-ai/DeepSeek-V4-Flash"`.

**Status:** Functionally complete but hardcoded. The test video `"test.mp4"` likely does not exist (actual video is in `vids/`). The PBS script (`run_vllm.pbs`) references `test_vlm.py` which does not exist — this script may have evolved from or replaced that earlier file.

---

### `README.md`

**Purpose:** Minimal project readme.

**Contents:** A single heading: `#Episodic-Memory`

**Status:** Placeholder. No installation instructions, usage examples, or project description.

---

### `requirements.txt`

**Purpose:** Root-level Python dependencies.

**Contents:**
- `vlm` — Likely meant to be `vllm` (VLM inference engine).
- `decord` — Video reading and decoding library used by `main.py`.

**Note:** `vlm` may be a typo; the actual package name is `vllm`.

---

### `run_vllm.pbs`

**Purpose:** PBS (Portable Batch System) job script for running the vLLM pipeline on NCI's Gadi HPC cluster.

**Job Configuration:**
- **Job name:** `qwen_vllm_large`
- **Queue:** `dgxa100` (NVIDIA A100 GPU nodes)
- **Resources:** 16 CPUs, 1 GPU, 64 GB RAM, 10-hour walltime, 50 GB local SSD (jobfs)
- **Modules loaded:** `python3/3.11.7`, `cuda/12.2.2`
- **Environment variables set:**
  - `CPATH`, `CPLUS_INCLUDE_PATH`, `CUDA_HOME` (compilation fixes for A100)
  - `VLLM_ATTENTION_BACKEND=FLASH_ATTN` (optimization)
  - `TORCH_EXTENSIONS_DIR` and `XDG_CACHE_HOME` redirected to fast local SSD (`$PBS_JOBFS`)
- **Entry point:** Activates conda env at `/scratch/pg06/vm4618/envs/vllm_env` and runs `python test_vlm.py`

**Status:** References `test_vlm.py` which does **not** exist in this repository. This script likely needs updating to reference `main.py` instead.

---

## `sparse_event_pipeline/` — Sparse Event Localisation Sub-Pipeline

This directory focuses specifically on **Benchmark Category 1: Sparse Event Localisation** — detecting rare, outlier events in dashcam footage.

**Workflow:** `video_parser.py` → per-section frame folders + manifest JSON → `AIParser.py` → event output JSON.

---

### `sparse_event_pipeline/video_parser.py`

**Purpose:** Takes a video, splits it into overlapping time sections, extracts sampled JPEG frames into a per-section folder structure, and writes a manifest JSON describing every section. This is the first stage of the sparse event pipeline — it produces the frame files that `AIParser.py` consumes.

No video files are split or re-encoded — only frames are extracted. This avoids quality loss from re-encoding and is much faster than splitting video files.

**Key Dependencies:** `decord` (VideoReader), `PIL.Image`, `json`, `argparse`, `math`, `os`.

**Output folder structure:**
```
<frames_root>/<video_name>/<video_name>_section_0000/frame_00000.jpg
                                           ⋮                  /frame_00149.jpg
                        /<video_name>_section_0001/frame_00000.jpg
                                           ⋮
```

**Class: `FrameParser`**

| Constructor Parameter | Description | Default |
|-----------------------|-------------|---------|
| `video_path` | Path to input video file | (required) |
| `section_duration` | Section length in seconds | `300` (5 min) |
| `overlap` | Overlap between adjacent sections in seconds | `10` |
| `target_fps` | Frame sampling rate (frames per second of video) | `1.0` |
| `frame_width` | Resize width for extracted frames | `768` |
| `quality` | JPEG save quality (1–100) | `85` |
| `frames_root` | Root directory for output frame folders | `"./frames"` |

**Methods:**

| Method | Purpose |
|--------|---------|
| `run()` | Entry point. Computes section ranges, creates output directories, extracts frames for each section via `_extract_section_frames()`, and writes the manifest JSON to `<frames_root>/<video_name>_manifest.json`. |

**Manifest JSON format:**
```json
{
  "original_video": "/absolute/path/to/video.mp4",
  "video_name": "video_name",
  "fps": 30.0,
  "total_frames": 54000,
  "total_duration": 1800.0,
  "section_duration": 300,
  "overlap": 10,
  "target_fps": 1.0,
  "num_sections": 7,
  "sections": [
    {
      "section_id": 0,
      "start_frame": 0,
      "end_frame": 9000,
      "start_time_global": 0.0,
      "end_time_global": 300.0,
      "duration": 300.0,
      "frame_dir": "./frames/video_name/video_name_section_0000"
    }
  ],
  "frames_root": "./frames"
}
```

**CLI Interface:**

| Argument | Description | Default |
|----------|-------------|---------|
| `video_path` (positional) | Path to input video | (required) |
| `--duration` | Section duration in seconds | `300` |
| `--overlap` | Overlap between sections in seconds | `10` |
| `--fps` | Frame sampling rate | `1.0` |
| `--width` | Resize width for frames | `768` |
| `--quality` | JPEG quality | `85` |
| `--frames-root` | Output root directory | `"./frames"` |

**Status:** Fully functional and production-ready.

---

### `sparse_event_pipeline/AIParser.py`

**Purpose:** Reads frame images from disk and sends them to a VLM via the opencode.ai API gateway (Anthropic-compatible interface) to identify rare/noteworthy events. Lightweight — no frame extraction, no section logic, no deduplication. Just: load frames → call API → return parsed JSON.

**Key Dependencies:** `anthropic` (AsyncAnthropic), `asyncio`, `json`, `base64`, `os`, `glob`, `re`.

**Class: `AIEventParser`**

| Constructor Parameter | Description | Default |
|-----------------------|-------------|---------|
| `model_name` | Model identifier on the gateway | `"qwen3.7-plus"` |
| `max_concurrent_task` | Max concurrent API calls (via asyncio.Semaphore) | `3` |
| `api_key` | API key for the opencode gateway | `OPENCODE_API_KEY` env var |
| `base_url` | API base URL | `"https://opencode.ai/zen/go/v1"` |
| `max_retries` | Max retry attempts per API call | `2` |
| `request_timeout` | Per-request timeout in seconds | `600` |

**Methods:**

| Method | Purpose |
|--------|---------|
| `query(frame_paths, prompt)` (async) | Core method. Reads JPEG frames from disk, base64-encodes them, sends them to the VLM along with a prompt, parses the JSON response. Includes retry logic with exponential backoff (retries on 5xx/429, does not retry on other 4xx). Uses a prefill trick (`{"role": "assistant", "content": "{"}`) to force the model to output JSON. |
| `query_directory(frames_dir, prompt)` (async) | Convenience wrapper: globs `frame_*.jpg` from a directory (sorted chronologically) and calls `query`. |
| `query_multiple(batch_specs)` (async) | Runs multiple `query` calls concurrently via `asyncio.gather`, all bounded by the semaphore. Each item in batch_specs is a dict with `frame_paths` (list) and optional `prompt`. |
| `_parse_json(body)` (static) | Robust JSON parser: handles markdown code fences, missing leading `{` (prefill artifact), and extracts the first `{...}` via regex if needed. |

**Default prompt (set in `__init__`):** Instructs the VLM to look for rare, out-of-place, or noteworthy events in dashcam footage from forest/rural/tunnel environments. Ignores normal driving. Outputs JSON with an `interesting_events` array.

**Output JSON format (per query):**
```json
{
  "interesting_events": [
    {
      "section_id": 0,
      "frame": "frame_00042.jpg",
      "event_description": "A deer crosses the road from left to right",
      "why_interesting": "Uncommon wildlife encounter on an otherwise empty road",
      "terrain": "forest"
    }
  ]
}
```

**Status:** Complete.

---

### `sparse_event_pipeline/run.py`

**Purpose:** CLI entry point for `AIParser.py`. Uses argparse to accept a directory (or single frame file), constructs an `AIEventParser`, and runs a query.

**Key Dependencies:** `AIParser` (AIEventParser, DEFAULT_MODEL), `asyncio`, `argparse`, `json`, `os`.

**CLI Interface:**

| Argument | Description | Default |
|----------|-------------|---------|
| `input` (positional) | Directory of `frame_*.jpg` files OR a single frame file | (required) |
| `--prompt` | Custom prompt | (built-in event prompt) |
| `--output` | Write JSON response to file | `None` (prints to stdout) |
| `--model` | Model name on the gateway | `"qwen3.7-plus"` |
| `--api-key` | API key | `OPENCODE_API_KEY` env var |
| `--max-concurrent` | Max concurrent API calls | `3` |

**Status:** Complete.

---

### `vids/118014-714270866_medium.mp4`

**Purpose:** Sample dashcam video used as test input for the pipeline.

**Details:** Binary MP4 video file. Filename suggests it is a "medium" resolution/quality version of a dashcam recording (ID: `118014-714270866`). This is the only video in the `vids/` directory.

**Git status:** The entire `vids/` directory is git-ignored (per `.gitignore`).

---

## Architecture: Two API Provider Approaches

The codebase demonstrates two different LLM/VLM API strategies:

| Approach | File | Model | Interface |
|----------|------|-------|-----------|
| **Self-hosted vLLM** | `main.py` | Qwen2-VL-7B (vision), DeepSeek-V4-Flash (text) | Direct `vllm.LLM()` class |
| **Anthropic-compatible Gateway** | `sparse_event_pipeline/AIParser.py` | qwen3.7-plus | opencode.ai gateway via `AsyncAnthropic` |

Both pipelines use **Decord** for video reading. `main.py` does frame extraction inline; `video_parser.py` extracts frames to disk for `AIParser.py` to consume later.

---

## Known Issues & Missing Pieces

1. **`run_vllm.pbs` references non-existent file** — The PBS script runs `python test_vlm.py` which does not exist. It should likely reference `main.py`.

2. **`requirements.txt` typo** — Lists `vlm` instead of `vllm`.

3. **`main.py` hardcoded video path** — Bottom-of-file invocation uses `"test.mp4"` which does not exist. The actual video is at `vids/118014-714270866_medium.mp4`.

4. **No test suite** — No unit tests, integration tests, or test framework configuration exists.

5. **No CI/CD or containerization** — No Dockerfile, Makefile, GitHub Actions, or other automation.

---

## Summary of Implementation Status

| Component | Status |
|-----------|--------|
| Project design/spec (`AI.md`) | Complete |
| Full-video chunked pipeline (`main.py`) | Functional (hardcoded) |
| Video parser (`video_parser.py`) | Complete & production-ready |
| AI event parser (`AIParser.py`) | Complete with CLI |
| HPC job script (`run_vllm.pbs`) | Exists but references wrong file |
| Benchmark categories 2–4 | Not yet implemented |
| User Interface | Not yet implemented |
| Hybrid Search Module | Not yet implemented |
| Reporting frontend | Not yet implemented |
| Tests, CI/CD, Docker, Makefile | None exist |
