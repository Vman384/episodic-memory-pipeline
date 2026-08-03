# Codebase Documentation — Episodic Memory Pipeline

## Overview

This project is a **benchmarking pipeline** to assess **Vision Language Models' (VLMs) episodic memory capabilities** in long-form **egocentric dashcam video**. The input data consists of pre-extracted frame images (epoch-timestamped PNG/JPG files in a directory), sourced from dashcam recordings of vehicles driving through forests, rural roads, and tunnels, ranging from 6 minutes to 1 hour in duration. These videos have very low information density (few events), which intentionally stresses the model's ability to recall details over long time spans.

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

1. **Frame Parser** (`pipeline/frame_parser.py`) — Splits a directory of numerically named frame images into VLM-sized section folders. Supports configurable chunk size, frame step sampling, and copy/move semantics.

2. **AI Parser** (`pipeline/AIParser.py`) — Local vLLM wrapper. Exposes `call_llm` for text prompts and `call_vlm` for prompts with all image frames from a folder.

3. **Sparse Event Pipeline** (`pipeline/sparse_event_main.py`) — `SparseEventPipeline` creates a `ConfigLoader`, reads a JSON config file, creates `FrameParser` and `AIParser`, then runs the sparse-event workflow. A separate runner will later own CLI argument parsing.

5. **Prototype** (`main.py`) — Reference implementation using self-hosted vLLM with Qwen2-VL-7B. Reads video via Decord, chunks into 300-frame segments, generates descriptions and change-detection QA drafts via few-shot prompting. Hardcoded for Gadi A100 nodes but demonstrates the consecutive-comparison technique and single-model text+vision pattern. **Not pipeline-ified** — serves as design reference.

6. **HPC Job Script** (`run_vllm.pbs`) — PBS batch script for running the vLLM pipeline on NCI's Gadi cluster with A100 GPUs.

### What Is Not Yet Implemented

- **Temporal pipeline** (`timeline_builder.py`, `draft_questions.py`, `ingest_review.py`) — designed but not yet built.
- The dedicated benchmarks for categories 2–4 (attribute perception, spatial reasoning, counting).
- `run_vllm.pbs` references a non-existent file `test_vlm.py`.
- The `AI.md` design brief mentions a **User Interface**, a **Hybrid Search Module**, and a **Reporting frontend** — none exist.
- No test suite, CI/CD, Dockerfile, or Makefile.

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
├── pipeline/
│   ├── ConfigLoader.py
│   ├── AIParser.py
│   ├── frame_parser.py
│   ├── sparse_event_main.py
│   ├── temporal_main.py
│   └── prompts/
│       ├── temporal.txt
│       └── sparse_event_prompt.txt
├── vids/                           (git-ignored directory for video files)
├── configs/                        (example JSON config files for each pipeline stage)
│   ├── frame_split.json
│   ├── sparse_events.json
│   └── narrative_pass.json
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
- Input data characteristics: dashcam footage from forest, rural road, and tunnel driving. 6 minutes to 1 hour duration, minimal events, low information density.
- High-level pipeline structure: a master pipeline composed of sub-pipelines for each benchmark category.

**Status:** Reference document. Not all planned components have been implemented.

---

### `.gitignore`

**Purpose:** Version control exclusion rules.

**Contents:**
- `vids/` — Excludes all video files from git.
- `.venv/` — Excludes the Python virtual environment.

---

### `main.py`

**Purpose:** Coworker's prototype for full-video chunked description and change detection using self-hosted vLLM. **Not pipeline-ified** — serves as design reference demonstrating the consecutive-comparison technique.

**Key Dependencies:** `vllm` (LLM, SamplingParams), `decord` (VideoReader), `PIL.Image`, `json`, `os`.

**Functions:**

| Function | Purpose |
|----------|---------|
| `format_timestamp(seconds)` | Converts a float seconds value to `MM:SS` string format. |
| `sample_frames_from_indices(vr, start_idx, end_idx, num_samples=8, max_size=(448, 448))` | Uniformly samples and downscales frames from a Decord VideoReader. Returns a list of PIL Images. |
| `run_video_pipeline(video_path, ...)` | **Main entry point.** Processes video end-to-end. |

**Pipeline Flow:**
1. Opens the video via Decord, computes total frames and FPS.
2. Initializes a single vLLM model (Qwen2-VL-7B) used for both vision description and text comparison — halves GPU memory.
3. Iterates through the video in 300-frame chunks.
4. For each chunk: samples 8 downscaled frames → VLM description → few-shot LLM comparison with previous chunk → `{something_new, what_changed, question, answer}` JSON.
5. Stream-saves `full_history.json` and `change_analysis.json` after every chunk (crash-resilient).
6. The few-shot comparator prompt is well-calibrated: explicit "most chunks should be false" threshold, no meta-questions, three calibration examples.

**Status:** Prototype. Hardcoded scratch paths and model snapshot. Generates auto-QA drafts (circular ground truth — same model that describes also answers). No resume logic; restart reprocesses from chunk 0.

---

### `README.md`

**Purpose:** Project readme with structure overview and quick-start instructions.

**Contents:** Documents `main.py`, `frame_parser.py`, `AIParser.py`, `run.py`, and `prompts/descene_scene.txt`. Quick Start shows the three-stage workflow: frame partitioning → sparse event detection → narrative pass.

**Status:** Complete.

---

### `requirements.txt`

**Purpose:** Root-level Python dependencies.

**Contents:**
- `vllm` — VLM inference engine (used by `main.py`).
- `decord` — Video reading and decoding library (used by `main.py`).
- `anthropic>=0.120` — Anthropic Python SDK configured for the OpenCode Go gateway (`AIParser.py`).
- `python-dotenv` — Load environment variables from a `.env` file (`AIParser.py`).
- `tqdm` — Progress bar library.
- `Pillow` — Image manipulation library.

---

### `run_vllm.pbs`

**Purpose:** PBS job script for running the vLLM pipeline on NCI's Gadi HPC cluster.

**Job Configuration:**
- **Job name:** `qwen_vllm_large`
- **Queue:** `dgxa100` (NVIDIA A100 GPU nodes)
- **Resources:** 16 CPUs, 1 GPU, 64 GB RAM, 10-hour walltime, 50 GB local SSD
- **Modules:** `python3/3.11.7`, `cuda/12.2.2`
- **Entry point:** Activates conda env and runs `python test_vlm.py`

**Status:** References `test_vlm.py` which does **not** exist. Needs updating to reference `main.py` or a pipeline-ified entry point.

---

## `pipeline/` — Sparse Event Localisation Sub-Pipeline

This directory focuses on **Benchmark Category 1: Sparse Event Localisation** and serves as the foundation for the planned temporal/episodic memory pipeline.

**Workflow:** `run.py --config config.json` → `frame_parser.py` (split frames into sections) → `AIParser.call_vlm(prompt, section_dir)` for each section → per-section output JSON + aggregated `all_results.json`.

---

### `pipeline/frame_parser.py`

**Purpose:** Splits an existing directory of numerically named frame images into VLM-sized section folders. Works on pre-extracted frame directories (e.g., Boreas dataset camera images) — no video input, no Decord dependency. This is the first stage of the sparse event pipeline.

**Key Dependencies:** `shutil`, `pathlib.Path`, `argparse`.

**Output folder structure:**
```
<output_dir>/
    section_0000/1733343593917869.png
                1733343596067826.png
                ...
    section_0001/1733343656170615.png
                ...
```

**Class: `FrameParser`**

| Constructor Parameter | Description | Default |
|-----------------------|-------------|---------|
| `frames_dir` | Directory containing input frame files | (required) |
| `output_dir` | Output root directory | `<frames_dir>_sections` |
| `frames_per_section` | Maximum frames per section | `100` |
| `step_size` | Keep every Nth frame after sorting | `1` |
| `move` | Move frames instead of copying | `False` |
| `extensions` | Accepted image extensions (case-insensitive) | `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp` |

**Methods:**

| Method | Purpose |
|--------|---------|
| `create_section_dir()` | Sorts frame files by numeric stem, applies step sampling, partitions into per-section directories, and copies (or moves) files. Returns list of created section directory paths. |

**CLI Interface:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Required JSON config file containing all frame-processing settings | (required) |

**Status:** Complete.

---

### `pipeline/AIParser.py`

**Purpose:** Provides a small interface to a locally hosted vLLM model for text and multi-image generation.

**Key Dependencies:** `vllm`, `Pillow`, and `pathlib.Path`.

**Class: `AIParser`**

| Constructor Parameter | Description | Default |
|-----------------------|-------------|---------|
| `model` | Local model identifier or path | `None` |
| `temperature` | Sampling temperature | `0.2` |
| `max_tokens` | Maximum generated tokens | `100` |
| `enforce_eager` | Disable CUDA graph capture | `True` |
| `dtype` | Model data type | `"half"` |
| `max_model_len` | Maximum model context length | `4096` |
| `gpu_memory_utilization` | Fraction of GPU memory available to vLLM | `0.9` |

**Public methods:**

| Method | Purpose |
|--------|---------|
| `call_llm(prompt)` → `str` | Sends a text prompt to vLLM and returns generated text. |
| `call_vlm(prompt, folder_path)` → `str` | Loads all supported image frames from `folder_path`, sorts them by numeric filename, sends them together as a multi-image input, and returns generated text. |

Configuration loading belongs to the pipeline entry point, such as
`pipeline/sparse_event_main.py`, which reads the JSON file
and passes the model settings into the `AIParser` constructor.

**Output structure (per section):**
```
<output_dir>/all_results.json                (aggregated single-file view)
<output_dir>/section_0000_output/result.json (per-section persistent artifact)
<output_dir>/section_0001_output/result.json
...
```

**Status:** Complete.

---

### `pipeline/ConfigLoader.py`

**Purpose:** Shared configuration component used by pipeline entry points.
`ConfigLoader(config_path).load()` validates that the JSON file exists, parses
it, verifies that it contains an object, and returns the configuration
dictionary.

---

### `pipeline/sparse_event_main.py`

**Purpose:** Defines `SparseEventPipeline`, which loads configuration through
`ConfigLoader`, creates `FrameParser` and `AIParser`, splits frames into
sections, queries the local vLLM model for each section, and writes results.

The class is currently invoked programmatically. A separate main file will
later handle CLI argument parsing.

**Config file schema:**

| Key | Description | Default |
|-----|-------------|---------|
| `task` | Benchmark category: 1 = sparse events, 2 = temporal/narrative | (required) |
| `frames_dir` | Directory of frame images | (required) |
| `sections_dir` | Intermediate section output | `"./sections"` |
| `output` | Final VLM results directory | (required) |
| `frames_per_section` | Max frames per section | `100` |
| `step` | Keep every Nth frame | `1` |
| `model` | Local vLLM model name or path | (required) |
| `temperature` | Sampling temperature | `0.2` |
| `max_tokens` | Maximum generated tokens | `100` |
| `enforce_eager` | Disable CUDA graph capture | `true` |
| `dtype` | Model data type | `"half"` |
| `max_model_len` | Maximum model context length | `4096` |
| `gpu_memory_utilization` | Fraction of GPU memory available to vLLM | `0.9` |

**Task → prompt mapping:**

| `task` | Prompt file | Purpose |
|--------|-------------|---------|
| 1 | `prompts/sparse_event_prompt.txt` | Sparse event localisation |
| 2 | `prompts/temporal.txt` | Full scene description |

**Status:** Complete. Configuration is loaded by `ConfigLoader`; this entry
point creates `FrameParser` and `AIParser` and runs the sparse-event pipeline.

---

### `pipeline/prompts/temporal.txt`

**Purpose:** Full-narrative VLM prompt for the temporal pipeline's Stage 1. Asks the VLM to describe everything observable in a section of dashcam footage — terrain, road conditions, weather, lighting, notable objects, and changes/transitions — rather than filtering for rare events only. The resulting structured JSON (`narrative.json`) is designed to feed the upcoming `timeline_builder.py`.

**Output JSON schema:** `{summary, terrain, road_conditions, weather, lighting, notable_objects[], changes[], interesting_events[]}`

**Status:** Complete.

---

### `pipeline/prompts/sparse_event_prompt.txt`

**Purpose:** VLM prompt for the sparse event localisation benchmark (task 1). Asks the VLM to filter for rare or noteworthy events — unusual occurrences, unexpected objects, sudden changes — while ignoring normal driving. Output is a JSON object with an `interesting_events` array.

**Output JSON schema:** `{interesting_events[]}`

**Status:** Complete.

---

### `configs/`

**Purpose:** Example JSON config files for each benchmark run. The `task` field selects the benchmark category and corresponding VLM prompt. `run.py` reads these files to run the full pipeline (frame splitting + VLM query).

**Files:**

| File | task | Purpose |
|------|------|---------|
| `sparse_events.json` | 1 | Sparse event localisation (uses `prompts/sparse_event_prompt.txt`) |
| `narrative_pass.json` | 2 | Full scene description (uses `prompts/temporal.txt`) |

**Status:** Complete.

---

## Architecture: Local vLLM Implementations

The codebase contains a prototype and a pipeline wrapper around local vLLM:

| Approach | File | Model | Interface |
|----------|------|-------|-----------|
| **Prototype** | `main.py` | Qwen2-VL-7B (vision + text) | Direct `vllm.LLM()` class, offline HF cache |
| **Pipeline wrapper** | `pipeline/AIParser.py` | Configured local vision-language model | Direct `vllm.LLM()` class |

The coworker's prototype (`main.py`) reads video directly via Decord. The pipeline wrapper reads pre-extracted frame folders and sends all frames in each section to the local vLLM model. Both paths require GPU nodes with the model available locally.

---

## Known Issues & Missing Pieces

1. **`run_vllm.pbs` references non-existent file** — The PBS script runs `python test_vlm.py` which does not exist. It should likely reference `main.py`.

2. **`main.py` hardcoded paths** — Bottom-of-file invocation uses absolute scratch paths and a model snapshot path specific to one Gadi filesystem. Not portable.

3. **No test suite** — No unit tests, integration tests, or test framework configuration exists.

4. **No CI/CD or containerization** — No Dockerfile, Makefile, GitHub Actions, or other automation.

---

## Summary of Implementation Status

| Component | Status |
|-----------|--------|
| Project design/spec (`AI.md`) | Complete |
| Coworker prototype (`main.py`) | Functional (hardcoded, reference only) |
| Frame parser (`frame_parser.py`) | Complete |
| AI parser (`pipeline/AIParser.py`) | Complete, local vLLM wrapper with reusable `call_llm`/`call_vlm` |
| Sparse event pipeline (`pipeline/sparse_event_main.py`) | Complete — class-based workflow; runner pending |
| Narrative prompt (`prompts/temporal.txt`) | Complete |
| Sparse event prompt (`prompts/sparse_event_prompt.txt`) | Complete |
| Example configs (`configs/*.json`) | Complete |
| HPC job script (`run_vllm.pbs`) | Exists but references wrong file |
| Temporal pipeline (timeline builder, draft questions, review) | Designed, not yet implemented |
| Benchmark categories 2–4 | Not yet implemented |
| User Interface | Not yet implemented |
| Hybrid Search Module | Not yet implemented |
| Reporting frontend | Not yet implemented |
| Tests, CI/CD, Docker, Makefile | None exist |
