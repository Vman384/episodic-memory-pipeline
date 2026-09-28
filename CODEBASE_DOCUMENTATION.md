# Codebase Documentation — Episodic Memory Pipeline

## Overview

This project is a **benchmarking pipeline** to assess **Vision Language Models' (VLMs) episodic memory capabilities** in long-form **egocentric dashcam video**. The input data consists of pre-extracted frame images (epoch-timestamped PNG/JPG files in a directory), sourced from dashcam recordings of vehicles driving through forests, rural roads, and tunnels, ranging from 6 minutes to 1 hour in duration. These videos have very low information density (few events), which intentionally stresses the model's ability to recall details over long time spans.

### Benchmark Categories

| # | Category | Description |
|---|----------|-------------|
| 1 | **Sparse Event Localisation** | Detecting rare, outlier events without hallucination. Includes temporal perception, existence verification, noteworthy object/moment queries, and deceptive questioning. |
| 2 | **Attribute Perception & Temporal Order** | Testing if the model truly remembers event order or just guesses from training distribution. |
| 3 | **Spatial Reasoning** | Physically grounded reasoning (e.g., distance estimation) in environments with scarce reference points. |
| 4 | **Counting** | Episodic-memory-based counting across long time spans where objects cannot all be seen in a single frame. |

---

## Project Status Summary

### What Is Currently Implemented

1. **Frame Parser** (`pipeline/frame_parser.py`) — Splits a directory of timestamp-named frame images into VLM-sized section folders. Supports configurable section size, per-section frame stride and inter-section skip, copy/move semantics, and optional per-frame JPEG re-encoding via `ImagePreprocessor`.

2. **Image Preprocessor** (`pipeline/image_preprocessor.py`) — Re-encodes frame images as resized JPEGs while sections are built, shrinking API request payloads and vision-token usage.

2. **Timeframe Converter** (`pipeline/timeframe_converter.py`) — Converts temporal event `start_frame` and `end_frame` values into elapsed video seconds relative to the earliest frame in the source directory.

3. **AI Parser** (`pipeline/AIParser.py`) — Configurable local vLLM or OpenAI-compatible API wrapper. `call_llm` handles text prompts and `call_vlm` handles multiple images from a section directory.

4. **Sparse Event Pipeline** (`pipeline/SparseEventPipeline.py`) — `SparseEventPipeline` reads a JSON config file, creates `FrameParser` and `AIParser`, partitions frames, and runs staged section extraction, LLM review/filtering, and question generation.

5. **Temporal Pipeline** (`pipeline/TemporalPipeline.py`) — Provides resumable section extraction, timestamp-sorted timeline construction, windowed LLM merging, storyline generation, and question generation from a human-reviewed timeline.

6. **Top-Level CLI Dispatcher** (`main.py`) — Accepts `--mode` values for sparse event, temporal, spatial, and counting benchmarks. The temporal branch also accepts `--stage extract|timeline|questions`; spatial and counting remain placeholders.

7. **Prototype/Test Script** (`test_vlm.py`) — Reference implementation using self-hosted vLLM with Qwen2-VL-7B. It reads video via Decord, chunks into 300-frame segments, and generates descriptions and change-detection QA drafts via few-shot prompting.

8. **Legacy HPC Job Script** (`vllm.pbs`) — PBS batch script for the old prototype on NCI's Gadi cluster.

9. **Sparse-Event HPC Job Script** (`sparse_event.pbs`) — PBS batch script for the current sparse-event CLI.

10. **Forest Run Script** (`forest_run.sh`) — Combined PBS job script for the forest K-01 dataset. Holds an API version and a local vLLM version of both the sparse and temporal runs; the version that is not being run is commented out before submission.

### What Is Not Yet Implemented

- The dedicated benchmarks for categories 2–4 (attribute perception, spatial reasoning, counting).
- `temporal_event.pbs` runs the local vLLM temporal configuration on four GPUs.
- A **User Interface**, **Hybrid Search Module**, and **Reporting frontend** are not implemented.
- No test suite, CI/CD, Dockerfile, or Makefile.

---

## Full Directory Tree

```
episodic-memory-pipeline/
├── .gitignore
├── main.py
├── README.md
├── run_timeframe_converter.py
├── requirements.txt
├── test_vlm.py
├── vllm.pbs
├── sparse_event.pbs
├── temporal_event.pbs
├── forest_run.sh
├── CODEBASE_DOCUMENTATION.md
├── pipeline/
│   ├── ConfigLoader.py
│   ├── AIParser.py
│   ├── frame_parser.py
│   ├── image_preprocessor.py
│   ├── timeframe_converter.py
│   ├── SparseEventPipeline.py
│   ├── TemporalPipeline.py
│   └── prompts/
│       ├── temporal_vlm.txt
│       ├── temporal_llm_filter.txt
│       ├── temporal_storyline.txt
│       ├── temporal_question_gen.txt
│       ├── sparse_event_prompt.txt
│       ├── sparse_event_filter.txt
│       └── sparse_event_question_gen.txt
├── boreas-*/                       (sample frame data)
├── configs/                        (pipeline configuration files)
│   ├── sparse_events.json
│   └── .temporal_events.json
```

---

## Gadi Storage Layout

All project data has moved from `/scratch/pg06/FYP2026S1_3473` to the project's
mass-data storage at `/g/data/pg06/FYP2026S1_3473`. The configs, PBS job
scripts, and pipeline scripts point there for frames, outputs, and the Hugging
Face cache.

| Location | Contents |
|----------|----------|
| `/g/data/pg06/FYP2026S1_3473` | Boreas frame datasets, pipeline outputs, forest dataset downloads, Hugging Face cache |
| `/scratch/pg06/vm4618/envs/vllm_env` | Shared Python environment with vLLM (used by every Gadi job script) |
| `/scratch/pg06/vm4618/huggingface_cache` | Qwen2-VL-7B cache used by the `vedansh/` prototype only |

Because the venv remains on scratch, the Gadi job scripts request both
filesystems: `#PBS -l storage=scratch/pg06+gdata/pg06`.

---

## File-by-File Documentation

---

### Project Context

**Purpose:** The project context defines the goal of benchmarking VLM episodic memory with long-form egocentric dashcam footage across sparse-event, temporal-order, spatial-reasoning, and counting tasks.

 Reference context. Sparse-event and temporal modes have connected pipelines. Temporal extraction, timeline construction, storyline generation, and question generimplemented; human review remains part of the intended data-creation workflow.

---

### `.gitignore`

**Purpose:** Version control exclusion rules.

**Contents:**
- `vids/` — Excludes all video files from git.
- `.venv/` — Excludes the Python virtual environment.
- `.env` — Excludes local environment variables and secrets.

---

### `main.py`

**Purpose:** Top-level command-line dispatcher for the benchmark modes.

**Command-line interface:**

```bash
python main.py --mode sparse
```

The accepted values are `sparse`, `temporal`, `spatial`, and `counting`. `sparse` imports `pipeline/SparseEventPipeline.py` and loads three sparse prompts, while `temporal` imports `pipeline/TemporalPipeline.py` and loads four temporal prompts. Spatial and counting remain placeholders.

The optional `--stage` argument applies to `sparse` and `temporal` mode:

```bash
python main.py --mode sparse --stage extract
python main.py --mode sparse --stage review
python main.py --mode sparse --stage questions

python main.py --mode temporal --stage extract
python main.py --mode temporal --stage timeline
python main.py --mode temporal --stage questions
```

With no `--stage`, sparse mode runs `extract` followed by `review`, and
temporal mode runs `extract` followed by `timeline`. The `questions` stages are
intentionally separate so a human can review and correct `events.json` or
`timeline.json` first.

**Functions:**

| Function | Purpose |
|----------|---------|
| `main()` | Parses `--mode` and dispatches to the selected benchmark mode. |

 Active dispatcher. Sparse-event and all temporal stages are connected; spatial and counting are pla

---

### `README.md`

**Purpose:** Project readme with current CLI usage, supported modes, sparse-event workflow, and configuration details.

**Contents:** Documents `main.py`, `pipeline/frame_parser.py`, `pipeline/AIParser.py`, `pipeline/SparseEventPipeline.py`, `pipeline/TemporalPipeline.py`, temporal stages, and the current configuration files. The quick start uses `python main.py --mode sparse`.


---

### `requirements.txt`

**Purpose:** Root-level Python dependencies.

**Contents:**
- `vllm` — VLM inference engine used by the prototype and sparse-event pipeline.
- `decord` — Video reading and decoding library used by `test_vlm.py`.
- `anthropic>=0.120` — Anthropic Python SDK listed for the AI parser integration.
- `openai>=1.66.0` — OpenAI-compatible Responses API client.
- `python-dotenv` — Load environment variables from a `.env` file.
- `tqdm` — Progress bar library.
- `Pillow` — Image manipulation library.

---

### `vllm.pbs`

**Purpose:** PBS job script for running the vLLM pipeline on NCI's Gadi HPC cluster.

**Job Configuration:**
- **Job name:** `qwen_vllm_large`
- **Queue:** `gpuhopper`
- **Resources:** 12 CPUs, 1 GPU, 25-minute walltime, 50 GB local SSD
- **Modules:** `python3/3.11.7`, `cuda/12.2.2`
- **Entry point:** Activates the environment and runs `python main.py`

 Exists, but currently invokes `python main.py` without the required `--mode`

---

### `sparse_event.pbs`

**Purpose:** PBS job script for running the current sparse-event pipeline on Gadi.

**Entry point:** Loads Python and CUDA, activates the shared vLLM environment,
changes to the PBS working directory, and runs:

```bash
python3 main.py --mode sparse --stage extract
python3 main.py --mode sparse --stage review
```

Submit it from the repository root with `qsub sparse_event.pbs`.

**Resources:** 48 CPUs, 4 GPUs, 1024 GB memory, and a 1-hour walltime.

**Prerequisites:** Requires the local vLLM environment and the cached
`Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` model. It exports
`VLLM_USE_DEEP_GEMM=0` because DeepGEMM's JIT requires NVCC >= 12.3 while the
`cuda/12.2.2` module provides 12.2.2; vLLM then falls back to its CUTLASS FP8
kernels.

---

### `temporal_event.pbs`

**Purpose:** PBS wrapper for the temporal pipeline.

**Entry point:** Runs:

```bash
python3 main.py --mode temporal
```

With no `--stage`, this invokes temporal extraction followed by timeline
construction. Question generation must be run separately after reviewing
`timeline.json`.

**Current status:** The script is configured for the local Qwen vLLM temporal
configuration. It requests four GPUs, loads CUDA, enables offline Hugging Face
resolution, uses the model already present in the shared cache, and exports
`VLLM_USE_DEEP_GEMM=0` so vLLM does not JIT-compile DeepGEMM kernels with the
`cuda/12.2.2` module's NVCC, which is older than DeepGEMM's 12.3 minimum.

---

### `run_all_local.sh`

**Purpose:** PBS batch script that runs the sparse-event pipeline with the local
vLLM backend over the remaining Boreas lists, extract then review per list, in a
single job.

**Job configuration:** `gpuhopper` queue, 48 CPUs, 4 GPUs (one full node), 1024 GB
memory, 5-hour walltime, and `scratch/pg06+gdata/pg06` storage.

**Entry point:** For each list it rewrites only the per-list `frames_dir`,
`sections_dir`, and `output` paths in `configs/sparse_events.json`, then runs
`python3 main.py --mode sparse` (extract then review), the same stages as
`sparse_event.pbs`. It exports `VLLM_USE_DEEP_GEMM=0` for the same reason as the
other local job scripts. `backend`, `model`, and `tensor_parallel_size` are read
from the config: the supplied sparse config uses local vLLM with
`Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` sharded across four GPUs. The Hugging Face
cache resolves the repo id offline. The original config is restored on exit, and
completed section results are reused so the job can be resubmitted.

---

### `scripts/run_all_api.sh`

**Purpose:** PBS batch script that runs the sparse-event pipeline over the Boreas
lists on the CPU-only `copyq` queue.

**Entry point:** Before the loop it forces `backend: "api"` and
`model: "gpt-5.6-luna"` in `configs/sparse_events.json`, so no manual config edit
is needed. For each list it then rewrites only the per-list `frames_dir`,
`sections_dir`, and `output` paths and runs `python3 main.py --mode sparse`. The
original config is restored on exit.

---

### `forest_run.sh`

**Purpose:** PBS batch script for the forest K-01 dataset. It runs both the
sparse and temporal pipelines in one job and holds an API version and a local
vLLM version; the PBS header and run blocks of the version that is not being run
are commented out before submission.

**Entry point:** Loads Python and CUDA, activates the shared vLLM environment,
rewrites `configs/sparse_events.json` and `configs/temporal_events.json` with the
K-01 `frames_dir`, `sections_dir`, and `output` paths, then runs
`python3 main.py --mode sparse` and `python3 main.py --mode temporal`. The API
version also sets `backend: "api"` and `model: "gpt-5.6-luna"`; the local
version sets `backend: "local"` and the Qwen VL model used by each config. The
original configs are restored on exit, and completed section results are reused
so the job can be resubmitted. Outputs are written next to the dataset under
`forest_dataset/K-01_sparse_outputs/` and `forest_dataset/K-01_temporal_outputs/`.

**Resources:** The local header requests the `gpuhopper` queue (48 CPUs, 4 GPUs,
1024 GB memory, 10-hour walltime); the API header requests the CPU-only `copyq`
queue (1 CPU, 4 GB memory, 10-hour walltime). Both request
`scratch/pg06+gdata/pg06` storage.

---

## `pipeline/` — Pipeline Components

This directory contains the shared helpers plus the sparse-event and
temporal-order benchmark pipelines.

**Workflow:** `python main.py --mode sparse` → `SparseEventPipeline` → `ConfigLoader` → `FrameParser` (split frames into sections) → prompt resolution → `AIParser.call_vlm(prompt, section_dir)` → result persistence.

---

### `pipeline/frame_parser.py`

**Purpose:** Splits an existing directory of timestamp-named frame images into VLM-sized section folders. Works on pre-extracted frame directories (e.g., Boreas dataset camera images named `<epoch-microseconds>.png` or WildScenes images named `<epoch-seconds>.<fraction>.png`) — no video input, no Decord dependency. This is the first stage of the sparse event pipeline.

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
| `frames_per_section` | Number of frames sampled per section (the final section may be shorter) | `100` |
| `step_size` | Stride between sampled frames inside one section | `1` |
| `skip` | Frames from a section's last sampled frame to the next section's first frame; defaults to `step_size` | `None` (uses `step_size`) |
| `move` | Move frames instead of copying | `False` |
| `preprocessor` | Optional `ImagePreprocessor` re-encoding sampled frames as resized JPEGs | `None` |
| `extensions` | Accepted image extensions (case-insensitive) | `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp` |

**Methods:**

| Method | Purpose |
|--------|---------|
| `create_section_dir()` | Sorts frame files by parsed frame timestamp (`parse_frame_timestamp`), samples `frames_per_section` frames per section at stride `step_size`, and starts the next section `skip` frames after the last sampled frame. Copies (or moves) files. When a `preprocessor` is provided, each sampled frame is re-encoded as a JPEG (`.jpg` suffix, same filename stem) instead of being copied. Also writes a `sections.json` manifest mapping each section to its first/last frame and elapsed start/end seconds relative to the first sampled frame. Returns list of created section directory paths. |

**CLI Interface:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Required JSON config file containing all frame-processing settings | (required) |



---

### `pipeline/image_preprocessor.py`

**Purpose:** Re-encodes frame images as resized JPEGs while `FrameParser`
writes sections. Full-resolution camera frames produce base64 request payloads
that API backends reject and vision tokens that local models must pay for;
JPEG re-encoding shrinks payloads dramatically and downscaling keeps
multi-image requests within the vision-token budget of API models such as
`grok-4.6`. Images are never upscaled and timestamped filenames keep their stem.

**Key Dependencies:** `Pillow`, `pathlib.Path`.

**Class: `ImagePreprocessor`**

| Constructor Parameter | Description | Default |
|-----------------------|-------------|---------|
| `max_size` | Longest output side in pixels; `None` re-encodes without resizing | `None` |
| `quality` | JPEG quality, `1` to `95` | `90` |

| Member | Purpose |
|--------|---------|
| `SUFFIX` | Output filename suffix (`.jpg`) |
| `from_config(config)` | Build a preprocessor from `image_max_size` / `image_quality` config keys, or `None` when neither is set |
| `process(src, dst)` | Convert `src` to RGB, optionally fit the longest side to `max_size`, and save as a JPEG at `dst` |

Both pipelines pass `ImagePreprocessor.from_config(self.config)` to
`FrameParser`, so frames are only preprocessed when a configuration opts in.

---

### `pipeline/timeframe_converter.py`

**Purpose:** Converts frame timestamps from temporal event outputs into
elapsed seconds in the overall video. The earliest frame in the source
directory is treated as time zero. Frame filenames must be timestamps in either
the Boreas format `<epoch-microseconds>.png` or the WildScenes format
`<epoch-seconds>.<fraction>.png`; both are parsed to microseconds by
`parse_frame_timestamp(frame)`, which raises `ValueError` for other names.

**Functions:**

| Function | Purpose |
|----------|---------|
| `parse_frame_timestamp(frame)` → `int` | Parse a frame filename to epoch microseconds for either supported format. |

**Class: `TimeframeConverter`**

| Method | Purpose |
|--------|---------|
| `frame_to_seconds(frame)` → `float` | Convert one frame filename or timestamp to elapsed seconds. |
| `timeframe_to_seconds(start_frame, end_frame)` → `dict` | Return `start_seconds` and `end_seconds` for one temporal event. |

`parse_frame_timestamp` is also used by `FrameParser`, `AIParser`, and the
sparse/temporal event sort keys so all ordering and seconds math share one
filename parser.

For temporal runs the conversion is redundant: per-section seconds are computed
from the actual frame timestamps when `FrameParser` builds the sections and
attached to every event in each section result and in `timeline.json`,
so the model never has to produce exact timestamps itself. The class remains a
standalone utility for manual conversion.

---

### `run_timeframe_converter.py`

**Purpose:** Reads `timeline.json`, converts every event's `start_frame` and
`end_frame` with `TimeframeConverter`, and writes the result to
`timeline_with_seconds.json`.

**Usage:**

```bash
python3 run_timeframe_converter.py
python3 run_timeframe_converter.py --timeline input.json --output output.json
```

The source frame directory must be available and contain timestamped
filenames.

---

### `pipeline/AIParser.py`

**Purpose:** Provides a small interface for local vLLM or OpenAI-compatible API text and multi-image generation.

**Key Dependencies:** `vllm` for local mode, `openai` for API mode, `Pillow`, and `pathlib.Path`.

**Class: `AIParser`**

| Constructor Parameter | Description |
|----------------------|-------------|
| `config` | Complete JSON configuration dictionary |

The relevant configuration keys are:

| Key | Description |
|-----|-------------|
| `backend` | `local` for vLLM or `api` for the OpenAI-compatible Responses API |
| `model` | Local model path/name or API model identifier |
| `temperature` | Local vLLM sampling temperature; falls back to `0.2` when `null` |
| `api_temperature` | Responses API sampling temperature; omitted from the request when `null` or unset |
| `reasoning_effort` | Optional Responses API reasoning effort, such as `low` or `none` |
| `max_tokens` | Maximum generated output tokens |
| `enforce_eager` | vLLM CUDA graph setting used in local mode |
| `dtype` | vLLM model data type used in local mode |
| `max_model_len` | vLLM maximum model context length |
| `gpu_memory_utilization` | vLLM GPU memory fraction |
| `tensor_parallel_size` | Number of GPUs used by vLLM |
| `api_base_url` | OpenAI-compatible API base URL or full `/responses` endpoint |
| `api_key_env` | Environment variable containing the API key |

In API mode, `AIParser` sends an identifying user agent, an OpenCode client
header, and one stable `x-opencode-session` ID for the process. API errors retain
the provider response body and request ID when available.

**Public methods:**

| Method | Purpose |
|--------|---------|
| `call_llm(prompt, max_tokens=None)` → `str` | Sends a text prompt to the selected backend and returns generated text. An optional per-call token limit overrides the configured default. |
| `call_vlm(prompt, folder_path)` → `str` | Loads all supported image frames from `folder_path`, sorts them by parsed frame timestamp, includes an in-memory ordered filename manifest in the prompt, sends them together to the selected backend, and returns generated text. |

Configuration loading belongs to the pipeline classes, which read the JSON file and pass the complete dictionary into the `AIParser` constructor. In API mode, the API key is read from the environment variable named by `api_key_env`; it is not stored in configuration files.

**Output structure (per section):**
```
<output_dir>/all_results.json                (aggregated single-file view)
<output_dir>/section_0000_output/result.json (transient, deleted after aggregation)
<output_dir>/section_0001_output/result.json
...
```

Each `result.json` contains the raw VLM `response` string plus a parsed `events`
list where every event carries `start_seconds` and `end_seconds` taken from its
section's window in the `sections.json` manifest written by `FrameParser`. The
seconds are derived from the actual frame timestamps, never from model output.
Both pipelines delete the section frame images as each section is processed and
remove the `section_*_output` folders once `all_results.json` has been written,
leaving the aggregate as the only extraction artifact.

 Implemented for local vLLM and OpenAI-compatible API back
calls support an optional per-call `max_tokens` override for longer storyline
and question-generation responses.

---

### `pipeline/TemporalPipeline.py`

**Purpose:** Runs the temporal-order benchmark pipeline in independently
runnable stages. It partitions frames, extracts structured section observations,
sorts events by their frame timestamps, merges continuing or duplicate events,
creates a reviewable storyline, and generates questions from a human-verified
timeline.

**Stages:**

| Stage | Behavior |
|-------|----------|
| `extract` | Creates frame sections, queries the VLM with `temporal_vlm.txt`, writes one result per section, and skips existing results for resumability. Parsed events from each response are stored with the section's elapsed `start_seconds`/`end_seconds` from `sections.json`. Section frames are deleted once the VLM has processed them, and the per-section result folders are deleted after `all_results.json` is written. |
| `timeline` | Loads section results from `all_results.json`, parses their `events` arrays, sorts them by parsed `start_frame` timestamp, merges events in LLM windows, re-attaches per-section seconds after merging, and writes `timeline.json` and `storyline.txt`. |
| `questions` | Loads `timeline.json`, intended to be human-reviewed first, and writes generated temporal questions to `questions.json`. |

When no stage is supplied, the pipeline runs `extract` followed by `timeline`.
The question stage is never included in the default run so that human review can
occur between timeline creation and question generation.

**Timeline behavior:** Event ordering is determined in Python from the parsed
timestamp in each frame filename. The LLM is instructed only to merge duplicate
or continuing observations, preserve order, and avoid inventing events. The
`merge_window` configuration controls how many sorted events are sent in one
merge request. Invalid model JSON is handled with a warning and the unmerged
events are retained for review.

**Output:**

```text
<output>/
    all_results.json
    timeline.json
    storyline.txt
    questions.json
```

`all_results.json` is the only extraction artifact: section frames and the
`section_*_output` folders are deleted after it is written. `questions.json` is
produced only by the `questions` stage. Each generated
question is expected to contain a type, options, answer indices, event IDs, and
frame evidence for review and later grading.

 Implemented, with human review required for benchma
timeline and question artifacts.

---

### `pipeline/ConfigLoader.py`

**Purpose:** Shared configuration component used by pipeline entry points.
`ConfigLoader(config_path).load()` validates that the JSON file exists, parses
it, verifies that it contains an object, and returns the configuration
dictionary.

---

### `pipeline/SparseEventPipeline.py`

**Purpose:** Defines `SparseEventPipeline`, which loads configuration through
`ConfigLoader`, creates `FrameParser` and `AIParser`, reuses existing section
directories or splits frames into sections, and runs the staged sparse-event
workflow.

The class is invoked by `main.py` when the user selects
`--mode sparse`.

**Stages:**

| Stage | Behavior |
|-------|----------|
| `extract` | Creates frame sections, queries the VLM with `sparse_event_prompt.txt`, writes one result per section, and skips existing results for resumability. Parsed detections are stored with the section's elapsed `start_seconds`/`end_seconds` from `sections.json` and a `frame_seconds` value computed from the detected frame's timestamp. Section frames are deleted once the VLM has processed them, and the per-section result folders are deleted after `all_results.json` is written. |
| `review` | Loads section results from `all_results.json`, parses their `interesting_events` into detections, sorts them by parsed `frame` timestamp, merges duplicate observations and filters spurious detections in LLM windows, re-attaches seconds after merging, and writes `events.json`. |
| `questions` | Loads `events.json`, intended to be human-reviewed first, and writes generated sparse questions to `questions.json`. |

When no stage is supplied, the pipeline runs `extract` followed by `review`.
The question stage is never included in the default run so that human review can
occur between review and question generation.

**Review behavior:** Detection ordering and event seconds are computed in
Python from the parsed timestamp in each frame filename. The LLM is
instructed only to merge duplicate observations, remove spurious detections
(with reasons in a `removed` list), preserve order, and avoid inventing events.
The `review_window` configuration controls how many sorted detections are sent
in one merge request. Invalid model JSON is handled with a warning and the
unmerged detections are retained for review.

**Output:**

```text
<output>/
    all_results.json
    events.json
    questions.json
```

`all_results.json` is the only extraction artifact: section frames and the
`section_*_output` folders are deleted after it is written. `questions.json` is
produced only by the `questions` stage. Each generated
question carries a type, options, answer indices, event IDs, and frame
evidence. Deceptive questions refer to an invented plausible false event
(always answered "no"), and noteworthy questions use three. False events are
invented per question and never reused across questions, so every question has
its own incorrect answers. Each invented false event is listed once in the
top-level `false_events` array with its `question_id` and rationale so a
reviewer can confirm it never occurred.

**Shared config file schema:**

| Key | Description | Default |
|-----|-------------|---------|
| `task` | Benchmark category metadata: `1` = sparse events, `2` = temporal/narrative | (required) |
| `frames_dir` | Directory of frame images | (required) |
| `sections_dir` | Intermediate section output | `"./sections"` |
| `output` | Intended final VLM results directory | (required) |
| `frames_per_section` | Frames sampled per section | `100` |
| `step` | Stride between sampled frames inside one section | `1` |
| `skip` | Frames from a section's last sampled frame to the next section's first frame; defaults to `step` | (optional) |
| `model` | Local vLLM model name/path or API model identifier | (required) |
| `backend` | `local` for vLLM or `api` for the Responses API | `local` |
| `temperature` | Local vLLM sampling temperature; `null` falls back to `0.2` | (optional) |
| `api_temperature` | Responses API sampling temperature; omitted from API requests when `null` or unset | (optional) |
| `reasoning_effort` | Responses API reasoning effort, such as `low` or `none` | (optional) |
| `max_tokens` | Default maximum generated output tokens. Used by VLM calls and LLM calls without an override | `100` |
| `review_window` | Maximum sorted detections supplied to one sparse review call | `50` |
| `merge_window` | Maximum sorted events supplied to one temporal merge call | `50` |
| `storyline_max_tokens` | Maximum tokens for the temporal storyline call | (optional) |
| `question_max_tokens` | Maximum tokens for temporal/sparse question generation | (optional) |
| `enforce_eager` | Disable CUDA graph capture | `true` |
| `dtype` | Model data type | `"half"` |
| `max_model_len` | Maximum model context length | `4096` |
| `gpu_memory_utilization` | Fraction of GPU memory available to vLLM | `0.9` |
| `tensor_parallel_size` | Number of GPUs used to shard the model | `1` |
| `api_base_url` | API base URL or full `/responses` endpoint | (optional) |
| `api_key_env` | Environment variable containing the API key | (optional) |

**Configuration selection guide:**

- `frames_per_section` controls how many images are sent in one VLM request. Lower values reduce memory use; start around `5-10` for high-resolution frames.
- `step` is the stride between sampled frames inside a section, and `skip` is the number of frames from a section's last sampled frame to the next section's first frame. `skip` defaults to `step`, which samples every `step`-th frame without gaps; use a larger value to reduce compute between sections at the cost of temporal detail.
- `max_model_len` is the total token budget for the prompt, visual image tokens, and generated output. It is not a duration or frame count. Start at `4096`; increase to `8192` if requests are too long, or reduce it and/or the section size if GPU memory is exhausted.
- `max_tokens` reserves the output portion of the context budget. For reasoning API models, it includes hidden reasoning tokens as well as visible output. The temporal API example uses `2048` for section JSON extraction. Temporal storyline and question calls use `storyline_max_tokens` and `question_max_tokens`.
- `dtype` controls numerical precision. `bfloat16` is appropriate for the current Qwen2.5-VL model on Hopper GPUs; `half` uses FP16.
- `gpu_memory_utilization` should usually remain around `0.85-0.9` so CUDA and image-processing allocations have room.
- `tensor_parallel_size` is the number of GPUs used by one model instance. It must match the GPU allocation; the current 72B PBS job uses `4`.
- `temperature` controls local vLLM output variation. Use `0.0-0.2` when reliable JSON is more important than diversity. API requests use `api_temperature`, which must be `null` (or unset) for API models that do not accept the temperature parameter.
- `frames_dir` can be an absolute path on Gadi. Relative `sections_dir` and `output` paths resolve from the job working directory.

The `task` field identifies the configuration's benchmark category. The
top-level `--mode` selects the pipeline and prompt files:

| `task` | Prompt file | Purpose |
|--------|-------------|---------|
| 1 | `prompts/sparse_event_prompt.txt` | Sparse event localisation |
| 2 | `prompts/temporal_vlm.txt` | Temporal section extraction |

 Implemented. Configuration loading, frame partitioning, prompt resolution,
VLM calls, review/filtering, and question generation are active.

---

### `pipeline/prompts/temporal_vlm.txt`

**Purpose:** Section-level VLM prompt for the temporal pipeline. Asks the VLM to describe the section and record timestamped objects and events in chronological order.

**Output JSON schema:** `{summary, terrain, road_conditions, weather, lighting, notable_objects[], events[]}`



---

### `pipeline/prompts/temporal_llm_filter.txt`

**Purpose:** LLM prompt for merging duplicate or continuing events from a
chronologically sorted event list. It instructs the LLM to preserve the Python
computed order and not invent events.

**Output JSON schema:** `{events[]}` with `event_id`, `section`, `start_frame`,
`end_frame`, `description`, `uncertain`, and `notes` fields.

 Used by `TemporalPipeline.py` for windowed timeline cons

---

### `pipeline/prompts/temporal_storyline.txt`

**Purpose:** Converts the merged chronological timeline into plain prose for
human review. The prompt requires the LLM to mention events in order without
adding unsupported details.

**Output:** Plain text written to `storyline.txt`.



---

### `pipeline/prompts/temporal_question_gen.txt`

**Purpose:** Generates temporal-order benchmark questions from the
human-verified `timeline.json`.

**Question types:**

- `pairwise_order` — asks which of two events happened first.
- `sequence_order` — asks the model to order three to five events.
- `before_after` — asks what happened immediately before or after an event.

Questions include `event_ids`, `frame_evidence`, `options`, and
`answer_indices` so they can be reviewed and later graded programmatically.



---

### `pipeline/prompts/sparse_event_prompt.txt`

**Purpose:** VLM prompt for the sparse event localisation benchmark (task 1). Asks the VLM to filter for rare or noteworthy events — unusual occurrences, unexpected objects, sudden changes — while ignoring normal driving. Output is a JSON object with an `interesting_events` array.

**Output JSON schema:** `{interesting_events[]}`

Used by `SparseEventPipeline.py` during the `extract` stage.

---

### `pipeline/prompts/sparse_event_filter.txt`

**Purpose:** LLM prompt for reviewing a chronologically sorted list of sparse
detections. It instructs the LLM to merge duplicate observations of the same
rare event across section boundaries, remove detections that are not
noteworthy, preserve the Python computed order, and not invent events.

**Output JSON schema:** `{events[], removed[]}` with `event_id`, `section`,
`frame`, `event_description`, `why_interesting`, `terrain`, `uncertain`, and
`notes` fields.

Used by `SparseEventPipeline.py` for windowed review in the `review` stage.

---

### `pipeline/prompts/sparse_event_question_gen.txt`

**Purpose:** Generates sparse-event benchmark questions from the
human-verified `events.json`.

**Question types:**

- `existence` — asks whether a real, verified event occurred during the drive.
- `deceptive` — asks whether an invented plausible false event occurred; the
  answer is always no.
- `noteworthy` — asks which event was the noteworthy rare event that actually
  occurred. Options are four plausible events: exactly one real event and
  three invented false events; the answer is the real event.

Temporal-perception questions are intentionally not generated here; they
belong to the temporal-order benchmark. Questions include `event_ids`,
`frame_evidence`, `options`, and `answer_indices` so they can be reviewed and
later graded programmatically.
Every invented false event is repeated in the top-level `false_events` array
with its rationale and the `question_id` of the question that uses it for
human verification. False events are invented per question and never reused,
so each question has its own incorrect answers.


---

### `configs/`

**Purpose:** JSON config files for benchmark runs. The `task` field identifies
the benchmark category; `main.py --mode` selects the pipeline and prompt set.

**Files:**

| File | task | Purpose |
|------|------|---------|
| `sparse_events.json` | 1 | Sparse event localisation (uses `prompts/sparse_event_prompt.txt`, `sparse_event_filter.txt`, and `sparse_event_question_gen.txt`) |
| `temporal_events.json` | 2 | Temporal extraction, timeline, storyline, and question stages |

The current sparse configuration uses local `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8`
with `frames_per_section: 7`, `step: 8`, `skip: 2`, `image_max_size: null`,
`image_quality: 85`, `temperature: 0.4`, `max_tokens: 10000`,
`review_window: 50`, `question_max_tokens: 10000`, `max_model_len: 50000`, and
`tensor_parallel_size: 4`. The `backend` and `model` fields are the single
source of truth for what a run uses; the batch scripts only rewrite per-list
paths.
The current temporal example uses local `Qwen/Qwen2.5-VL-72B-Instruct` with
`frames_per_section: 5`, `step: 10`, `skip: 10`, `max_tokens: 3090`,
`merge_window: 50`, `storyline_max_tokens: 3000`, `question_max_tokens: 3000`,
`max_model_len: 50000`, and `tensor_parallel_size: 4`.




## Known Issues & Missing Pieces

1. **Backend-specific environment is required** — The supplied sparse and
   temporal configurations use local vLLM and require four GPUs plus access to
   the cached `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` and
   `Qwen/Qwen2.5-VL-72B-Instruct` models. Switching a configuration to
   `backend: "api"` requires a valid `OPENCODE_API_KEY` and network access.

2. **`vllm.pbs` runs the prototype** — The PBS script invokes `test_vlm.py`, not the mode dispatcher in `main.py`.

3. **Relative configuration paths** — `main.py` expects to be run from the repository root because the configured paths are relative.

4. **Human review remains necessary** — LLM output is used to draft sparse
   events, timelines, and questions. `events.json` and `timeline.json` should
   be checked against source frames before running the question stage, and
   `questions.json` (including the listed `false_events`) should be reviewed
   before benchmark use.

5. **No test suite** — No unit tests, integration tests, or test framework configuration exists.

6. **No CI/CD or containerization** — No Dockerfile, Makefile, GitHub Actions, or other automation.

---

## Summary of Implementation Status

| Component | Status |
|-----------|--------|
| Project design/spec | Reference context |
| Top-level mode dispatcher (`main.py`) | Sparse and temporal branches connected |
| Prototype (`test_vlm.py`) | Reference implementation |
| Frame parser (`frame_parser.py`) | Complete |
| Image preprocessor (`image_preprocessor.py`) | Complete |
| Timeframe converter (`timeframe_converter.py`) | Implemented standalone utility |
| AI parser (`pipeline/AIParser.py`) | Implemented local vLLM and OpenAI-compatible API wrapper |
| Sparse event pipeline (`pipeline/SparseEventPipeline.py`) | Extract, review, and question stages implemented |
| Temporal pipeline (`pipeline/TemporalPipeline.py`) | Extract, timeline, storyline, and question stages implemented |
| Temporal VLM prompt (`prompts/temporal_vlm.txt`) | Complete |
| Temporal LLM filter prompt (`prompts/temporal_llm_filter.txt`) | Implemented merge-only timeline prompt |
| Sparse event prompt (`prompts/sparse_event_prompt.txt`) | Complete |
| Sparse event filter prompt (`prompts/sparse_event_filter.txt`) | Implemented merge-and-filter review prompt |
| Sparse event question prompt (`prompts/sparse_event_question_gen.txt`) | Complete |
| Example configs (`configs/*.json`) | Complete |
| Legacy HPC job script (`vllm.pbs`) | Exists for the old prototype |
| Sparse-event HPC job script (`sparse_event.pbs`) | Local vLLM sparse-event wrapper; requests four GPUs for the 235B FP8 model |
| Temporal storyline prompt (`prompts/temporal_storyline.txt`) | Complete |
| Temporal question prompt (`prompts/temporal_question_gen.txt`) | Complete |
| Temporal PBS job script (`temporal_event.pbs`) | Local vLLM temporal wrapper; requests four GPUs for the 72B model |
| Forest run script (`forest_run.sh`) | API/local wrapper for the forest K-01 dataset; sparse and temporal runs in one file |
| Temporal question generation | Implemented; requires human-reviewed timeline |
| Sparse event question generation | Implemented; requires human-reviewed events list |
| Benchmark categories 2–4 | Not yet implemented |
| User Interface | Not yet implemented |
| Hybrid Search Module | Not yet implemented |
| Reporting frontend | Not yet implemented |
| Tests, CI/CD, Docker, Makefile | None exist |
