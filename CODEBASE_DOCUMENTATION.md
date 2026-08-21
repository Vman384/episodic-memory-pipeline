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

1. **Frame Parser** (`pipeline/frame_parser.py`) — Splits a directory of numerically named frame images into VLM-sized section folders. Supports configurable chunk size, frame step sampling, and copy/move semantics.

2. **Timeframe Converter** (`pipeline/timeframe_converter.py`) — Converts temporal event `start_frame` and `end_frame` values into elapsed video seconds relative to the earliest frame in the source directory.

3. **AI Parser** (`pipeline/AIParser.py`) — Configurable local vLLM or OpenAI-compatible API wrapper. `call_llm` handles text prompts and `call_vlm` handles multiple images from a section directory.

4. **Sparse Event Pipeline** (`pipeline/SparseEventPipeline.py`) — `SparseEventPipeline` reads a JSON config file, creates `FrameParser` and `AIParser`, partitions frames, resolves the task prompt, queries the VLM, and writes results.

5. **Temporal Pipeline** (`pipeline/TemporalPipeline.py`) — Provides resumable section extraction, timestamp-sorted timeline construction, windowed LLM merging, storyline generation, and question generation from a human-reviewed timeline.

6. **Top-Level CLI Dispatcher** (`main.py`) — Accepts `--mode` values for sparse event, temporal, spatial, and counting benchmarks. The temporal branch also accepts `--stage extract|timeline|questions`; spatial and counting remain placeholders.

7. **Prototype/Test Script** (`test_vlm.py`) — Reference implementation using self-hosted vLLM with Qwen2-VL-7B. It reads video via Decord, chunks into 300-frame segments, and generates descriptions and change-detection QA drafts via few-shot prompting.

8. **Legacy HPC Job Script** (`vllm.pbs`) — PBS batch script for the old prototype on NCI's Gadi cluster.

9. **Sparse-Event HPC Job Script** (`sparse_event.pbs`) — PBS batch script for the current sparse-event CLI.

### What Is Not Yet Implemented

- The dedicated benchmarks for categories 2–4 (attribute perception, spatial reasoning, counting).
- `temporal_event.pbs` runs the API-backed temporal configuration and forwards exported environment variables to the PBS job.
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
├── CODEBASE_DOCUMENTATION.md
├── pipeline/
│   ├── ConfigLoader.py
│   ├── AIParser.py
│   ├── frame_parser.py
│   ├── timeframe_converter.py
│   ├── SparseEventPipeline.py
│   ├── TemporalPipeline.py
│   └── prompts/
│       ├── temporal_vlm.txt
│       ├── temporal_llm_filter.txt
│       ├── temporal_storyline.txt
│       ├── temporal_question_gen.txt
│       └── sparse_event_prompt.txt
├── boreas-*/                       (sample frame data)
├── configs/                        (pipeline configuration files)
│   ├── sparse_events.json
│   └── .temporal_events.json
```

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

The accepted values are `sparse`, `temporal`, `spatial`, and `counting`. `sparse` imports `pipeline/SparseEventPipeline.py`, while `temporal` imports `pipeline/TemporalPipeline.py` and loads four temporal prompts. Spatial and counting remain placeholders.

The optional `--stage` argument applies to `temporal` mode:

```bash
python main.py --mode temporal --stage extract
python main.py --mode temporal --stage timeline
python main.py --mode temporal --stage questions
```

With no `--stage`, temporal mode runs `extract` followed by `timeline`. The
`questions` stage is intentionally separate so a human can review and correct
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

**Entry point:** Loads the Python and CUDA modules, activates the configured virtual environment, changes to the PBS working directory, and runs:

```bash
python3 main.py --mode sparse
```

Submit it from the repository root with `qsub sparse_event.pbs`.

**Resources:** 24 CPUs, 4 GPUs, 1024 GB memory, and a five-hour walltime.

**Prerequisites:** Requires `vllm`, four GPUs, and the model already present in `/scratch/pg06/FYP2026S1_3473/huggingface_cache`. This is the local/offline job; API runs should use `temporal_event.pbs` instead.

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

**Current status:** The script is configured for the API-backed temporal
configuration. It requests CPU and memory only, forwards the submission
environment with `#PBS -V`, and relies on `OPENCODE_API_KEY` or `.env` for
authentication. It does not load CUDA or a local Hugging Face model.

---

## `pipeline/` — Pipeline Components

This directory contains the shared helpers plus the sparse-event and
temporal-order benchmark pipelines.

**Workflow:** `python main.py --mode sparse` → `SparseEventPipeline` → `ConfigLoader` → `FrameParser` (split frames into sections) → prompt resolution → `AIParser.call_vlm(prompt, section_dir)` → result persistence.

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



---

### `pipeline/timeframe_converter.py`

**Purpose:** Converts numeric frame timestamps from temporal event outputs into
elapsed seconds in the overall video. The earliest numeric frame in the source
directory is treated as time zero, and timestamps are interpreted as
microseconds.

**Class: `TimeframeConverter`**

| Method | Purpose |
|--------|---------|
| `frame_to_seconds(frame)` → `float` | Convert one frame filename or timestamp to elapsed seconds. |
| `timeframe_to_seconds(start_frame, end_frame)` → `dict` | Return `start_seconds` and `end_seconds` for one temporal event. |

The class is currently a standalone utility; timeline output is not modified
automatically.

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

The source frame directory must be available and contain numeric timestamp
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
| `temperature` | Sampling temperature |
| `reasoning_effort` | Optional Responses API reasoning effort, such as `low` or `none` |
| `max_tokens` | Maximum generated output tokens |
| `enforce_eager` | vLLM CUDA graph setting used in local mode |
| `dtype` | vLLM model data type used in local mode |
| `max_model_len` | vLLM maximum model context length |
| `gpu_memory_utilization` | vLLM GPU memory fraction |
| `tensor_parallel_size` | Number of GPUs used by vLLM |
| `api_base_url` | OpenAI-compatible API base URL or full `/responses` endpoint |
| `api_key_env` | Environment variable containing the API key |

**Public methods:**

| Method | Purpose |
|--------|---------|
| `call_llm(prompt, max_tokens=None)` → `str` | Sends a text prompt to the selected backend and returns generated text. An optional per-call token limit overrides the configured default. |
| `call_vlm(prompt, folder_path)` → `str` | Loads all supported image frames from `folder_path`, sorts them by numeric filename, includes an in-memory ordered filename manifest in the prompt, sends them together to the selected backend, and returns generated text. |

Configuration loading belongs to the pipeline classes, which read the JSON file and pass the complete dictionary into the `AIParser` constructor. In API mode, the API key is read from the environment variable named by `api_key_env`; it is not stored in configuration files.

**Output structure (per section):**
```
<output_dir>/all_results.json                (aggregated single-file view)
<output_dir>/section_0000_output/result.json (per-section persistent artifact)
<output_dir>/section_0001_output/result.json
...
```

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
| `extract` | Creates frame sections, queries the VLM with `temporal_vlm.txt`, writes one result per section, and skips existing results for resumability. |
| `timeline` | Loads persisted section results, parses their `events` arrays, sorts them by numeric `start_frame`, merges events in LLM windows, and writes `timeline.json` and `storyline.txt`. |
| `questions` | Loads `timeline.json`, intended to be human-reviewed first, and writes generated temporal questions to `questions.json`. |

When no stage is supplied, the pipeline runs `extract` followed by `timeline`.
The question stage is never included in the default run so that human review can
occur between timeline creation and question generation.

**Timeline behavior:** Event ordering is determined in Python from the numeric
timestamp in each frame filename. The LLM is instructed only to merge duplicate
or continuing observations, preserve order, and avoid inventing events. The
`merge_window` configuration controls how many sorted events are sent in one
merge request. Invalid model JSON is handled with a warning and the unmerged
events are retained for review.

**Output:**

```text
<output>/
    all_results.json
    section_0000_output/result.json
    section_0001_output/result.json
    timeline.json
    storyline.txt
    questions.json
```

`questions.json` is produced only by the `questions` stage. Each generated
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
directories or splits frames into sections, queries each section with the prompt
supplied by `main.py`, and persists the results.

The class is invoked by `main.py` when the user selects
`--mode sparse`.

**Shared config file schema:**

| Key | Description | Default |
|-----|-------------|---------|
| `task` | Benchmark category metadata: `1` = sparse events, `2` = temporal/narrative | (required) |
| `frames_dir` | Directory of frame images | (required) |
| `sections_dir` | Intermediate section output | `"./sections"` |
| `output` | Intended final VLM results directory | (required) |
| `frames_per_section` | Max frames per section | `100` |
| `step` | Keep every Nth frame | `1` |
| `model` | Local vLLM model name/path or API model identifier | (required) |
| `backend` | `local` for vLLM or `api` for the Responses API | `local` |
| `temperature` | Sampling temperature; set to `null` to omit it from API requests | (optional) |
| `reasoning_effort` | Responses API reasoning effort, such as `low` or `none` | (optional) |
| `max_tokens` | Default maximum generated output tokens. Used by VLM calls and LLM calls without an override | `100` |
| `merge_window` | Maximum sorted events supplied to one temporal merge call | `50` |
| `storyline_max_tokens` | Maximum tokens for the temporal storyline call | (optional) |
| `question_max_tokens` | Maximum tokens for temporal question generation | (optional) |
| `enforce_eager` | Disable CUDA graph capture | `true` |
| `dtype` | Model data type | `"half"` |
| `max_model_len` | Maximum model context length | `4096` |
| `gpu_memory_utilization` | Fraction of GPU memory available to vLLM | `0.9` |
| `tensor_parallel_size` | Number of GPUs used to shard the model | `1` |
| `api_base_url` | API base URL or full `/responses` endpoint | (optional) |
| `api_key_env` | Environment variable containing the API key | (optional) |

**Configuration selection guide:**

- `frames_per_section` controls how many images are sent in one VLM request. Lower values reduce memory use; start around `5-10` for high-resolution frames.
- `step` keeps every Nth frame after sorting. Increase it to reduce compute, at the cost of temporal detail.
- `max_model_len` is the total token budget for the prompt, visual image tokens, and generated output. It is not a duration or frame count. Start at `4096`; increase to `8192` if requests are too long, or reduce it and/or the section size if GPU memory is exhausted.
- `max_tokens` reserves the output portion of the context budget. For reasoning API models, it includes hidden reasoning tokens as well as visible output. The temporal API example uses `2048` for section JSON extraction. Temporal storyline and question calls use `storyline_max_tokens` and `question_max_tokens`.
- `dtype` controls numerical precision. `bfloat16` is appropriate for the current Qwen2.5-VL model on Hopper GPUs; `half` uses FP16.
- `gpu_memory_utilization` should usually remain around `0.85-0.9` so CUDA and image-processing allocations have room.
- `tensor_parallel_size` is the number of GPUs used by one model instance. It must match the GPU allocation; the current 72B PBS job uses `4`.
- `temperature` controls output variation. Use `0.0-0.2` when reliable JSON is more important than diversity.
- `frames_dir` can be an absolute path on Gadi. Relative `sections_dir` and `output` paths resolve from the job working directory.

The `task` field identifies the configuration's benchmark category. The
top-level `--mode` selects the pipeline and prompt files:

| `task` | Prompt file | Purpose |
|--------|-------------|---------|
| 1 | `prompts/sparse_event_prompt.txt` | Sparse event localisation |
| 2 | `prompts/temporal_vlm.txt` | Temporal section extraction |

 Implemented. Configuration loading, frame partitioni
resolution, VLM calls, timeline construction, and output writing are active.

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



---

### `configs/`

**Purpose:** JSON config files for benchmark runs. The `task` field identifies
the benchmark category; `main.py --mode` selects the pipeline and prompt set.

**Files:**

| File | task | Purpose |
|------|------|---------|
| `sparse_events.json` | 1 | Sparse event localisation (uses `prompts/sparse_event_prompt.txt`) |
| `temporal_events.json` | 2 | Temporal extraction, timeline, storyline, and question stages |

The current temporal example uses `frames_per_section: 10`, `step: 15`,
no API `temperature`, `reasoning_effort: low`, `max_tokens: 2048`,
`merge_window: 50`, `storyline_max_tokens: 1024`,
`question_max_tokens: 2048`, `max_model_len: 8192`, and
`tensor_parallel_size: 4`.




## Known Issues & Missing Pieces

1. **Backend-specific environment is required** — The sparse configuration uses local vLLM and requires four GPUs plus access to the cached `Qwen/Qwen2.5-VL-72B-Instruct` model. The temporal configuration uses the OpenCode Go API and requires a valid `OPENCODE_API_KEY`, network access, and an image-capable model.

2. **`vllm.pbs` runs the prototype** — The PBS script invokes `test_vlm.py`, not the mode dispatcher in `main.py`.

3. **Relative configuration paths** — `main.py` expects to be run from the repository root because the configured paths are relative.

4. **Human review remains necessary** — LLM output is used to draft timelines and questions. `timeline.json` should be checked against source frames before running the question stage, and `questions.json` should be reviewed before benchmark use.

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
| Timeframe converter (`timeframe_converter.py`) | Implemented standalone utility |
| AI parser (`pipeline/AIParser.py`) | Implemented local vLLM and OpenAI-compatible API wrapper |
| Sparse event pipeline (`pipeline/SparseEventPipeline.py`) | Implemented sparse-event workflow |
| Temporal pipeline (`pipeline/TemporalPipeline.py`) | Extract, timeline, storyline, and question stages implemented |
| Temporal VLM prompt (`prompts/temporal_vlm.txt`) | Complete |
| Temporal LLM filter prompt (`prompts/temporal_llm_filter.txt`) | Implemented merge-only timeline prompt |
| Sparse event prompt (`prompts/sparse_event_prompt.txt`) | Complete |
| Example configs (`configs/*.json`) | Complete |
| Legacy HPC job script (`vllm.pbs`) | Exists for the old prototype |
| Sparse-event HPC job script (`sparse_event.pbs`) | Current PBS wrapper; requests four GPUs for the 72B model |
| Temporal storyline prompt (`prompts/temporal_storyline.txt`) | Complete |
| Temporal question prompt (`prompts/temporal_question_gen.txt`) | Complete |
| Temporal PBS job script (`temporal_event.pbs`) | API-backed temporal wrapper; no GPU required |
| Temporal question generation | Implemented; requires human-reviewed timeline |
| Benchmark categories 2–4 | Not yet implemented |
| User Interface | Not yet implemented |
| Hybrid Search Module | Not yet implemented |
| Reporting frontend | Not yet implemented |
| Tests, CI/CD, Docker, Makefile | None exist |
