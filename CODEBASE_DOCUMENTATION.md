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

2. **AI Parser** (`pipeline/AIParser.py`) — Configurable local vLLM or OpenAI-compatible API wrapper. `call_llm` handles text prompts and `call_vlm` handles multiple images from a section directory.

3. **Sparse Event Pipeline** (`pipeline/SparseEventPipeline.py`) — `SparseEventPipeline` reads a JSON config file, creates `FrameParser` and `AIParser`, partitions frames, resolves the task prompt, queries the VLM, and writes results.

4. **Top-Level CLI Dispatcher** (`main.py`) — Accepts `--mode` values for sparse event, temporal, spatial, and counting benchmarks. `sparse` and `temporal` dispatch to their current pipelines; spatial and counting remain placeholders.

5. **Prototype/Test Script** (`test_vlm.py`) — Reference implementation using self-hosted vLLM with Qwen2-VL-7B. It reads video via Decord, chunks into 300-frame segments, and generates descriptions and change-detection QA drafts via few-shot prompting.

6. **Legacy HPC Job Script** (`vllm.pbs`) — PBS batch script for the old prototype on NCI's Gadi cluster.

7. **Sparse-Event HPC Job Script** (`sparse_event.pbs`) — PBS batch script for the current sparse-event CLI.

### What Is Not Yet Implemented

- **Temporal timeline filter** — section-level VLM extraction is implemented; LLM timeline consolidation is not yet implemented.
- The dedicated benchmarks for categories 2–4 (attribute perception, spatial reasoning, counting).
- `vllm.pbs` runs the prototype script rather than the top-level mode dispatcher.
- A **User Interface**, **Hybrid Search Module**, and **Reporting frontend** are not implemented.
- No test suite, CI/CD, Dockerfile, or Makefile.

---

## Full Directory Tree

```
episodic-memory-pipeline/
├── .gitignore
├── main.py
├── README.md
├── requirements.txt
├── test_vlm.py
├── vllm.pbs
├── sparse_event.pbs
├── CODEBASE_DOCUMENTATION.md
├── pipeline/
│   ├── ConfigLoader.py
│   ├── AIParser.py
│   ├── frame_parser.py
│   ├── SparseEventPipeline.py
│   ├── TemporalPipeline.py
│   └── prompts/
│       ├── temporal_vlm.txt
│       ├── temporal_llm_filter.txt
│       └── sparse_event_prompt.txt
├── boreas-*/                       (sample frame data)
├── configs/                        (pipeline configuration files)
│   ├── sparse_events.json
│   └── narrative_pass.json
```

---

## File-by-File Documentation

---

### Project Context

**Purpose:** The project context defines the goal of benchmarking VLM episodic memory with long-form egocentric dashcam footage across sparse-event, temporal-order, spatial-reasoning, and counting tasks.

**Status:** Reference context. Only the sparse-event mode currently has a connected pipeline.

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

The accepted values are `sparse`, `temporal`, `spatial`, and `counting`. `sparse` imports `pipeline/SparseEventPipeline.py`, while `temporal` imports `pipeline/TemporalPipeline.py` and uses `pipeline/prompts/temporal_vlm.txt`. Spatial and counting remain placeholders.

**Functions:**

| Function | Purpose |
|----------|---------|
| `main()` | Parses `--mode` and dispatches to the selected benchmark mode. |

**Status:** Active dispatcher. Sparse-event and section-level temporal branches are connected to pipelines.

---

### `README.md`

**Purpose:** Project readme with current CLI usage, supported modes, sparse-event workflow, and configuration details.

**Contents:** Documents `main.py`, `pipeline/frame_parser.py`, `pipeline/AIParser.py`, `pipeline/SparseEventPipeline.py`, and the current configuration files. The quick start uses `python main.py --mode sparse`.

**Status:** Current.

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

**Status:** Exists, but currently invokes `python main.py` without the required `--mode` argument.

---

### `sparse_event.pbs`

**Purpose:** PBS job script for running the current sparse-event pipeline on Gadi.

**Entry point:** Loads the Python and CUDA modules, activates the configured virtual environment, changes to the PBS working directory, and runs:

```bash
python3 main.py --mode sparse
```

Submit it from the repository root with `qsub sparse_event.pbs`.

**Resources:** 24 CPUs, 4 GPUs, 512 GB memory, and a five-hour walltime.

**Prerequisites:** Local mode requires `vllm`, the model must already be present in `/scratch/pg06/vm4618/huggingface_cache`, and four GPUs must be available. The job explicitly runs Hugging Face in offline mode because compute nodes cannot access the network. API mode requires `openai`, the configured API key environment variable, and approved outbound HTTPS access instead.

---

## `pipeline/` — Sparse Event Localisation Sub-Pipeline

This directory focuses on **Benchmark Category 1: Sparse Event Localisation** and serves as the foundation for the planned temporal/episodic memory pipeline.

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

**Status:** Complete.

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
| `call_llm(prompt)` → `str` | Sends a text prompt to the selected backend and returns generated text. |
| `call_vlm(prompt, folder_path)` → `str` | Loads all supported image frames from `folder_path`, sorts them by numeric filename, sends them together to the selected backend, and returns generated text. |

Configuration loading belongs to the pipeline classes, which read the JSON file and pass the complete dictionary into the `AIParser` constructor. In API mode, the API key is read from the environment variable named by `api_key_env`; it is not stored in configuration files.

**Output structure (per section):**
```
<output_dir>/all_results.json                (aggregated single-file view)
<output_dir>/section_0000_output/result.json (per-section persistent artifact)
<output_dir>/section_0001_output/result.json
...
```

**Status:** Implemented for local vLLM and OpenAI-compatible API backends.

---

### `pipeline/TemporalPipeline.py`

**Purpose:** Runs the first temporal pipeline stage. It loads the temporal
configuration, partitions frames into sections, sends each section to the VLM
using `temporal_vlm.txt`, and saves the raw section responses.

**Output:** Writes one `result.json` per section and an aggregated
`all_results.json` under the configured `output` directory.

**Status:** Section-level VLM extraction implemented. LLM timeline filtering is
the next stage.

---

### `pipeline/ConfigLoader.py`

**Purpose:** Shared configuration component used by pipeline entry points.
`ConfigLoader(config_path).load()` validates that the JSON file exists, parses
it, verifies that it contains an object, and returns the configuration
dictionary.

---

### `pipeline/SparseEventPipeline.py`

**Purpose:** Defines `SparseEventPipeline`, which loads configuration through
`ConfigLoader`, creates `FrameParser` and `AIParser`, splits frames into
sections, and resolves the prompt selected by the configuration.

The class is invoked by `main.py` when the user selects
`--mode sparse`.

**Config file schema:**

| Key | Description | Default |
|-----|-------------|---------|
| `task` | Benchmark category: 1 = sparse events, 2 = temporal/narrative | (required) |
| `frames_dir` | Directory of frame images | (required) |
| `sections_dir` | Intermediate section output | `"./sections"` |
| `output` | Intended final VLM results directory | (required) |
| `frames_per_section` | Max frames per section | `100` |
| `step` | Keep every Nth frame | `1` |
| `model` | Local vLLM model name or path | (required) |
| `temperature` | Sampling temperature | `0.2` |
| `max_tokens` | Maximum generated tokens | `100` |
| `enforce_eager` | Disable CUDA graph capture | `true` |
| `dtype` | Model data type | `"half"` |
| `max_model_len` | Maximum model context length | `4096` |
| `gpu_memory_utilization` | Fraction of GPU memory available to vLLM | `0.9` |
| `tensor_parallel_size` | Number of GPUs used to shard the model | `1` |

**Configuration selection guide:**

- `frames_per_section` controls how many images are sent in one VLM request. Lower values reduce memory use; start around `5-10` for high-resolution frames.
- `step` keeps every Nth frame after sorting. Increase it to reduce compute, at the cost of temporal detail.
- `max_model_len` is the total token budget for the prompt, visual image tokens, and generated output. It is not a duration or frame count. Start at `4096`; increase to `8192` if requests are too long, or reduce it and/or the section size if GPU memory is exhausted.
- `max_tokens` reserves the output portion of the context budget. Use `100-256` for initial JSON event extraction and increase it if responses are truncated.
- `dtype` controls numerical precision. `bfloat16` is appropriate for the current Qwen2.5-VL model on Hopper GPUs; `half` uses FP16.
- `gpu_memory_utilization` should usually remain around `0.85-0.9` so CUDA and image-processing allocations have room.
- `tensor_parallel_size` is the number of GPUs used by one model instance. It must match the GPU allocation; the current 72B PBS job uses `4`.
- `temperature` controls output variation. Use `0.0-0.2` when reliable JSON is more important than diversity.
- `frames_dir` can be an absolute path on Gadi. Relative `sections_dir` and `output` paths resolve from the job working directory.

**Task → prompt mapping:**

| `task` | Prompt file | Purpose |
|--------|-------------|---------|
| 1 | `prompts/sparse_event_prompt.txt` | Sparse event localisation |
| 2 | `prompts/temporal_vlm.txt` | Temporal section description |

**Status:** Implemented. Configuration loading, frame partitioning, prompt
resolution, VLM calls, and output writing are active.

---

### `pipeline/prompts/temporal_vlm.txt`

**Purpose:** Section-level VLM prompt for the temporal pipeline. Asks the VLM to describe the section and record timestamped objects and events in chronological order.

**Output JSON schema:** `{summary, terrain, road_conditions, weather, lighting, notable_objects[], events[]}`

**Status:** Complete.

---

### `pipeline/prompts/temporal_llm_filter.txt`

**Purpose:** Planned LLM prompt for merging section-level temporal observations into a single ordered timeline.

**Status:** Prompt draft; not used by the current pipeline stage.

---

### `pipeline/prompts/sparse_event_prompt.txt`

**Purpose:** VLM prompt for the sparse event localisation benchmark (task 1). Asks the VLM to filter for rare or noteworthy events — unusual occurrences, unexpected objects, sudden changes — while ignoring normal driving. Output is a JSON object with an `interesting_events` array.

**Output JSON schema:** `{interesting_events[]}`

**Status:** Complete.

---

### `configs/`

**Purpose:** JSON config files for benchmark runs. The `task` field selects the benchmark category and corresponding prompt. `main.py --mode sparse` uses `sparse_events.json`.

**Files:**

| File | task | Purpose |
|------|------|---------|
| `sparse_events.json` | 1 | Sparse event localisation (uses `prompts/sparse_event_prompt.txt`) |
| `narrative_pass.json` | 2 | Temporal section extraction (uses `prompts/temporal_vlm.txt`) |

**Status:** Complete.

---

## Architecture: Local VLM Implementations

The codebase contains a prototype, an active sparse-event pipeline wrapper, and a section-level temporal pipeline:

| Approach | File | Model | Interface |
|----------|------|-------|-----------|
| **Prototype** | `test_vlm.py` | Qwen2-VL-7B (vision + text) | Direct `vllm.LLM()` class, offline HF cache |
| **Pipeline wrapper** | `pipeline/AIParser.py` | Configured local vision-language model | Direct `vllm.LLM()` class |
| **Temporal section pipeline** | `pipeline/TemporalPipeline.py` | Configured local vision-language model | Section VLM calls with persisted JSON responses |

The prototype (`test_vlm.py`) reads video directly via Decord. The pipeline wrapper reads pre-extracted frame folders and sends all frames in each section to a local vLLM model.

---

## Known Issues & Missing Pieces

1. **GPU/model environment is required** — The local environment does not include vLLM, and the pipeline requires four GPUs plus access to the cached `Qwen/Qwen2.5-VL-72B-Instruct` model.

2. **`vllm.pbs` runs the prototype** — The PBS script invokes `test_vlm.py`, not the mode dispatcher in `main.py`.

3. **Relative configuration paths** — `main.py` expects to be run from the repository root because the configured paths are relative.

4. **No test suite** — No unit tests, integration tests, or test framework configuration exists.

5. **No CI/CD or containerization** — No Dockerfile, Makefile, GitHub Actions, or other automation.

---

## Summary of Implementation Status

| Component | Status |
|-----------|--------|
| Project design/spec | Reference context |
| Top-level mode dispatcher (`main.py`) | Sparse and temporal branches connected |
| Prototype (`test_vlm.py`) | Reference implementation |
| Frame parser (`frame_parser.py`) | Complete |
| AI parser (`pipeline/AIParser.py`) | Implemented local vLLM wrapper |
| Sparse event pipeline (`pipeline/SparseEventPipeline.py`) | Implemented sparse-event workflow |
| Temporal pipeline (`pipeline/TemporalPipeline.py`) | Section VLM extraction implemented; timeline filter pending |
| Temporal VLM prompt (`prompts/temporal_vlm.txt`) | Complete |
| Temporal LLM filter prompt (`prompts/temporal_llm_filter.txt`) | Draft |
| Sparse event prompt (`prompts/sparse_event_prompt.txt`) | Complete |
| Example configs (`configs/*.json`) | Complete |
| Legacy HPC job script (`vllm.pbs`) | Exists for the old prototype |
| Sparse-event HPC job script (`sparse_event.pbs`) | Current PBS wrapper; requests four GPUs for the 72B model |
| Temporal timeline builder and question generation | Not yet implemented |
| Benchmark categories 2–4 | Not yet implemented |
| User Interface | Not yet implemented |
| Hybrid Search Module | Not yet implemented |
| Reporting frontend | Not yet implemented |
| Tests, CI/CD, Docker, Makefile | None exist |
