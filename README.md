# Episodic Memory Pipeline

Benchmarking pipeline for assessing Vision Language Models' (VLMs) episodic memory capabilities using long-form egocentric dashcam frames.

## Current Status

The top-level runner accepts four benchmark modes. Only `sparse_event` is currently connected to a pipeline; the other modes are recognized but report that they are not implemented yet.

| Mode | Status |
|------|--------|
| `sparse_event` | Connected to `SparseEventPipeline` |
| `temporal` | Not implemented |
| `spatial` | Not implemented |
| `counting` | Not implemented |

The sparse-event branch is connected, but the pipeline is not currently runnable end-to-end: the `AIParser` constructor and VLM methods are commented out. The intended frame partitioning, prompt resolution, VLM query, and result-writing flow is documented below.

## Setup

Install the dependencies listed in `requirements.txt`. The current mode dispatcher does not require a `.env` file.

## Run A Benchmark Mode

Run commands from the repository root because configuration paths are relative to the current working directory:

```bash
python main.py --mode sparse_event
```

The sparse-event mode uses `configs/sparse_events.json` by default. The other accepted modes currently print a not-implemented message:

```bash
python main.py --mode temporal
python main.py --mode spatial
python main.py --mode counting
```

## Structure

| Path | Purpose |
|------|---------|
| `main.py` | CLI dispatcher for the benchmark modes |
| `pipeline/SparseEventPipeline.py` | Sparse-event pipeline class |
| `pipeline/ConfigLoader.py` | Shared JSON configuration loader |
| `pipeline/frame_parser.py` | Splits numerically named frame images into sections |
| `pipeline/AIParser.py` | Scaffold for local VLM/LLM calls |
| `pipeline/prompts/*.txt` | Prompt files for benchmark tasks |
| `configs/*.json` | Pipeline configuration files |
| `test_vlm.py` | VLM prototype/test script |
| `vllm.pbs` | PBS job script for running the prototype on Gadi |

## Sparse-Event Pipeline

The pipeline reads a JSON configuration, creates a `FrameParser`, and prepares an `AIParser` using the configured model settings. Its intended flow is:

1. Load `configs/sparse_events.json`.
2. Sort and sample the input frames.
3. Copy or move frames into section directories.
4. Query the VLM for each section using the task prompt.
5. Write per-section and aggregated results.

At present, steps 1-3 and prompt resolution are scaffolded, but execution stops while constructing `AIParser` until its implementation is restored. Steps 4-5 remain commented scaffolding.

## Configuration

The sparse-event configuration is stored in `configs/sparse_events.json`:

```json
{
    "task": 1,
    "frames_dir": "./boreas-2024-12-04-15-19/camera",
    "sections_dir": "./sections",
    "output": "./events",
    "frames_per_section": 10,
    "step": 12,
    "model": "Qwen/Qwen2-VL-7B-Instruct",
    "temperature": 0.2,
    "max_tokens": 100,
    "enforce_eager": true,
    "dtype": "half",
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.9
}
```

The `task` value selects the prompt:

| Task | Prompt | Purpose |
|------|--------|---------|
| `1` | `pipeline/prompts/sparse_event_prompt.txt` | Sparse event localisation |
| `2` | `pipeline/prompts/temporal.txt` | Full scene description |

## Frame Sections

`FrameParser` accepts `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp` files. It sorts frames by their numeric filename, keeps every `step`-th frame, and groups them into directories such as:

```text
sections/
    section_0000/
    section_0001/
```

Set `move` to `true` in the configuration to move frames instead of copying them. It defaults to `false` when omitted.

## Intended Output

When VLM querying and result writing are enabled, the intended output is:

```text
<output>/
    all_results.json
    section_0000_output/result.json
    section_0001_output/result.json
```

The current sparse-event run does not yet write these result files. It must first have the `AIParser` implementation and VLM loop enabled.

## Full Documentation

See [CODEBASE_DOCUMENTATION.md](CODEBASE_DOCUMENTATION.md) for a detailed breakdown of the codebase and implementation status.
