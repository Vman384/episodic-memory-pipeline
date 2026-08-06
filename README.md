# Episodic Memory Pipeline

Benchmarking pipeline for assessing Vision Language Models' (VLMs) episodic memory capabilities using long-form egocentric dashcam frames.

## Current Status

The top-level runner accepts four benchmark modes. Only `sparse` is currently connected to a pipeline; the other modes are recognized but report that they are not implemented yet.

| Mode | Status |
|------|--------|
| `sparse` | Connected to `SparseEventPipeline` |
| `temporal` | Not implemented |
| `spatial` | Not implemented |
| `counting` | Not implemented |

The sparse-event branch is connected and the VLM parser implementation is present. It requires a Gadi environment with `vllm`, the Qwen2.5-VL-72B-Instruct model available in the configured Hugging Face cache, and input frames configured in `configs/sparse_events.json`.

## Setup

Install the dependencies listed in `requirements.txt`. The current mode dispatcher does not require a `.env` file.

## Run A Benchmark Mode

Run commands from the repository root because configuration paths are relative to the current working directory:

```bash
python main.py --mode sparse
```

The sparse mode uses `configs/sparse_events.json` by default. The other accepted modes currently print a not-implemented message:

```bash
python main.py --mode temporal
python main.py --mode spatial
python main.py --mode counting
```

On Gadi, submit the current sparse-event job with:

```bash
qsub sparse_event.pbs
```

The PBS script sources `/scratch/pg06/vm4618/envs/vllm_env/bin/activate` and uses `/scratch/pg06/vm4618/huggingface_cache` in offline mode. The 72B model must already be present in that cache.

The 72B model is sharded across four GPUs using `tensor_parallel_size: 4`.
There is no official Qwen2.7 VLM model name; `Qwen2.5-VL-72B-Instruct` is the
72B Qwen vision-language model used here.

## Structure

| Path | Purpose |
|------|---------|
| `main.py` | CLI dispatcher for the benchmark modes |
| `pipeline/SparseEventPipeline.py` | Sparse-event pipeline class |
| `pipeline/ConfigLoader.py` | Shared JSON configuration loader |
| `pipeline/frame_parser.py` | Splits numerically named frame images into sections |
| `pipeline/AIParser.py` | Local vLLM wrapper for text and multi-image calls |
| `pipeline/prompts/*.txt` | Prompt files for benchmark tasks |
| `configs/*.json` | Pipeline configuration files |
| `test_vlm.py` | VLM prototype/test script |
| `vllm.pbs` | Legacy PBS job script for the prototype |
| `sparse_event.pbs` | PBS job script for the current sparse-event pipeline |

## Sparse-Event Pipeline

The pipeline reads a JSON configuration, creates a `FrameParser`, and prepares an `AIParser` using the configured model settings. Its intended flow is:

1. Load `configs/sparse_events.json`.
2. Sort and sample the input frames.
3. Copy or move frames into section directories.
4. Query the VLM for each section using the task prompt.
5. Write per-section and aggregated results.

The pipeline loads the model once, partitions frames, sends each section to the VLM, and writes one JSON result per section plus `all_results.json`.

## Configuration

The sparse-event configuration is stored in `configs/sparse_events.json`:

```json
{
    "task": 1,
    "frames_dir": "/scratch/pg06/FYP2026S1_3473/boreas_dataset/boreas-2024-12-04-11-56/camera",
    "sections_dir": "./sections",
    "output": "./events",
    "frames_per_section": 10,
    "step": 12,
    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
    "temperature": 0.2,
    "max_tokens": 100,
    "enforce_eager": true,
    "dtype": "bfloat16",
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.9,
    "tensor_parallel_size": 4
}
```

The `task` value selects the prompt:

| Task | Prompt | Purpose |
|------|--------|---------|
| `1` | `pipeline/prompts/sparse_event_prompt.txt` | Sparse event localisation |
| `2` | `pipeline/prompts/temporal.txt` | Full scene description |

## Configuration Guide

### `max_model_len`

`max_model_len` is the maximum number of tokens in one VLM request, including:

```text
text prompt tokens + visual tokens from all images + generated output tokens
```

It is not the video length and it does not directly represent the number of
frames. More frames and higher-resolution images consume more visual tokens.
The model's theoretical context limit is an upper bound, not a value that will
necessarily fit in GPU memory. A larger value also reserves more KV-cache
memory. The current Qwen2.5-VL model configuration advertises a much larger
maximum position length, but that does not mean the full value is practical for
10 high-resolution images on one request.

Use this process when choosing a value:

1. Start with `4096` for a small section such as the current 10-frame setup.
2. If vLLM reports that the prompt is too long, increase it to `8192` or
   higher, provided the model and GPUs support it.
3. If vLLM runs out of memory, reduce `frames_per_section`, increase `step`,
   reduce image resolution, or lower `max_model_len`.
4. Keep `max_tokens` within the context budget. For JSON event answers,
   `100-256` output tokens is usually a reasonable starting range.

### Other Parameters

| Parameter | Selection guidance |
|-----------|--------------------|
| `task` | `1` selects sparse-event localisation; `2` selects the narrative prompt. |
| `frames_dir` | Use an absolute path on Gadi when the data is outside the repository. |
| `sections_dir` | Intermediate frame sections. Use scratch storage for large runs. |
| `output` | Final JSON results directory. Use scratch storage for large runs. |
| `frames_per_section` | Images sent in one VLM request. Lower values reduce memory; `5-10` is a useful starting range. |
| `step` | Keeps every Nth sorted frame. Higher values reduce compute but lose temporal detail. |
| `model` | Must be a VLM compatible with vLLM. The current model is `Qwen/Qwen2.5-VL-72B-Instruct`. |
| `temperature` | Use `0.0-0.2` for stable JSON; higher values produce more variation. |
| `max_tokens` | Maximum generated output tokens. Increase if responses are truncated. |
| `enforce_eager` | `true` is usually safer; `false` may improve speed but can require more memory. |
| `dtype` | Use `bfloat16` on Hopper GPUs for the current Qwen model; use `half` when FP16 is required. |
| `gpu_memory_utilization` | Usually `0.85-0.9`. Leave some memory for CUDA and image processing. |
| `tensor_parallel_size` | Number of GPUs used by one model instance. It must match the PBS GPU allocation; current value is `4`. |
| `move` | Set `true` only if input frames may be moved instead of copied. Defaults to `false`. |

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

The sparse-event run writes these result files after the model is loaded and each section is processed.

## Full Documentation

See [CODEBASE_DOCUMENTATION.md](CODEBASE_DOCUMENTATION.md) for a detailed breakdown of the codebase and implementation status.
