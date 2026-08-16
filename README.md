# Episodic Memory Pipeline

Benchmarking pipeline for assessing Vision Language Models' (VLMs) episodic memory capabilities using long-form egocentric dashcam frames.

## Current Status

The top-level runner accepts four benchmark modes. `sparse` and `temporal` are connected to pipelines; `spatial` and `counting` are recognized but not implemented yet.

| Mode | Status |
|------|--------|
| `sparse` | Connected to `SparseEventPipeline` |
| `temporal` | Resumable extraction, timeline construction, storyline, and question generation implemented |
| `spatial` | Not implemented |
| `counting` | Not implemented |

Both connected branches support local vLLM and OpenAI-compatible API inference. The supplied sparse-event configuration uses local vLLM; the supplied temporal configuration uses the OpenCode Go Responses API with `gpt-5.6-luna` and does not require a local VLM or GPUs.

## Setup

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The current mode dispatcher does not require a `.env` file. API mode requires
the environment variable named by `api_key_env` in the selected configuration;
the supplied temporal configuration uses `OPENCODE_API_KEY`.

## Run A Benchmark Mode

Run commands from the repository root because configuration paths are relative to the current working directory:

```bash
python main.py --mode sparse
```

The sparse mode uses `configs/sparse_events.json` by default. The temporal mode uses `configs/temporal_events.json` and runs extraction followed by timeline construction when no stage is specified. The temporal stages can be run independently:

```bash
python main.py --mode temporal --stage extract
python main.py --mode temporal --stage timeline
python main.py --mode temporal --stage questions
```

The expected temporal review workflow is:

1. Run `--stage extract` to create section summaries. Existing section results are reused.
2. Run `--stage timeline` to sort events by frame timestamp, merge duplicate or continuing events, and write `timeline.json` and `storyline.txt`.
3. Review and edit `timeline.json` against the referenced frames.
4. Run `--stage questions` to generate temporal questions from the reviewed timeline.
5. Review `questions.json` before using it as benchmark data.

The other accepted modes currently print a not-implemented message:

```bash
python main.py --mode spatial
python main.py --mode counting
```

On Gadi, submit the current sparse-event job with:

```bash
qsub sparse_event.pbs
```

The sparse-event PBS script sources `/scratch/pg06/vm4618/envs/vllm_env/bin/activate`, requests four GPUs, and runs local vLLM with `HF_HOME=/scratch/pg06/FYP2026S1_3473/huggingface_cache` in offline mode; the model must already be present in that cache.

The 72B model is sharded across four GPUs using `tensor_parallel_size: 4`.
There is no official Qwen2.7 VLM model name; `Qwen2.5-VL-72B-Instruct` is the
72B Qwen vision-language model used here.

`temporal_event.pbs` invokes `python3 main.py --mode temporal`, which uses the
default extract-plus-timeline behavior. It runs the API backend, so no GPU or
Hugging Face model cache is required. Export `OPENCODE_API_KEY` before `qsub`
or provide it through the repository's local `.env` file.

## Structure

| Path | Purpose |
|------|---------|
| `main.py` | CLI dispatcher for the benchmark modes |
| `pipeline/SparseEventPipeline.py` | Sparse-event pipeline class |
| `pipeline/TemporalPipeline.py` | Staged temporal extraction, timeline, storyline, and question pipeline |
| `pipeline/ConfigLoader.py` | Shared JSON configuration loader |
| `pipeline/frame_parser.py` | Splits numerically named frame images into sections |
| `pipeline/AIParser.py` | Configurable local vLLM or OpenAI-compatible API wrapper |
| `pipeline/prompts/*.txt` | Prompt files for benchmark tasks |
| `configs/*.json` | Pipeline configuration files |
| `test_vlm.py` | VLM prototype/test script |
| `vllm.pbs` | Legacy PBS job script for the prototype |
| `sparse_event.pbs` | PBS job script for the current sparse-event pipeline |
| `temporal_event.pbs` | API-backed temporal PBS job script |

## Sparse-Event Pipeline

The pipeline reads a JSON configuration, creates a `FrameParser`, and prepares an `AIParser` using the complete configuration. The `backend` setting selects local vLLM or the OpenAI-compatible Responses API. Its intended flow is:

1. Load `configs/sparse_events.json`.
2. Sort and sample the input frames.
3. Copy or move frames into section directories.
4. Query the VLM for each section using the task prompt.
5. Write per-section and aggregated results.

The pipeline loads the model once, partitions frames, sends each section to the VLM, and writes one JSON result per section plus `all_results.json`.

## Temporal Pipeline

The temporal pipeline targets temporal-order recall. It uses the following
stages:

1. **Extract:** `FrameParser` samples and partitions frames. `AIParser.call_vlm`
   produces one structured JSON response per section. Existing
   `<output>/section_NNNN_output/result.json` files are skipped, allowing an
   interrupted extraction job to resume.
2. **Timeline:** Section events are parsed, sorted in Python by the numeric
   timestamp in `start_frame`, and sent to the LLM in windows controlled by
   `merge_window`. The LLM only merges continuing or duplicate observations;
   it is instructed not to reorder or invent events. The result is written to
   `timeline.json`.
3. **Storyline:** The merged timeline is converted into chronological prose in
   `storyline.txt` for human review.
4. **Questions:** After `timeline.json` has been reviewed and corrected by a
   human, the LLM generates pairwise-order, sequence-order, and before/after
   questions in `questions.json`. Each question includes event IDs and frame
   evidence.

The temporal prompts are:

| Prompt | Purpose |
|--------|---------|
| `pipeline/prompts/temporal_vlm.txt` | Extract section summaries and timestamped events |
| `pipeline/prompts/temporal_llm_filter.txt` | Merge duplicate or continuing events without reordering |
| `pipeline/prompts/temporal_storyline.txt` | Write the chronological prose storyline |
| `pipeline/prompts/temporal_question_gen.txt` | Generate reviewed temporal-order questions |

The timeline and question stages can be run separately because question
generation intentionally consumes a human-verified timeline rather than raw
VLM output.

## Configuration

The sparse-event configuration is stored in `configs/sparse_events.json`; the
temporal configuration is stored in `configs/temporal_events.json`.


The `task` value identifies the benchmark category. The top-level `--mode`
selects the pipeline and its prompt files:

| Task | Prompt | Purpose |
|------|--------|---------|
| `1` | `pipeline/prompts/sparse_event_prompt.txt` | Sparse event localisation |
| `2` | `pipeline/prompts/temporal_vlm.txt` | Temporal section extraction |

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
4. Keep `max_tokens` within the context budget. The temporal configuration uses
   `512` for section JSON extraction because its schema includes summaries,
   objects, and events. Storyline and question calls use their own output
   limits.

### Other Parameters

| Parameter | Selection guidance |
|-----------|--------------------|
| `task` | Benchmark category metadata: `1` is sparse-event localisation; `2` is temporal/narrative. |
| `frames_dir` | Use an absolute path on Gadi when the data is outside the repository. |
| `sections_dir` | Intermediate frame sections. Use scratch storage for large runs. |
| `output` | Final JSON results directory. Use scratch storage for large runs. |
| `frames_per_section` | Images sent in one VLM request. Lower values reduce memory; `5-10` is a useful starting range. |
| `step` | Keeps every Nth sorted frame. Higher values reduce compute but lose temporal detail. |
| `model` | Local model identifier or API model identifier, depending on `backend`. |
| `backend` | `local` loads vLLM; `api` uses the OpenAI-compatible Responses API. |
| `api_base_url` | API base URL or full Responses endpoint. The `/responses` suffix is normalized automatically. |
| `api_key_env` | Environment variable containing the API key. Do not put the key in JSON. |
| `temperature` | Use `0.0-0.2` for stable JSON; higher values produce more variation. |
| `reasoning_effort` | API reasoning effort, such as `low` or `none`; omit when unsupported. |
| `max_tokens` | Maximum generated output tokens. Increase if responses are truncated. |
| `merge_window` | Maximum number of sorted events supplied to one timeline-merge call. The temporal configuration uses `50`. |
| `storyline_max_tokens` | Output limit for the temporal storyline call. |
| `question_max_tokens` | Output limit for temporal question generation. |
| `enforce_eager` | `true` is usually safer; `false` may improve speed but can require more memory. |
| `dtype` | Use `bfloat16` on Hopper GPUs for the current Qwen model; use `half` when FP16 is required. |
| `gpu_memory_utilization` | Usually `0.85-0.9`. Leave some memory for CUDA and image processing. |
| `tensor_parallel_size` | Number of GPUs used by one model instance. It must match the PBS GPU allocation; current value is `4`. |
| `move` | Set `true` only if input frames may be moved instead of copied. Defaults to `false`. |

The supplied temporal configuration currently uses `frames_per_section: 10`,
`step: 15`, no API `temperature`, `reasoning_effort: low`, `max_tokens: 2048`, `merge_window: 50`,
`storyline_max_tokens: 1024`, `question_max_tokens: 2048`,
`max_model_len: 8192`, and `tensor_parallel_size: 4`.

### API Backend

Set the following values in the selected configuration:

```json
{
    "backend": "api",
    "model": "gpt-5.6-luna",
    "api_base_url": "https://opencode.ai/zen/go/v1/responses",
    "api_key_env": "OPENCODE_API_KEY",
    "temperature": null,
    "reasoning_effort": "low"
}
```

Then export the key using the configured environment variable:

```bash
export OPENCODE_API_KEY="your-api-key"
python main.py --mode temporal --stage extract
```

The API backend sends the prompt and every image in a section as a single
Responses API request. `gpt-5.6-luna` supports image inputs. For GPT-5.6
reasoning models, the temporal configuration omits `temperature`; its
`max_tokens` value includes both reasoning and visible output tokens.

If an API call fails, the pipeline reports the model, endpoint, HTTP status,
response body, and provider request ID when available. A 401 indicates a key
or subscription problem; a 5xx response should be retried and reported to the
provider with the request ID.

## Frame Sections

`FrameParser` accepts `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp` files. It sorts frames by their numeric filename, keeps every `step`-th frame, and groups them into directories such as:

```text
sections/
    section_0000/
    section_0001/
```

Set `move` to `true` in the configuration to move frames instead of copying them. It defaults to `false` when omitted.

## Intended Output

For a temporal run, the output is:

```text
<output>/
    all_results.json
    timeline.json
    storyline.txt
    questions.json
    section_0000_output/result.json
    section_0001_output/result.json
```

`questions.json` is created only by the explicit `questions` stage. Sparse-event
runs write `all_results.json` and per-section results but do not create the
temporal timeline, storyline, or questions files.

## Full Documentation

See [CODEBASE_DOCUMENTATION.md](CODEBASE_DOCUMENTATION.md) for a detailed breakdown of the codebase and implementation status.
