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

The temporal review workflow is:

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

## Structure

| Path | Purpose |
|------|---------|
| `main.py` | CLI dispatcher for the benchmark modes |
| `pipeline/SparseEventPipeline.py` | Sparse-event pipeline class |
| `pipeline/TemporalPipeline.py` | Staged temporal extraction, timeline, storyline, and question pipeline |
| `pipeline/timeframe_converter.py` | Converts timestamped event frame ranges into elapsed video seconds |
| `pipeline/ConfigLoader.py` | Shared JSON configuration loader |
| `pipeline/frame_parser.py` | Splits numerically named frame images into sections |
| `pipeline/AIParser.py` | Configurable local vLLM or OpenAI-compatible API wrapper |
| `pipeline/prompts/*.txt` | Prompt files for benchmark tasks |
| `configs/*.json` | Pipeline configuration files |
| `sparse_event.pbs` | PBS job script for the current sparse-event pipeline |
| `temporal_event.pbs` | API-backed temporal PBS job script |

## Sparse-Event Pipeline

The pipeline reads a JSON configuration, creates a `FrameParser`, and prepares an `AIParser` using the complete configuration. The `backend` setting selects local vLLM or the OpenAI-compatible Responses API. Its intended flow is:

1. Load `configs/sparse_events.json`.
2. Sort and sample the input frames, unless section directories already exist
   in `sections_dir`.
3. Copy or move frames into section directories when they do not already exist.
4. Query the VLM for each section using the task prompt.
5. Write per-section and aggregated results.

The pipeline loads the model once, reuses existing `section_*` directories when
available, sends each section to the VLM, and writes one JSON result per section
plus `all_results.json`.

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

## Config Guide

The JSON configuration controls frame sampling, output locations, and model
inference. Paths are interpreted relative to the directory from which the
command is run unless an absolute path is provided.

### Spatial

The spatial pipeline is not implemented yet, so there is no runnable spatial
configuration file. When spatial support is added, use the shared settings
below and choose values based on the spatial detail needed by the benchmark:

| Parameter | Meaning and guidance |
|-----------|----------------------|
| `task` | Identifies the benchmark category. Keep the value defined by the spatial pipeline when it is available. |
| `frames_dir` | Directory containing the numbered input frames. Point this to the camera frames for the video being evaluated. |
| `sections_dir` | Directory where sampled frames are grouped into sections. Use a separate directory for each run so results are not mixed. |
| `output` | Directory for model results. Use a location with enough space for one result per section and the aggregate output. |
| `frames_per_section` | Maximum frames sent to the model in one request. Start small enough to fit the model context, then increase it if spatial references need more surrounding frames. |
| `step` | Sampling interval after frames are sorted. Use `1` to keep every frame; use a larger value to reduce redundant frames, but keep enough samples to judge distances and landmarks. |
| `move` | If `true`, moves frames into sections; if `false` or omitted, copies them. Keep it `false` when the original frames must be preserved. |
| `model` | Model used for the run, such as `Qwen/Qwen2.5-VL-72B-Instruct`. Pick a vision-language model that supports image inputs and has enough context for each section. |
| `backend` | Inference backend: `local` for vLLM or `api` for the OpenAI-compatible API. Choose `local` for an available local model and GPUs, otherwise use `api`. |
| `api_base_url` | OpenAI-compatible API endpoint, used only with `backend: "api"`. Set it to the endpoint provided by the API service. |
| `api_key_env` | Name of the environment variable containing the API key, used only with the API backend. Choose any variable name, then export that variable before running. |
| `temperature` | Controls response variation. Use a low value, such as `0` to `0.4`, for consistent spatial observations and comparisons. |
| `max_tokens` | Maximum generated response length. Set it high enough for the requested spatial explanation, but avoid a large value that adds unnecessary cost or latency. |
| `reasoning_effort` | API reasoning level, when supported. Use a lower value for faster runs or a higher value when distance and location reasoning needs more deliberation. |
| `enforce_eager` | Local vLLM execution setting. Keep `true` unless the selected local model and vLLM setup support a different execution mode. |
| `dtype` | Local model numeric precision, such as `bfloat16`. Use the precision supported by the hardware to balance memory use and quality. |
| `max_model_len` | Maximum local model context length. Set it large enough for the prompt and all section images, within the model and GPU limits. |
| `gpu_memory_utilization` | Fraction of GPU memory allocated to vLLM. Start around `0.9` and lower it if model loading or other GPU processes run out of memory. |
| `tensor_parallel_size` | Number of GPUs across which a local model is split. Set it to the number of compatible GPUs allocated to the job. |

### Temporal

Temporal runs use `configs/temporal_events.json`. The most important choices
are `step` and `frames_per_section`: use smaller sampling intervals for short
events and larger sections only when the model can handle the added context.

| Parameter | Meaning and guidance |
|-----------|----------------------|
| `task` | Set to `2` for temporal section extraction. |
| `frames_dir` | Directory containing the numbered camera frames. Set this to the frames from the video under evaluation. |
| `sections_dir` | Directory where sampled frames are grouped into temporal sections. Use a fresh directory when changing sampling settings. |
| `output` | Directory for section results, `timeline.json`, `storyline.txt`, and `questions.json`. Use a separate output directory for each experiment. |
| `frames_per_section` | Number of sampled frames grouped into one extraction request. Increase it for wider context, but keep it within the model's image/context limits; `5` is the supplied starting point. |
| `step` | Number of input frames skipped between samples. Use `1` for maximum temporal coverage; increase it when adjacent frames are redundant. Reduce it when brief events may be missed; `15` is the supplied starting point. |
| `move` | Controls whether input frames are moved or copied into sections. Leave it `false` or omit it unless the source frames can be removed. |
| `model` | Model used for extraction and later temporal stages, such as `gpt-5.6-luna`. Pick a model that accepts the selected backend and image inputs. |
| `backend` | Use `api` for the supplied Responses API configuration or `local` for a local vLLM model. The backend must match the model and available infrastructure. |
| `api_base_url` | OpenAI-compatible Responses API endpoint. Configure this only when using `backend: "api"`. |
| `api_key_env` | Environment variable name containing the API key. Export the matching variable before an API run, for example `OPENCODE_API_KEY`. |
| `temperature` | Controls generation variation. Keep it `null` for the supplied reasoning-model API configuration; use a low value when the selected backend supports temperature and deterministic extraction is preferred. |
| `reasoning_effort` | API reasoning effort. Use `low` for faster extraction, or increase it when the model needs more effort to distinguish event order. |
| `max_tokens` | Maximum response length for each section extraction. Increase it if event lists are being truncated; lower it to reduce cost when responses are short. |
| `merge_window` | Number of chronologically sorted events sent to the timeline-merging model at once. Increase it to give the merger more context, but keep it within the model's context limit; `50` is the supplied starting point. |
| `storyline_max_tokens` | Maximum length of the generated chronological storyline. Choose a value large enough to describe all reviewed events without excessive prose. |
| `question_max_tokens` | Maximum length of generated temporal questions. Increase it when generating many questions or detailed evidence fields. |
| `enforce_eager` | Local vLLM execution setting. Keep `true` unless the selected local model and vLLM setup support another mode. |
| `dtype` | Local model numeric precision. Use a hardware-supported value such as `bfloat16` to balance memory use and quality. |
| `max_model_len` | Maximum local model context length. Increase it only when the model and available GPU memory can support the larger context. |
| `gpu_memory_utilization` | Fraction of GPU memory allocated to vLLM. Start around `0.9` and lower it if the local model does not fit. |
| `tensor_parallel_size` | Number of GPUs used by local vLLM. Set it to the number of compatible GPUs assigned to the run. |


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

## Frame Time Conversion

`TimeframeConverter` converts the `start_frame` and `end_frame` values from
temporal events into elapsed seconds. It uses the earliest numeric frame
timestamp in `frames_dir` as time zero; frame filenames are expected to be
microsecond timestamps.

```python
from pipeline.timeframe_converter import TimeframeConverter

converter = TimeframeConverter("/path/to/frames")
seconds = converter.timeframe_to_seconds(
    "1733343593917869.png",
    "1733343596067826.png",
)
# {"start_seconds": 0.0, "end_seconds": 2.149957}
```

## Full Documentation

See [CODEBASE_DOCUMENTATION.md](CODEBASE_DOCUMENTATION.md) for a detailed breakdown of the codebase and implementation status.
