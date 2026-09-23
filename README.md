# Episodic Memory Pipeline

Benchmarking pipeline for assessing Vision Language Models' (VLMs) episodic memory capabilities using long-form egocentric dashcam frames.

## Current Status

The top-level runner accepts four benchmark modes. `sparse` and `temporal` are connected to pipelines; `spatial` and `counting` are recognized but not implemented yet.

| Mode | Status |
|------|--------|
| `sparse` | Resumable extraction, review/filter, and question generation implemented |
| `temporal` | Resumable extraction, timeline construction, storyline, and question generation implemented |
| `spatial` | Not implemented |
| `counting` | Not implemented |

Both connected branches support local vLLM and OpenAI-compatible API inference. The supplied sparse-event configuration uses local vLLM with `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` and four GPUs; the supplied temporal configuration uses local vLLM with `Qwen/Qwen2.5-VL-72B-Instruct` and also requires four GPUs. Set `backend` and `model` in a configuration to switch that run to the OpenAI-compatible API.

## Setup

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

API mode requires the environment variable named by `api_key_env` in the
selected configuration. The supplied sparse configuration uses
`OPENCODE_API_KEY`, loaded from the environment or the repository `.env` file.

## Gadi Data Paths

All project data now lives on the project's mass-data storage at
`/g/data/pg06/FYP2026S1_3473` (moved from `/scratch/pg06/FYP2026S1_3473`).
This covers the Boreas frame datasets, pipeline outputs, and the Hugging Face
cache. The shared vLLM environment is still on scratch at
`/scratch/pg06/vm4618/envs/vllm_env`, so every Gadi job script requests both
filesystems with `#PBS -l storage=scratch/pg06+gdata/pg06`.

The local vLLM job scripts also export `VLLM_USE_DEEP_GEMM=0`. DeepGEMM
JIT-compiles its FP8 kernels with NVCC and requires version 12.3 or newer,
while the `cuda/12.2.2` module provides 12.2.2; disabling it lets vLLM use its
CUTLASS FP8 kernels instead.

## Run A Benchmark Mode

Run commands from the repository root because configuration paths are relative to the current working directory:

```bash
python main.py --mode sparse
```

The sparse mode uses `configs/sparse_events.json` by default and runs extraction
followed by review when no stage is specified. The temporal mode uses
`configs/temporal_events.json` and runs extraction followed by timeline
construction when no stage is specified. Either pipeline's stages can be run
independently:

```bash
python main.py --mode sparse --stage extract
python main.py --mode sparse --stage review
python main.py --mode sparse --stage questions

python main.py --mode temporal --stage extract
python main.py --mode temporal --stage timeline
python main.py --mode temporal --stage questions
```

The sparse review workflow is:

1. Run `--stage extract` to detect noteworthy events per section. Existing
   section results are reused.
2. Run `--stage review` to sort detections by frame timestamp, merge
   duplicates across section boundaries, filter spurious detections, and write
   `events.json`.
3. Review and edit `events.json` against the referenced frames.
4. Run `--stage questions` to generate sparse-event questions from the
   reviewed events.
5. Review `questions.json` (including the listed `false_events`) before using
   it as benchmark data.

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
| `run_timeframe_converter.py` | Applies timeframe conversion to events in `timeline.json` |
| `pipeline/ConfigLoader.py` | Shared JSON configuration loader |
| `pipeline/frame_parser.py` | Splits numerically named frame images into sections |
| `pipeline/image_preprocessor.py` | Re-encodes frames as resized JPEGs while sections are built |
| `pipeline/AIParser.py` | Configurable local vLLM or OpenAI-compatible API wrapper |
| `pipeline/prompts/*.txt` | Prompt files for benchmark tasks |
| `configs/*.json` | Pipeline configuration files |
| `sparse_event.pbs` | PBS job script for the current sparse-event pipeline |
| `temporal_event.pbs` | Local vLLM temporal PBS job script |

## Sparse-Event Pipeline

The sparse pipeline targets rare-event localisation and anti-hallucination. It
uses the following stages:

1. **Extract:** `FrameParser` samples and partitions frames and writes
   `sections.json` with each section's elapsed start/end seconds.
   `AIParser.call_vlm` produces one structured JSON response per section.
   Each section's parsed detections are stored with the section's
   `start_seconds`/`end_seconds` and a `frame_seconds` value computed from the
   detected frame's timestamp. Existing
   `<output>/section_NNNN_output/result.json` files are skipped, allowing an
   interrupted extraction job to resume. Each section's frame images are
   deleted once the VLM has processed it, and the per-section result folders
   are deleted once `all_results.json` has been written.
2. **Review:** Detections are parsed, sorted in Python by the numeric
   timestamp in `frame`, and sent to the LLM in windows controlled by
   `review_window`. The LLM merges duplicate observations of the same rare
   event across a section boundary and filters out spurious detections (normal
   driving, ordinary traffic, mundane scenery), reporting them in `removed`.
   It is instructed not to reorder or invent events. Seconds are re-attached
   after merging. The result is written to `events.json`.
3. **Questions:** After `events.json` has been reviewed and corrected by a
   human, the LLM generates existence, deceptive, and noteworthy questions in
   `questions.json`. Deceptive questions use one invented plausible false
   event, and noteworthy questions offer four options of which exactly one is
   a real event and three are invented false events; every false event is
   listed in `questions.json` as a `false_events` array for review.

The sparse prompts are:

| Prompt | Purpose |
|--------|---------|
| `pipeline/prompts/sparse_event_prompt.txt` | Detect noteworthy events per section with the VLM |
| `pipeline/prompts/sparse_event_filter.txt` | Merge duplicate detections and filter spurious ones |
| `pipeline/prompts/sparse_event_question_gen.txt` | Generate reviewed sparse-event questions |

## Temporal Pipeline

The temporal pipeline targets temporal-order recall. It uses the following
stages:

1. **Extract:** `FrameParser` samples and partitions frames and writes
   `sections.json` with each section's elapsed start/end seconds. `AIParser.call_vlm`
   produces one structured JSON response per section. Each section's parsed
   events are stored with the section's `start_seconds`/`end_seconds` attached.
   Existing `<output>/section_NNNN_output/result.json` files are skipped,
   allowing an interrupted extraction job to resume. Each section's frame
   images are deleted once the VLM has processed it, and the per-section result
   folders are deleted once `all_results.json` has been written.
2. **Timeline:** Section events are parsed, sorted in Python by the numeric
   timestamp in `start_frame`, and sent to the LLM in windows controlled by
   `merge_window`. The LLM only merges continuing or duplicate observations;
   it is instructed not to reorder or invent events. Per-section seconds are
   re-attached after merging. The result is written to `timeline.json`.
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

The batch scripts under `scripts/` (`run_all_local.sh`, `run_all_api.sh`) rewrite
only the per-list `frames_dir`, `sections_dir`, and `output` paths before each
run. `backend`, `model`, and every other inference setting are read from the
configuration file, so set `backend` and `model` there before submitting a batch
job.


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
| `step` | Stride between sampled frames inside one section. Use `1` to keep every frame; use a larger value to reduce redundant frames, but keep enough samples to judge distances and landmarks. |
| `skip` | Number of frames from a section's last sampled frame to the next section's first frame. Defaults to `step` when omitted, which samples without gaps; use a larger value to leave unsampled gaps between sections. |
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

### Sparse

Sparse runs use `configs/sparse_events.json`. The most important choices are
`step`, `skip`, and `frames_per_section`: brief rare events disappear when
sampling is too sparse, so sample more densely than for temporal runs where
section coverage matters less.

| Parameter | Meaning and guidance |
|-----------|----------------------|
| `task` | Set to `1` for sparse event localisation. |
| `frames_dir` | Directory containing the numbered camera frames. Set this to the frames from the video under evaluation. |
| `sections_dir` | Directory where sampled frames are grouped into sections. Use a fresh directory when changing sampling settings. |
| `output` | Directory for section results, `events.json`, and `questions.json`. Use a separate output directory for each experiment. |
| `frames_per_section` | Number of sampled frames grouped into one extraction request. Increase it for wider context, but keep it within the model's image/context limits; `5` is the supplied starting point. |
| `step` | Stride between sampled frames inside one section. Use `1` for maximum temporal coverage; increase it when adjacent frames are redundant. Reduce it when brief events may be missed. |
| `skip` | Number of frames from a section's last sampled frame to the next section's first frame. Defaults to `step` when omitted, which samples without gaps; use a larger value to leave unsampled gaps between sections. |
| `image_max_size` | Longest side, in pixels, of frames written into sections. Frames are downscaled to fit (never upscaled). Omit it to keep original frame sizes. Use a small value, such as `1024`, to keep API request payloads small and stay within the vision-token budget of models such as `grok-4.6`; omit it for local vLLM runs that accept full-resolution frames. |
| `image_quality` | JPEG quality, from `1` to `95`, used when frames are re-encoded for sections. `90` is the supplied value. Omit it with `image_max_size` to copy frames through untouched. |
| `move` | Controls whether input frames are moved or copied into sections. Leave it `false` or omit it unless the source frames can be removed. |
| `model` | Model used for extraction and later sparse stages. Pick a model that accepts the selected backend and image inputs. |
| `backend` | `local` for vLLM or `api` for the OpenAI-compatible Responses API. The configuration selects the backend for a run; the batch scripts in `scripts/` do not override it. The backend must match the model and available infrastructure. |
| `api_base_url` | OpenAI-compatible Responses API endpoint. Configure this only when using `backend: "api"`. |
| `api_key_env` | Environment variable name containing the API key. Export the matching variable before an API run, for example `OPENCODE_API_KEY`. |
| `temperature` | Local vLLM sampling temperature. It is ignored by the supplied API configuration; use a low value if switching sparse extraction to local vLLM. |
| `api_temperature` | API sampling temperature, used only with the API backend. Keep it `null` for API models that do not accept the temperature parameter, such as `gpt-5.6-luna`. |
| `reasoning_effort` | API reasoning effort, when supported. Use a lower value for faster extraction or a higher value when the model needs more effort to judge whether a detection is real. |
| `max_tokens` | Maximum response length for each section extraction. Increase it if detection lists are being truncated. |
| `review_window` | Number of chronologically sorted detections sent to the review model at once. Increase it to give the reviewer more context, but keep it within the model's context limit; `50` is the supplied starting point. |
| `question_max_tokens` | Maximum length of generated sparse questions and false events. Increase it when generating many questions or detailed evidence fields. |
| `enforce_eager` | Local vLLM execution setting. Keep `true` unless the selected local model and vLLM setup support another mode. |
| `dtype` | Local model numeric precision. Use a hardware-supported value such as `bfloat16` to balance memory use and quality. |
| `max_model_len` | Maximum local model context length. Increase it only when the model and available GPU memory can support the larger context. |
| `gpu_memory_utilization` | Fraction of GPU memory allocated to vLLM. Start around `0.9` and lower it if the local model does not fit. |
| `tensor_parallel_size` | Number of GPUs used by local vLLM. Set it to the number of compatible GPUs assigned to the run. |

### Temporal

Temporal runs use `configs/temporal_events.json`. The most important choices
are `step`, `skip`, and `frames_per_section`: use smaller sampling intervals
for short events and larger sections only when the model can handle the added
context.

| Parameter | Meaning and guidance |
|-----------|----------------------|
| `task` | Set to `2` for temporal section extraction. |
| `frames_dir` | Directory containing the numbered camera frames. Set this to the frames from the video under evaluation. |
| `sections_dir` | Directory where sampled frames are grouped into temporal sections. Use a fresh directory when changing sampling settings. |
| `output` | Directory for section results, `timeline.json`, `storyline.txt`, and `questions.json`. Use a separate output directory for each experiment. |
| `frames_per_section` | Number of sampled frames grouped into one extraction request. Increase it for wider context, but keep it within the model's image/context limits; `5` is the supplied starting point. |
| `step` | Stride between sampled frames inside one section. Use `1` for maximum temporal coverage; increase it when adjacent frames are redundant. Reduce it when brief events may be missed; `15` is the supplied starting point. |
| `skip` | Number of frames from a section's last sampled frame to the next section's first frame. Defaults to `step` when omitted, which samples without gaps; use a larger value to leave unsampled gaps between sections. |
| `move` | Controls whether input frames are moved or copied into sections. Leave it `false` or omit it unless the source frames can be removed. |
| `model` | Model used for extraction and later temporal stages. The supplied configuration uses `Qwen/Qwen2.5-VL-72B-Instruct` locally. |
| `backend` | Use `local` for the supplied four-GPU vLLM configuration or `api` for a Responses API model. The backend must match the model and available infrastructure. |
| `api_base_url` | OpenAI-compatible Responses API endpoint. Configure this only when using `backend: "api"`. |
| `api_key_env` | Environment variable name containing the API key. Export the matching variable before an API run, for example `OPENCODE_API_KEY`. |
| `temperature` | Local vLLM sampling temperature. Keep it `null` to use the local `0.2` default; use a low value when deterministic extraction is preferred. |
| `api_temperature` | API sampling temperature, used only with the API backend. Keep it `null` for models that do not accept the parameter. |
| `reasoning_effort` | API reasoning effort, used only when the temporal backend is switched to an API model. |
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
    "api_temperature": null,
    "reasoning_effort": "low"
}
```

Then export the key using the configured environment variable:

```bash
export OPENCODE_API_KEY="your-api-key"
python main.py --mode sparse --stage extract
```

The API backend sends the prompt and every image in a section as a single
Responses API request. `gpt-5.6-luna` supports image inputs. For GPT-5.6
reasoning models, the configuration omits `temperature`; API requests use
`api_temperature`, which is omitted when `null`, and `max_tokens` includes
both reasoning and visible output tokens.

The API wrapper sends a stable per-run `x-opencode-session` header and an
identifying user agent, as required by OpenCode Go for third-party clients.

If an API call fails, the pipeline reports the model, endpoint, HTTP status,
response body, and provider request ID when available. A 401 indicates a key
or subscription problem; a 5xx response should be retried and reported to the
provider with the request ID.

## Frame Sections

`FrameParser` accepts `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp` files. It sorts frames by their numeric filename and samples each section with `step` as the stride between frames up to `frames_per_section` frames; the next section starts `skip` frames after the last sampled frame (defaulting to `step`, so sampling continues without gaps). It groups the sampled frames into directories such as:

```text
sections/
    section_0000/
    section_0001/
```

While creating sections it also writes a `sections.json` manifest next to the
section directories. Each section maps to its first/last frame and the elapsed
start/end seconds of that window relative to the first sampled frame of the
video. These seconds are computed from the real frame timestamps, so later
stages know the time range of every section without relying on model output.

Set `move` to `true` in the configuration to move frames instead of copying them. It defaults to `false` when omitted.

When `image_max_size` or `image_quality` is set in the configuration, `FrameParser`
runs every sampled frame through `ImagePreprocessor` while writing sections:
frames are converted to RGB and re-encoded as JPEGs with the configured
quality, downscaled so the longest side fits `image_max_size`. Numeric
filenames keep their stem, so section and event timestamp handling is
unaffected. Downscaled JPEG sections produce much smaller base64 payloads for
the API backend, which otherwise rejects large multi-image requests.

## Intended Output

For a temporal run, the output is:

```text
<output>/
    all_results.json
    timeline.json
    storyline.txt
    questions.json
```

For a sparse-event run, the review stage replaces `timeline.json` with
`events.json` and writes no storyline. Extraction keeps only the aggregate:

```text
<output>/
    all_results.json
    events.json
    questions.json
```

Each element of `all_results.json` stores the raw VLM response plus a parsed
`events` list. Every event carries `start_seconds` and `end_seconds` taken from
the section's `start_seconds`/`end_seconds` window in `sections.json`, so a model
only needs to name the section and the time range is already known. Sparse events
also carry `frame_seconds`, computed from the detected frame's own timestamp.
`timeline.json`/`events.json` events carry the same seconds after merging.

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

To convert a timeline file and write a new copy with the seconds added:

```bash
python3 run_timeframe_converter.py
```

Use `--timeline` and `--output` to select different paths. The source frame
directory must be available, and frame names must contain numeric timestamps.
For temporal runs this conversion is redundant: per-section seconds are already
computed at frame-parsing time and attached to every event.

During VLM extraction, the ordered filenames from each section are included in
the prompt so temporal events can refer to the original frame files instead of
invented labels.

## Full Documentation

See [CODEBASE_DOCUMENTATION.md](CODEBASE_DOCUMENTATION.md) for a detailed breakdown of the codebase and implementation status.
