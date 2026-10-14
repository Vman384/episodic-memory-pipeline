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

Both connected branches support local vLLM and API inference through OpenAI-compatible or Anthropic Messages protocols. The supplied sparse-event configuration uses local vLLM with `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` and four GPUs; the supplied temporal configuration uses local vLLM with `Qwen/Qwen2.5-VL-72B-Instruct` and also requires four GPUs. Set `backend` and `model` in a configuration to switch that run to an API model.

## Setup

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

API mode requires the environment variable named by `api_key_env` in the
selected configuration. The supplied sparse configuration uses
`OPENCODE_API_KEY`; Gemini frame and native-video QA use `GEMINI_API_KEY`.
Keys can be exported in the environment or loaded from the repository `.env`
file. Local Qwen3-VL video input uses `qwen-vl-utils` inside the vLLM
environment.

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

## Plot Boreas Video Lengths

Run `plot_boreas_video_lengths.py` to query each
`boreas-*/video.mp4` under `/g/data/pg06/FYP2026S1_3473/boreas_dataset/` with
`ffprobe`, multiply each duration by three, and save a frequency histogram:

```bash
python3 plot_boreas_video_lengths.py
```

The histogram uses 8 equal-width bins across the observed scaled-duration range
and is saved as `boreas_video_length_frequency.png` in the current directory.
`ffprobe` is installed with ffmpeg, and matplotlib is included in
`requirements.txt`. Use `--dataset-root`, `--bins` (6–10, default 8), and
`--output` to change the input directory, bin count, and plot path.

## Count Generated Questions

Submit `qsub count_questions.sh` to count all entries in `questions.json` files
under each dataset's `sparse_outputs/events/` and
`temporal_outputs/narratives/` directories in
`/g/data/pg06/FYP2026S1_3473`. The PBS job requests access to the project's
Gadi storage and prints the combined count. To run directly on a node where the
data filesystem is already mounted, use `bash count_questions.sh`.

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
| `pipeline/frame_parser.py` | Splits timestamp-named frame images into sections |
| `pipeline/image_preprocessor.py` | Re-encodes frames as resized JPEGs while sections are built |
| `pipeline/AIParser.py` | Configurable local vLLM, OpenAI-compatible API, or Anthropic Messages wrapper |
| `pipeline/prompts/*.txt` | Prompt files for benchmark tasks |
| `configs/*.json` | Pipeline configuration files |
| `download_boreas.pbs` | Downloads only `camera/` for its explicit Boreas sequence list, skipping sequences whose destination camera folder already exists |
| `download_boreas_videos.pbs` | Downloads each Boreas sequence's `video.mp4` from `DATASET_PROGRESS.md`, excluding `boreas-2025-07-18-15-12` |
| `sparse_event.pbs` | PBS job script for the current sparse-event pipeline |
| `temporal_event.pbs` | Local vLLM temporal PBS job script |
| `forest_run.sh` | Local Qwen3-VL sparse-event PBS job for the forest K-01 dataset; API and temporal examples are inactive |
| `run_model_answers.py` | Legacy image-answer runner; use `QA/` for the bounded, shared-question evaluation path |
| `grade_answers.py` | Scores an answers JSON against the question file and writes per-question results |
| `run_all_eval.sh` | PBS job that runs `QA/` and grading for Boreas sequences listed in `DATASET_PROGRESS.md` on the `copyq` queue |
| `prepare_vertex_eval.pbs` | `copyq` job that samples and stages selected datasets' frames for local Vertex AI evaluation |
| `prepare_vertex_frames.py` | Reuses the QA frame sampler to create resized, timestamp-ordered JPEGs for a staged dataset |
| `run_vertex_frame_eval.py` | Local worker that SCP-pulls one dataset, calls Vertex AI, grades it, and deletes temporary images before continuing |
| `test_vertex.py` | Vertex AI ADC smoke test for the configured GCP project |
| `average_model_mark.py` | Prints the arithmetic mean of a model's completed dataset marks |
| `submit_remaining_opencode_eval.sh` | Submits `run_all_eval.sh` for OpenCode Go QA models, excluding completed GPT-6 Luna and Kimi K3 |
| `run_model_eval.sh` | PBS job that runs the QA runner and grader with local Qwen3-VL on Gadi |
| `count_questions.sh` | Counts questions across sparse and temporal output files on Gadi |
| `plot_boreas_video_lengths.py` | Reads Boreas MP4 durations with ffprobe, scales each by three, and saves a frequency histogram |
| `QA/` | Video questionnaire runner: strips answer metadata, shuffles questions, and selects frame or native-video input from model configuration |
| `tests/test_responses_store_recovery.py` | Tests recovery from Responses API stored-response size limits |
| `test_gateway.sh` | Checks that the OpenCode API keeps a conversation across requests (`previous_response_id`); run with `OPENCODE_API_KEY=... bash test_gateway.sh` |
| `configs/answer_eval.json` | Model configuration used by legacy `run_model_answers.py` |
| `configs/qa_kimi_k3.json` | OpenCode Go Kimi K3 frame-input config using Chat Completions |
| `configs/qa_claude_haiku_5_5.json` | OpenCode Go Claude Haiku 5.5 frame-input config using Anthropic Messages |
| `configs/qa_glm_5_3_flash.json` | OpenCode Go GLM-5.3-Flash frame-input config using Chat Completions |
| `configs/qa_gpt6_luna.json` | OpenCode Go GPT-6 Luna frame-input config using the Responses API |
| `configs/qa_grok_4_7.json` | OpenCode Go Grok 4.7 frame-input config using the Responses API; included by `submit_remaining_opencode_eval.sh` |
| `configs/qa_mimo_v2_6_pro.json` | OpenCode Go MiMo-V2.6-Pro frame-input config using Chat Completions |
| `configs/qa_qwen3_8_max.json` | OpenCode Go Qwen3.8 Max frame-input config using Chat Completions |
| `configs/qa_deepseek_v4_flash_vision_exp.json` | OpenCode Go DeepSeek V4 Flash Vision Exp frame-input config using Chat Completions |
| `configs/qa_gemini_3_8_flash.json` | Gemini 3.8 Flash frame-input config using Google's OpenAI-compatible Chat Completions API |
| `configs/qa_gemini_video.json` | Gemini QA config using native MP4 input in standalone QA or `run_all_eval.sh` |
| `configs/qa_qwen3.json` | Local Qwen3-VL QA config for native video input on Gadi |

To download the full videos for the Boreas sequences listed in
`DATASET_PROGRESS.md` (except `boreas-2025-07-18-15-12`), submit the PBS job from
the repository root:

```bash
qsub download_boreas_videos.pbs
```

The job saves each file as
`/g/data/pg06/FYP2026S1_3473/boreas_dataset/<sequence>/video.mp4` and skips files
that are already present.

## Testing Questions Against a Model

Use `QA/` to answer generated questions (`questions.json` from either pipeline)
and `grade_answers.py` to score them. Run both from the repository root.

1. Get the model's answers. The QA runner samples one bounded, deterministic
   frame set for the drive, then answers all questions together:

   ```bash
   python -m QA.run_questionnaire \
       --frames-dir /path/to/camera \
       --questions /path/to/questions.json \
       --config configs/qa_gpt6_luna.json \
       --frame-budget 128 \
       --max-image-size 512 \
       --api-batch-size 20 \
       --seed 42 \
       --output answers.json
   ```

   `--frame-budget` (alias: `--max-frames`) defaults to 128 and samples frames
   uniformly across elapsed video time. `--max-image-size` defaults to 512 and
   limits the longest side while preserving aspect ratio. `--frame-budget 0`
   explicitly uses all available frames. `--api-batch-size` controls images per
   request only in Responses API mode; Chat Completions sends all sampled frames
   in one request. The model is configured in
   `configs/qa_gpt6_luna.json`; set `OPENCODE_API_KEY` for its API backend.

2. Grade the answers:

   ```bash
   python grade_answers.py \
       --questions /path/to/questions.json \
       --answers answers.json \
       --output graded_results.json
   ```

The legacy `run_model_answers.py` script is retained for compatibility. Its
`--batch_size` mode sends every frame in batches and ignores its frame cap; use
the QA command above for normal evaluation. Responses API frame mode adds sampled
image batches to one stored conversation, then sends the full question list.
Chat Completions and Anthropic Messages frame modes initially send all sampled
images and the full question list in one request. If that request fails, they
retry with 50 fewer uniformly sampled frames, down to one frame.

`answers.json` contains model and input metadata, question order, an `answers`
list with each `question_id` and selected `answer_indices`, and the raw model
response. An answer that cannot be parsed is stored with `answer_indices: null`
and counted as wrong.

`graded_results.json` contains a `score` object (`correct`, `total`,
`accuracy`) and a `results` list. Each result has `correct` (true/false), the
`selected_indices` and `selected_answers` text, and the `correct_indices` and
`correct_answers` text. The score is also printed to the terminal.

To submit the QA and grading job on Gadi, run from the repository root:

```bash
qsub run_all_eval.sh
```

To submit the remaining OpenCode Go models while excluding the completed
GPT-6 Luna and Kimi K3 runs, use the helper script:

```bash
./submit_remaining_opencode_eval.sh
```

By default `run_all_eval.sh` runs Claude Haiku 5.5, MiMo-V2.6-Pro, and
GLM-5.3-Flash through OpenCode Go. It evaluates every sequence in its `LISTS`
array (17 Boreas sequences); leave `LIST` unset to use the full list. The helper
script separately sets Grok 4.7, Qwen3.8 Max, and DeepSeek V4 Flash Vision Exp.
The sequences are the active Boreas IDs
in the `LISTS` array in `run_all_eval.sh`, copied from the **Temporal** section of
`DATASET_PROGRESS.md` (`forest_dataset` is excluded). It reads camera frames from
`/g/data/pg06/FYP2026S1_3473/boreas_dataset/<list>/camera` and temporal questions
from `/g/data/pg06/FYP2026S1_3473/<list>/temporal_outputs/narratives/questions.json`.
Each sequence uses up to 512 frames by default, resized to a maximum side of
512 pixels. `MAX_FRAMES` can override the frame budget. Grok 4.7 uses the
Responses API and sends sampled frames in batches of 10 before sending the
questions in the same stored conversation. Qwen3.8 Max and DeepSeek V4 Flash
Vision Exp each send all sampled frames with the questions in one request.
Results are saved under the directory from which the job was submitted:

```text
model_answers/grok-4.7/<dataset>/temporal/
    question_answer.json
    graded_results.json
    final_mark.txt
```

`final_mark.txt` contains a score such as `10/20 (50.00%)`. Set
`BENCHMARK=sparse` to read `sparse_outputs/events/questions.json` for the same
dataset list and write results under `sparse/`. Set `LIST=boreas-2024-12-04-14-44` to
run just one dataset. `QUESTIONS`, `QUESTION_RELATIVE_PATH`, `MAX_FRAMES`, `MAX_IMAGE_SIZE`,
`API_BATCH_SIZE` (default: 10), and `SEED` are also overridable.

To choose different models, set `CONFIGS` to a space-separated list of config
paths. Each model runs on every selected dataset, one model after another. The
default `CONFIGS` value selects Claude Haiku 5.5, MiMo-V2.6-Pro, and
GLM-5.3-Flash; the default `LISTS` array covers the full Boreas set. For example,
to run only Grok 4.7 on the full list:

```bash
CONFIGS="configs/qa_grok_4_7.json" qsub run_all_eval.sh
```

Results for each model are stored under its own `model_answers/<model>/` folder.
Before calling a model, the script skips that model/dataset when non-empty answer,
graded-results, and final-mark files already exist, with the final mark newer than
the other two. It rejects API-reported errors and checks effective frame coverage
before grading; failed or incomplete runs continue to the next dataset.

Claude Haiku 5.5, MiMo-V2.6-Pro, and GLM-5.3-Flash are available as OpenCode Go
frame-input configs. Claude uses Anthropic Messages; MiMo and GLM-5.3-Flash use
Chat Completions. Use the same `OPENCODE_API_KEY` as the other OpenCode Go models:

```bash
qsub run_all_eval.sh
```

Set `LIST=boreas-2025-07-18-14-55` only when you want to restrict the default
three-model evaluation to one sequence. The PBS walltime is set to 72 hours for
the default full-list run; completed model/dataset evaluations are skipped when
the script is resubmitted.

GLM-5.3 itself is text-only, so this frame-based VLM run uses the vision-capable
GLM-5.3-Flash variant.

To run Gemini 3.8 Flash on one sequence with the same sampled-frame input as
the OpenCode Go models, add `GEMINI_API_KEY=your-key` to the repository-root
`.env` file (git-ignored), then submit the Gemini frame config:

```bash
CONFIGS="configs/qa_gemini_3_8_flash.json" \
LIST="boreas-2025-07-18-14-55" MAX_FRAMES=512 qsub run_all_eval.sh
```

The job sends up to 512 chronologically sampled camera frames, resized to a
512-pixel maximum side, in one Chat Completions request. Use the same sequence
and frame settings as the OpenCode Go run when comparing scores. After the
single-sequence run succeeds, omit `LIST` to evaluate the full configured set.
Gemini results are written under
`model_answers/gemini-3.8-flash/<dataset>/<benchmark>/`. This config uses
Google's OpenAI-compatible Gemini endpoint; `qa_gemini_video.json` is the
separate option for native MP4 upload, both standalone and through
`run_all_eval.sh`.

To evaluate Gemini on each original MP4 instead, use its native-video config:

```bash
CONFIGS="configs/qa_gemini_video.json" \
LIST="boreas-2025-07-18-14-55" qsub run_all_eval.sh
```

This reads `/g/data/pg06/FYP2026S1_3473/boreas_dataset/<sequence>/video.mp4`
and uploads it through Gemini's Files API. The video must already be downloaded.
Gemini's default static video processing samples about one frame per second;
`MAX_FRAMES` does not apply in native-video mode. Results are kept separate in
`model_answers/gemini-3.8-flash-native-video/<dataset>/<benchmark>/`.

## Run Vertex AI from Your Computer

Use this path when Gadi should prepare the frames but Vertex AI should be called
from your computer with your existing `gcloud` Application Default Credentials.
The Gadi job samples the same up-to-512 frames at a 512-pixel maximum side; it
does not make model calls or need a Google API key.

From the repository root on Gadi, submit the copyq job:

```bash
qsub prepare_vertex_eval.pbs
```

By default, it stages sparse-event questions for the same Boreas sequence list
as `run_all_eval.sh` under
`/g/data/pg06/FYP2026S1_3473/gemini_vertex_staging/<PBS_JOBID>/`; PBS output and
error logs also go to `/g/data/pg06/FYP2026S1_3473/`. Set `LIST` to prepare one
sequence, or override `BENCHMARK`, `QUESTIONS`, `MAX_FRAMES`, `MAX_IMAGE_SIZE`,
and `SEED`. Set `BENCHMARK=temporal` to stage temporal-order questions instead.
Wait for the PBS job to finish before starting the local worker.

On your computer, from your repository checkout, run:

```bash
python run_vertex_frame_eval.py \
    --ssh-target YOUR_GADI_SSH_ALIAS \
    --remote-stage /absolute/path/printed/by/the/job
```

Replace the SSH alias with the one that works with `ssh`/`scp` from your
computer, and use the staging path printed by the job. The local worker uses
the same Vertex project, location, and model as `test_vertex.py`
(`episodic-memory-benchmark`, `us-central1`, `gemini-2.5-flash`). It calls
Vertex directly using local ADC, not `GEMINI_API_KEY`. After each dataset's
request and grading finish, it deletes that dataset's temporary local frames
before pulling the next one. It also removes that dataset's staged Gadi copy
after success. Answer JSON, grading, and marks stay on your computer under
`model_answers/gemini-2.5-flash-vertex/<dataset>/<benchmark>/`. If needed, run
`gcloud auth application-default login` locally first. The worker reuses an SSH
control connection, so by default it prompts once for Gadi authentication. If
you fill the local-only `ssh_pass` placeholder in `run_vertex_frame_eval.py`,
the worker supplies it to SSH through a pseudo-terminal; never commit a real
password. On reruns it skips completed datasets; an oversized dataset is
uniformly reduced by 30 frames per retry, and its effective frame count is
recorded in the answer metadata.

To average a model's percentages across datasets with completed marks, run:

```bash
python3 average_model_mark.py deepseek-v4-flash-vision-exp --benchmark temporal
```

Each dataset is weighted equally. The benchmark defaults to `temporal`; use
`--benchmark sparse` to average sparse-event marks instead.

## Ask Questions from a Video

The `QA/` runner takes an MP4 or a folder of timestamp-named camera frames and a
generated `questions.json` (sparse-event or temporal schema). It shuffles the
questions, removes answer/evidence metadata, and asks the configured model to
answer the remaining questions. The model config selects one of two input modes:

```bash
# GPT-6 Luna: uniformly sampled frame images (default config)
python -m QA.run_questionnaire \
    --video /path/to/video.mp4 \
    --questions /path/to/questions.json \
    --output answers.json

# Gemini: upload and pass the original MP4
python -m QA.run_questionnaire \
    --video /path/to/video.mp4 \
    --questions /path/to/questions.json \
    --config configs/qa_gemini_video.json \
    --output answers.json

# GPT-6 Luna: use an existing camera frame folder
python -m QA.run_questionnaire \
    --frames-dir /path/to/camera \
    --questions /path/to/questions.json \
    --output answers.json

# Gemini 3.8 Flash: same sampled camera-frame input used by run_all_eval.sh
python -m QA.run_questionnaire \
    --frames-dir /path/to/camera \
    --questions /path/to/questions.json \
    --config configs/qa_gemini_3_8_flash.json \
    --frame-budget 512 \
    --max-image-size 512 \
    --output answers.json
```

`configs/qa_gpt6_luna.json` remains the default for a direct
`QA.run_questionnaire` invocation; `run_all_eval.sh` defaults to the Qwen3.8 Max
config, while `submit_remaining_opencode_eval.sh` selects Grok 4.7, Qwen3.8 Max,
and DeepSeek V4 Flash Vision Exp through OpenCode Go, using
`OPENCODE_API_KEY`. Claude Haiku 5.5, MiMo-V2.6-Pro, and GLM-5.3-Flash are also
available through their configs and use `OPENCODE_API_KEY`; the Gemini frame
config uses `GEMINI_API_KEY`. GLM-5.3 is text-only, so the frame-input config
uses GLM-5.3-Flash. Frame mode
deterministically samples up to 128 images from
the MP4 or camera folder, uniformly across elapsed video time, and resizes each
to a longest side of at most 512 pixels while preserving aspect ratio. Use
`--frame-budget` (alias: `--max-frames`) to change the maximum number of frames;
`--max-image-size` adjusts the resize limit. `--frame-budget 0` uses every frame.
The model config's `api_protocol` selects `responses`, `chat_completions`, or
`anthropic_messages`. Responses API configs send sampled frames in
stored-conversation batches and then send the questions. Chat Completions and
Anthropic Messages configs send all sampled frames with the questions in one
request; `api_batch_size` applies only to Responses API mode. Single-request
configs can set `api_max_retries`; after exhausting that many retries on a 5xx
error, the runner reduces the uniform frame sample by 50 and retries down to 50
frames. Responses API runs also restart with a smaller uniform sample if OpenCode
returns `Response is too large to store`. Output metadata records the initial
and effective frame budgets and reductions.
The Gemini frame config uses Google's OpenAI-compatible endpoint, sets
`api_max_tokens_parameter` to `max_tokens`, and authenticates with
`GEMINI_API_KEY`.
`configs/qa_gemini_video.json`
uses `GEMINI_API_KEY` and the
Gemini Files API to upload the original MP4; the runner waits for provider-side
video processing and does not decode frames locally. Gemini's default static
video processing samples at about 1 FPS. Set `input_mode` in the
model config, or override it with `--input-mode frames` or
`--input-mode native_video`. Native video input currently supports local
Qwen3-VL through vLLM/`qwen-vl-utils` and Gemini through its Files API. Use
`--seed` to make question ordering reproducible. If an API context limit is
reached while adding Responses API batches, the runner stops and records the
number of frames accepted before asking the questions.

The output JSON records `input_mode`, input metadata, question order, parsed
answer indices, and the raw model response. The model-facing questionnaire
contains only `type`, `question`, and `options`; it omits IDs, answers,
event/frame evidence, false-event flags, and the top-level `false_events` list.
The model returns one positional list per question (for example,
`[[0], [1], [3, 0, 1, 2]]`); the runner maps those positions back to question IDs
for grading. Option indexes are 0-based. If an MP4 contains audio, the prompt
instructs the model to ignore it; use an audio-free file for strict vision-only
scoring.

## Run the Qwen3 QA Test on Gadi

`run_model_eval.sh` submits the QA runner and grader to the four-GPU `gpuhopper`
queue using `configs/qa_qwen3.json`. Qwen3-VL supports video input, so this
first test passes the MP4 through its vLLM video processor instead of sampling
frames in the QA script. The config requests 2 FPS and caps processing at 768
frames. It defaults to Boreas sequence
`boreas-2025-02-15-16-58`, reads the repository-root `questions.json`, and uses
the sequence's `video.mp4`. Outputs go under the submit directory, grouped by
model, task, and sequence:

```text
model_answers/Qwen-Qwen3-VL-235B-A22B-Instruct-FP8/temporal/<sequence>/
    answers.json
    results.json
```

Submit from the repository root:

```bash
qsub run_model_eval.sh
```

The frame-mode fallback defaults to 128 frames with a maximum side of 512
pixels. The video must already exist at
`/g/data/pg06/FYP2026S1_3473/boreas_dataset/<sequence>/video.mp4`; submit
`download_boreas_videos.pbs` first if needed. Since the PBS script exports the
submitter's environment, you can override `LIST`, `QUESTIONS`, `MAX_FRAMES`,
`MAX_IMAGE_SIZE`, `INPUT_MODE`, `CONFIG`, or `SEED`. Set `INPUT_MODE=frames` to
compare against externally sampled images:

```bash
INPUT_MODE=frames MAX_FRAMES=32 qsub run_model_eval.sh
```

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
2. **Review:** Detections are parsed, sorted in Python by the frame
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
   a real event and three are invented false events. False events are invented
   per question and never reused across questions, so each question has its
   own incorrect answers; every false event is listed once in `questions.json`
   as a `false_events` array entry tagged with its `question_id` for review.

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
2. **Timeline:** Section events are parsed, sorted in Python by the frame
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

The batch scripts (`run_all_local.sh`, `run_all_api.sh`) rewrite the per-list
`frames_dir`, `sections_dir`, and `output` paths before each run.
`run_all_api.sh` uses `configs/temporal_events.json`, forces `backend: "api"`
and `model: "gpt-6-luna"`, and runs `--mode temporal --stage questions` on
`copyq` with 8 GB of memory. Each list must already have a human-reviewed
`timeline.json` in `<list>/temporal_outputs/narratives/`; the script writes
`questions.json` alongside it. `run_all_local.sh` reads `backend`, `model`,
and every other inference setting from the configuration file, so set those
there before submitting a local batch job.

`forest_run.sh` is currently configured to run the local
`Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` model for sparse-event extraction and
review on forest K-01 (`K-01_data`). Its active PBS header requests four GPUs
from `gpuhopper`. The API and temporal run blocks remain as inactive examples.
The script updates the sparse config for K-01 and restores the original config
when it exits.


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
| `backend` | Inference backend: `local` for vLLM or `api` for an OpenAI-compatible or Anthropic Messages API. Choose `local` for an available local model and GPUs, otherwise use `api`. |
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
| `image_max_size` | Longest side, in pixels, of frames written into sections. Frames are downscaled to fit (never upscaled). `null` keeps the original dimensions. Use a small value, such as `1024`, to keep API request payloads small and stay within the vision-token budget of models such as `grok-4.6`; omit it for local vLLM runs that accept full-resolution frames. |
| `image_quality` | JPEG quality, from `1` to `95`, used when frames are re-encoded for sections. The supplied sparse value is `80`. Setting this alone converts frames to JPEG without resizing; if both image settings are omitted, frames are copied unchanged. |
| `move` | Controls whether input frames are moved or copied into sections. Leave it `false` or omit it unless the source frames can be removed. |
| `model` | Model used for extraction and later sparse stages. Pick a model that accepts the selected backend and image inputs. |
| `backend` | `local` for vLLM or `api` for a supported API protocol. The configuration selects the backend for a run; the batch scripts in `scripts/` do not override it. The backend must match the model and available infrastructure. |
| `api_base_url` | API base URL for the selected protocol. Configure this only when using `backend: "api"`. |
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
| `image_max_size` | Longest side, in pixels, of frames written into sections. Frames are downscaled to fit (never upscaled). The supplied `null` keeps the original dimensions. |
| `image_quality` | JPEG quality, from `1` to `95`, used when frames are re-encoded for sections. The supplied value `80` converts sampled PNG frames to JPEG without resizing. |
| `move` | Controls whether input frames are moved or copied into sections. Leave it `false` or omit it unless the source frames can be removed. |
| `model` | Model used for extraction and later temporal stages. The supplied configuration uses `Qwen/Qwen2.5-VL-72B-Instruct` locally. |
| `backend` | Use `local` for the supplied four-GPU vLLM configuration or `api` for a supported API model. The backend must match the model and available infrastructure. |
| `api_base_url` | API base URL for the selected protocol. Configure this only when using `backend: "api"`. |
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

OpenCode Go API requests include a stable per-run `x-opencode-session` header,
as required for third-party clients; other OpenAI-compatible endpoints do not
receive that OpenCode-specific header. Calls are sent sequentially, one at a
time. Connection/time-out errors and
HTTP 408, 409, 429, and 5xx errors are retried with exponential backoff and
jitter, capped at five minutes unless the server's `Retry-After` header asks
for a longer delay. Retries continue indefinitely unless `api_max_retries` is
set in the model config. For single-request frame modes, each failed request
triggers uniform frame reduction by 50 frames, down to one frame. For Responses
API frame mode, the specific `Response is too large to store` error restarts
frame ingestion with a smaller uniform sample. Each retry logs the model,
endpoint, status, response body, and provider request ID when available. Other
errors, such as an invalid API key or an explicit exhausted-quota response, are
not retried by the API client; single-request frame modes still reduce the frame
count after each such failed request.
Gadi's PBS walltime still applies, so a job can be terminated by the scheduler
if an outage lasts beyond its requested walltime.

## Frame Sections

`FrameParser` accepts `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp` files. It sorts frames by their timestamp filename and samples each section with `step` as the stride between frames up to `frames_per_section` frames; the next section starts `skip` frames after the last sampled frame (defaulting to `step`, so sampling continues without gaps). It groups the sampled frames into directories such as:

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
quality, downscaled so the longest side fits `image_max_size`. Timestamp
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
temporal events into elapsed seconds. It uses the earliest frame timestamp in
`frames_dir` as time zero. Two filename formats are supported: Boreas-style
epoch microseconds (`1733343593917869.png`) and WildScenes-style
`<epoch-seconds>.<fraction>` (`1624325785.833297127.png`); both are parsed
by `parse_frame_timestamp`.

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
directory must be available, and frame names must contain timestamps.
For temporal runs this conversion is redundant: per-section seconds are already
computed at frame-parsing time and attached to every event.

During VLM extraction, the ordered filenames from each section are included in
the prompt so temporal events can refer to the original frame files instead of
invented labels.

## Full Documentation

See [CODEBASE_DOCUMENTATION.md](CODEBASE_DOCUMENTATION.md) for a detailed breakdown of the codebase and implementation status.
