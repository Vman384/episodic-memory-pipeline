#!/bin/bash
#PBS -N new-models_temporal
#PBS -P pg06
#PBS -q copyq
#PBS -l ncpus=1
#PBS -l mem=8GB
#PBS -l walltime=3:00:00
#PBS -l storage=scratch/pg06+gdata/pg06
#PBS -l wd
#PBS -V

set -euo pipefail

module load python3/3.11.7

cd "$PBS_O_WORKDIR"
source /scratch/pg06/vm4618/envs/vllm_env/bin/activate

BASE="/g/data/pg06/FYP2026S1_3473"
BENCHMARK="${BENCHMARK:-temporal}"
CONFIGS="${CONFIGS:-configs/qa_kimi_k3.json}"

MAX_FRAMES="${MAX_FRAMES:-512}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-512}"
API_BATCH_SIZE="${API_BATCH_SIZE:-10}"
SEED="${SEED:-42}"
QUESTIONS_OVERRIDE="${QUESTIONS:-}"

case "$BENCHMARK" in
  temporal)
    DEFAULT_QUESTION_RELATIVE_PATH="temporal_outputs/narratives/questions.json"
    ;;
  sparse)
    DEFAULT_QUESTION_RELATIVE_PATH="sparse_outputs/events/questions.json"
    ;;
  *)
    echo "BENCHMARK must be 'temporal' or 'sparse'" >&2
    exit 1
    ;;
esac
QUESTION_RELATIVE_PATH="${QUESTION_RELATIVE_PATH:-$DEFAULT_QUESTION_RELATIVE_PATH}"

# CONFIGS is a space-separated list of API/native-video model configs, e.g.
# CONFIGS="configs/qa_grok_4_7.json configs/qa_gemini_video.json"
# The default evaluates Claude Haiku 5.5, MiMo-V2.6-Pro, and GLM-5.3-Flash.
# Override CONFIGS with a space-separated list of config paths to select models.
read -r -a CONFIG_LIST <<< "$CONFIGS"
if ((${#CONFIG_LIST[@]} == 0)); then
  echo "CONFIGS must list at least one model config" >&2
  exit 1
fi

MODEL_NAMES=()
MODEL_FOLDERS=()
API_PROTOCOLS=()
API_KEY_ENVS=()
INPUT_MODES=()
for CONFIG in "${CONFIG_LIST[@]}"; do
  if [[ ! -f "$CONFIG" ]]; then
    echo "Model config not found: $CONFIG" >&2
    exit 1
  fi
  mapfile -t CONFIG_SETTINGS < <(python3 - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
print(config.get("model", ""))
print(config.get("backend", "local").lower())
print(config.get("api_protocol", "responses").lower())
print(config.get("api_key_env", "OPENCODE_API_KEY"))
print(config.get("input_mode", "frames").lower())
PY
  )
  MODEL_NAME="${CONFIG_SETTINGS[0]:-}"
  BACKEND="${CONFIG_SETTINGS[1]:-}"
  API_PROTOCOL="${CONFIG_SETTINGS[2]:-responses}"
  API_KEY_ENV="${CONFIG_SETTINGS[3]:-OPENCODE_API_KEY}"
  INPUT_MODE="${CONFIG_SETTINGS[4]:-frames}"
  if [[ -z "$MODEL_NAME" ]]; then
    echo "Model is missing from config: $CONFIG" >&2
    exit 1
  fi
  if [[ "$INPUT_MODE" == "frames" && "$BACKEND" != "api" ]]; then
    echo "Frame input requires an API model configuration: $CONFIG" >&2
    exit 1
  fi
  if [[ "$INPUT_MODE" == "native_video" && "$BACKEND" != "gemini" ]]; then
    echo "Native-video batch input requires the Gemini backend: $CONFIG" >&2
    exit 1
  fi
  if [[ "$INPUT_MODE" != "frames" && "$INPUT_MODE" != "native_video" ]]; then
    echo "Unsupported input_mode in config: $CONFIG" >&2
    exit 1
  fi
  if [[ -z "${!API_KEY_ENV:-}" && ! -f .env ]]; then
    echo "Set $API_KEY_ENV or provide it through the repository .env file" >&2
    exit 1
  fi
  MODEL_NAMES+=("$MODEL_NAME")
  MODEL_FOLDER="${MODEL_NAME//\//-}"
  if [[ "$INPUT_MODE" == "native_video" ]]; then
    MODEL_FOLDER+="-native-video"
  fi
  MODEL_FOLDERS+=("$MODEL_FOLDER")
  API_PROTOCOLS+=("$API_PROTOCOL")
  API_KEY_ENVS+=("$API_KEY_ENV")
  INPUT_MODES+=("$INPUT_MODE")
done

# Boreas sequences from the Temporal section of DATASET_PROGRESS.md.

LISTS=(
  "boreas-2025-07-18-14-55"
  "boreas-2024-12-03-13-13"
  "boreas-2024-12-03-13-34"
  "boreas-2024-12-04-11-45"
  "boreas-2024-12-04-11-56"
  "boreas-2025-07-18-15-12"
  "boreas-2024-12-04-14-44"
  "boreas-2025-02-15-16-58"
  "boreas-2025-07-18-11-53"
  "boreas-2024-12-04-14-34"
  "boreas-2024-12-04-14-38"
  "boreas-2025-02-21-14-51"
  "boreas-2025-02-15-17-19"
  "boreas-2024-12-04-14-59"
  "boreas-2025-08-13-10-36"
)

if [[ -n "${LIST:-}" ]]; then
  LISTS=("$LIST")
fi

is_evaluation_complete() {
  local model_folder="$1"
  local dataset_name="$2"
  local output_dir="$PBS_O_WORKDIR/model_answers/$model_folder/$dataset_name/$BENCHMARK"
  local answers="$output_dir/question_answer.json"
  local graded_results="$output_dir/graded_results.json"
  local final_mark="$output_dir/final_mark.txt"

  # final_mark is written after grading; require it to be newer than both inputs.
  [[ -s "$answers" && -s "$graded_results" && -s "$final_mark" \
    && "$final_mark" -nt "$answers" && "$final_mark" -nt "$graded_results" ]]
}

# Check inputs only for datasets that still need at least one model run.
for LIST in "${LISTS[@]}"; do
  DATASET_NAME="${LIST#boreas-}"
  NEEDS_CAMERA=0
  NEEDS_VIDEO=0
  for INDEX in "${!MODEL_FOLDERS[@]}"; do
    MODEL_FOLDER="${MODEL_FOLDERS[$INDEX]}"
    if ! is_evaluation_complete "$MODEL_FOLDER" "$DATASET_NAME"; then
      if [[ "${INPUT_MODES[$INDEX]}" == "native_video" ]]; then
        NEEDS_VIDEO=1
      else
        NEEDS_CAMERA=1
      fi
    fi
  done
  if ((NEEDS_CAMERA == 0 && NEEDS_VIDEO == 0)); then
    echo "All configured models already evaluated for $LIST; skipping input checks"
    continue
  fi

  CAMERA_DIR="$BASE/boreas_dataset/$LIST/camera"
  VIDEO_FILE="$BASE/boreas_dataset/$LIST/video.mp4"
  QUESTIONS="$QUESTIONS_OVERRIDE"
  if [[ -z "$QUESTIONS" ]]; then
    QUESTIONS="$BASE/$LIST/$QUESTION_RELATIVE_PATH"
  fi

  if ((NEEDS_CAMERA == 1)) && [[ ! -d "$CAMERA_DIR" ]]; then
    echo "Camera frame folder not found for $LIST: $CAMERA_DIR" >&2
    exit 1
  fi
  if ((NEEDS_VIDEO == 1)) && [[ ! -f "$VIDEO_FILE" ]]; then
    echo "Native video file not found for $LIST: $VIDEO_FILE" >&2
    exit 1
  fi
  if [[ ! -f "$QUESTIONS" ]]; then
    echo "Question file not found for $LIST: $QUESTIONS" >&2
    exit 1
  fi
done

echo "Models: ${MODEL_NAMES[*]}"
echo "Benchmark: $BENCHMARK"
echo "Datasets: ${#LISTS[@]} Boreas sequences"
if [[ " ${INPUT_MODES[*]} " == *" frames "* ]]; then
  echo "Frame budget: $MAX_FRAMES; max image side: $MAX_IMAGE_SIZE px"
fi

FAILED_LISTS=()
for INDEX in "${!CONFIG_LIST[@]}"; do
  CONFIG="${CONFIG_LIST[$INDEX]}"
  MODEL_NAME="${MODEL_NAMES[$INDEX]}"
  MODEL_FOLDER="${MODEL_FOLDERS[$INDEX]}"
  API_PROTOCOL="${API_PROTOCOLS[$INDEX]}"
  API_KEY_ENV="${API_KEY_ENVS[$INDEX]}"
  INPUT_MODE="${INPUT_MODES[$INDEX]}"

  echo "----------------------------------------------"
  echo "Model: $MODEL_NAME ($CONFIG)"
  echo "Input mode: $INPUT_MODE"
  echo "API key environment variable: $API_KEY_ENV"
  if [[ "$INPUT_MODE" == "native_video" ]]; then
    echo "Gemini Files API uploads the original MP4"
  else
    echo "API protocol: $API_PROTOCOL"
    case "$API_PROTOCOL" in
      responses)
        echo "Responses API batch size: $API_BATCH_SIZE"
        ;;
      chat_completions)
        echo "Chat Completions sends all sampled frames in one request"
        ;;
      anthropic_messages)
        echo "Anthropic Messages sends all sampled frames in one request"
        ;;
      *)
        echo "Unsupported API protocol in config: $API_PROTOCOL" >&2
        exit 1
        ;;
    esac
  fi

  for LIST in "${LISTS[@]}"; do
    CAMERA_DIR="$BASE/boreas_dataset/$LIST/camera"
    VIDEO_FILE="$BASE/boreas_dataset/$LIST/video.mp4"
    QUESTIONS="$QUESTIONS_OVERRIDE"
    if [[ -z "$QUESTIONS" ]]; then
      QUESTIONS="$BASE/$LIST/$QUESTION_RELATIVE_PATH"
    fi

    DATASET_NAME="${LIST#boreas-}"
    OUTPUT_DIR="$PBS_O_WORKDIR/model_answers/$MODEL_FOLDER/$DATASET_NAME/$BENCHMARK"
    ANSWERS="$OUTPUT_DIR/question_answer.json"
    GRADED_RESULTS="$OUTPUT_DIR/graded_results.json"
    FINAL_MARK="$OUTPUT_DIR/final_mark.txt"

    if is_evaluation_complete "$MODEL_FOLDER" "$DATASET_NAME"; then
      echo "Skipping $MODEL_NAME for $LIST; completed answers already exist"
      continue
    fi

    echo "=============================================="
    echo "Running $MODEL_NAME $BENCHMARK evaluation for: $LIST"
    echo "Questions: $QUESTIONS"
    echo "Output: $OUTPUT_DIR"
    echo "=============================================="

    QA_ARGS=(
      --questions "$QUESTIONS"
      --config "$CONFIG"
      --input-mode "$INPUT_MODE"
      --seed "$SEED"
      --output "$ANSWERS"
    )
    if [[ "$INPUT_MODE" == "native_video" ]]; then
      QA_ARGS+=(--video "$VIDEO_FILE")
    else
      QA_ARGS+=(
        --frames-dir "$CAMERA_DIR"
        --frame-budget "$MAX_FRAMES"
        --max-image-size "$MAX_IMAGE_SIZE"
        --api-batch-size "$API_BATCH_SIZE"
      )
    fi

    if ! python3 -m QA.run_questionnaire "${QA_ARGS[@]}"; then
      echo "QA failed for $MODEL_NAME on $LIST; continuing with the next dataset" >&2
      FAILED_LISTS+=("$MODEL_NAME/$LIST")
      continue
    fi

    if ! python3 - "$ANSWERS" "$INPUT_MODE" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
metadata = result.get("input_metadata", {})
error = metadata.get("error") or metadata.get("question_error")
if error:
    raise SystemExit(f"QA recorded an API error; refusing to grade: {error}")
if sys.argv[2] == "native_video":
    if metadata.get("provider") != "gemini" or metadata.get("uploaded_file_state") != "ACTIVE":
        raise SystemExit("Missing successful Gemini video-upload metadata; refusing to grade")
    raise SystemExit(0)
sampled = metadata.get("frames_sampled", metadata.get("frames_used"))
received = metadata.get("frames_received", metadata.get("frames_used"))
if sampled is None or received is None:
    raise SystemExit("Missing frame coverage metadata; refusing to grade")
if received != sampled:
    raise SystemExit(
        f"Only {received}/{sampled} sampled frames were accepted; refusing to grade"
    )
PY
    then
      echo "QA error or incomplete frame coverage for $MODEL_NAME on $LIST; keeping answers and continuing" >&2
      FAILED_LISTS+=("$MODEL_NAME/$LIST")
      continue
    fi

    if ! python3 grade_answers.py \
      --questions "$QUESTIONS" \
      --answers "$ANSWERS" \
      --output "$GRADED_RESULTS"; then
      echo "Grading failed for $MODEL_NAME on $LIST; continuing with the next dataset" >&2
      FAILED_LISTS+=("$MODEL_NAME/$LIST")
      continue
    fi

    python3 - "$GRADED_RESULTS" "$FINAL_MARK" <<'PY'
import json
import sys
from pathlib import Path

graded = json.loads(Path(sys.argv[1]).read_text())
score = graded["score"]
mark = (
    f"{score['correct']}/{score['total']} "
    f"({score['accuracy'] * 100:.2f}%)"
)
Path(sys.argv[2]).write_text(mark + "\n")
print(f"Final mark: {mark}")
PY
  done
done

if ((${#FAILED_LISTS[@]} > 0)); then
  echo "Evaluation completed with failures: ${FAILED_LISTS[*]}" >&2
  exit 1
fi

echo "All $BENCHMARK Boreas evaluations completed successfully."
