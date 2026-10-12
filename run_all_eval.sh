#!/bin/bash
#PBS -N kimi-k3_temporal
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
CONFIG="${CONFIG:-configs/qa_kimi_k3.json}"
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
PY
)
MODEL_NAME="${CONFIG_SETTINGS[0]:-}"
BACKEND="${CONFIG_SETTINGS[1]:-}"
API_PROTOCOL="${CONFIG_SETTINGS[2]:-responses}"
if [[ -z "$MODEL_NAME" ]]; then
  echo "Model is missing from config: $CONFIG" >&2
  exit 1
fi
if [[ "$BACKEND" != "api" ]]; then
  echo "run_all_eval.sh requires an API model configuration" >&2
  exit 1
fi
MODEL_FOLDER="${MODEL_NAME//\//-}"

if [[ -z "${OPENCODE_API_KEY:-}" && ! -f .env ]]; then
  echo "Set OPENCODE_API_KEY or provide it through the repository .env file" >&2
  exit 1
fi

# Boreas sequences from the Temporal section of DATASET_PROGRESS.md.

LISTS=(
#  "boreas-2025-07-18-14-55"
#  "boreas-2024-12-03-13-13"
#  "boreas-2024-12-03-13-34"
#  "boreas-2024-12-04-11-45"
#  "boreas-2024-12-04-11-56"
#  "boreas-2025-07-18-15-12"
#  "boreas-2024-12-04-14-44"
  "boreas-2025-02-15-16-58"
  "boreas-2025-07-18-11-53"
  "boreas-2024-12-04-14-34"
  "boreas-2024-12-04-14-38"
  "boreas-2025-02-21-14-51"
  "boreas-2025-02-15-17-19"
  "boreas-2024-12-04-14-59"
  "boreas-2025-07-18-10-33"
  "boreas-2025-02-22-12-26"
  "boreas-2025-08-13-10-36"
)

if [[ -n "${LIST:-}" ]]; then
  LISTS=("$LIST")
fi

# Check every input before starting any paid API calls.
for LIST in "${LISTS[@]}"; do
  CAMERA_DIR="$BASE/boreas_dataset/$LIST/camera"
  QUESTIONS="$QUESTIONS_OVERRIDE"
  if [[ -z "$QUESTIONS" ]]; then
    QUESTIONS="$BASE/$LIST/$QUESTION_RELATIVE_PATH"
  fi

  if [[ ! -d "$CAMERA_DIR" ]]; then
    echo "Camera frame folder not found for $LIST: $CAMERA_DIR" >&2
    exit 1
  fi
  if [[ ! -f "$QUESTIONS" ]]; then
    echo "Question file not found for $LIST: $QUESTIONS" >&2
    exit 1
  fi
done

echo "Model: $MODEL_NAME"
echo "Benchmark: $BENCHMARK"
echo "Datasets: ${#LISTS[@]} Boreas sequences"
echo "Frame budget: $MAX_FRAMES; max image side: $MAX_IMAGE_SIZE px"
echo "API protocol: $API_PROTOCOL"
if [[ "$API_PROTOCOL" == "responses" ]]; then
  echo "Responses API batch size: $API_BATCH_SIZE"
else
  echo "Chat Completions sends all sampled frames in one request"
fi

FAILED_LISTS=()
for LIST in "${LISTS[@]}"; do
  CAMERA_DIR="$BASE/boreas_dataset/$LIST/camera"
  QUESTIONS="$QUESTIONS_OVERRIDE"
  if [[ -z "$QUESTIONS" ]]; then
    QUESTIONS="$BASE/$LIST/$QUESTION_RELATIVE_PATH"
  fi

  DATASET_NAME="${LIST#boreas-}"
  OUTPUT_DIR="$PBS_O_WORKDIR/model_answers/$MODEL_FOLDER/$DATASET_NAME/$BENCHMARK"
  ANSWERS="$OUTPUT_DIR/question_answer.json"
  GRADED_RESULTS="$OUTPUT_DIR/graded_results.json"
  FINAL_MARK="$OUTPUT_DIR/final_mark.txt"

  echo "=============================================="
  echo "Running $MODEL_NAME $BENCHMARK evaluation for: $LIST"
  echo "Questions: $QUESTIONS"
  echo "Output: $OUTPUT_DIR"
  echo "=============================================="

  if ! python3 -m QA.run_questionnaire \
    --frames-dir "$CAMERA_DIR" \
    --questions "$QUESTIONS" \
    --config "$CONFIG" \
    --input-mode frames \
    --frame-budget "$MAX_FRAMES" \
    --max-image-size "$MAX_IMAGE_SIZE" \
    --api-batch-size "$API_BATCH_SIZE" \
    --seed "$SEED" \
    --output "$ANSWERS"; then
    echo "QA failed for $LIST; continuing with the next dataset" >&2
    FAILED_LISTS+=("$LIST")
    continue
  fi

  if ! python3 - "$ANSWERS" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
metadata = result.get("input_metadata", {})
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
    echo "Incomplete frame coverage for $LIST; keeping answers and continuing" >&2
    FAILED_LISTS+=("$LIST")
    continue
  fi

  if ! python3 grade_answers.py \
    --questions "$QUESTIONS" \
    --answers "$ANSWERS" \
    --output "$GRADED_RESULTS"; then
    echo "Grading failed for $LIST; continuing with the next dataset" >&2
    FAILED_LISTS+=("$LIST")
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

if ((${#FAILED_LISTS[@]} > 0)); then
  echo "Evaluation completed with failures: ${FAILED_LISTS[*]}" >&2
  exit 1
fi

echo "All $BENCHMARK Boreas evaluations completed successfully."
