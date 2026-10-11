#!/bin/bash
#PBS -N model_eval_gpt6_luna
#PBS -P pg06
#PBS -q copyq
#PBS -l ncpus=1
#PBS -l mem=8GB
#PBS -l walltime=06:45:00
#PBS -l storage=scratch/pg06+gdata/pg06
#PBS -l wd
#PBS -V

set -euo pipefail

module load python3/3.11.7

cd "$PBS_O_WORKDIR"
source /scratch/pg06/vm4618/envs/vllm_env/bin/activate

BASE="/g/data/pg06/FYP2026S1_3473"
LISTS=(
  "boreas-2025-02-15-16-58"
)
QUESTIONS_OVERRIDE="${QUESTIONS:-}"
QUESTION_RELATIVE_PATH="${QUESTION_RELATIVE_PATH:-temporal_outputs/narratives/questions.json}"
CONFIG="${CONFIG:-configs/qa_gpt6_luna.json}"
MAX_FRAMES="${MAX_FRAMES:-128}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-512}"
API_BATCH_SIZE="${API_BATCH_SIZE:-20}"
SEED="${SEED:-42}"
BENCHMARK="${BENCHMARK:-temporal}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Model config not found: $CONFIG" >&2
  exit 1
fi
MODEL_NAME="$(python3 - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text())["model"])
PY
)"
MODEL_FOLDER="${MODEL_NAME//\//-}"

if [[ -z "${OPENCODE_API_KEY:-}" && ! -f .env ]]; then
  echo "Set OPENCODE_API_KEY or provide it through the repository .env file" >&2
  exit 1
fi

for LIST in "${LISTS[@]}"; do
  CAMERA_DIR="$BASE/boreas_dataset/$LIST/camera"
  if [[ -n "$QUESTIONS_OVERRIDE" ]]; then
    QUESTIONS="$QUESTIONS_OVERRIDE"
  else
    QUESTIONS="$BASE/$LIST/$QUESTION_RELATIVE_PATH"
  fi
  OUTPUT_DIR="$PBS_O_WORKDIR/model_answers/$MODEL_FOLDER/$BENCHMARK/$LIST"
  ANSWERS="$OUTPUT_DIR/answers.json"
  GRADED_RESULTS="$OUTPUT_DIR/results.json"

  for INPUT_FILE in "$QUESTIONS"; do
    if [[ ! -f "$INPUT_FILE" ]]; then
      echo "Question file not found for $LIST: $INPUT_FILE" >&2
      exit 1
    fi
  done
  if [[ ! -d "$CAMERA_DIR" ]]; then
    echo "Camera frame folder not found for $LIST: $CAMERA_DIR" >&2
    exit 1
  fi
  echo "=============================================="
  echo "Running GPT-6 Luna API questionnaire evaluation: $LIST"
  echo "Camera frames: $CAMERA_DIR"
  echo "Questions: $QUESTIONS"
  echo "Output: $OUTPUT_DIR"
  if [[ "$MAX_FRAMES" -eq 0 ]]; then
    echo "Frames: all camera frames (max side $MAX_IMAGE_SIZE px)"
  else
    echo "Uniformly sampled frames: $MAX_FRAMES (max side $MAX_IMAGE_SIZE px)"
  fi
  echo "Images per API request: $API_BATCH_SIZE"
  echo "=============================================="

  python3 -m QA.run_questionnaire \
    --frames-dir "$CAMERA_DIR" \
    --questions "$QUESTIONS" \
    --config "$CONFIG" \
    --input-mode frames \
    --max-frames "$MAX_FRAMES" \
    --max-image-size "$MAX_IMAGE_SIZE" \
    --api-batch-size "$API_BATCH_SIZE" \
    --seed "$SEED" \
    --output "$ANSWERS"

  python3 - "$ANSWERS" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
metadata = result.get("input_metadata", {})
sampled = metadata.get("frames_sampled")
received = metadata.get("frames_received")
if sampled is None or received is None:
    raise SystemExit("Missing API frame coverage metadata; refusing to grade")
if received != sampled:
    raise SystemExit(
        f"Only {received}/{sampled} sampled frames were accepted; "
        "refusing to grade this partial-context run"
    )
PY

  python3 grade_answers.py \
    --questions "$QUESTIONS" \
    --answers "$ANSWERS" \
    --output "$GRADED_RESULTS"
done

echo "All GPT-6 Luna evaluation lists done."
