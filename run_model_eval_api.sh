#!/bin/bash
#PBS -N model_eval_gpt6_luna
#PBS -P pg06
#PBS -q copyq
#PBS -l ncpus=1
#PBS -l mem=8GB
#PBS -l walltime=03:30:00
#PBS -l storage=scratch/pg06+gdata/pg06
#PBS -l wd
#PBS -V

set -euo pipefail

module load python3/3.11.7

cd "$PBS_O_WORKDIR"
source /scratch/pg06/vm4618/envs/vllm_env/bin/activate

BASE="/g/data/pg06/FYP2026S1_3473"
LIST="${LIST:-boreas-2025-02-15-16-58}"
CAMERA_DIR="$BASE/boreas_dataset/$LIST/camera"
QUESTIONS="${QUESTIONS:-$PBS_O_WORKDIR/questions.json}"
CONFIG="${CONFIG:-configs/qa_gpt6_luna.json}"
MAX_FRAMES="${MAX_FRAMES:-100}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-768}"
API_BATCH_SIZE="${API_BATCH_SIZE:-20}"
SEED="${SEED:-42}"

OUTPUT_DIR="$BASE/$LIST/eval_outputs"
ANSWERS="$OUTPUT_DIR/gpt6_luna_answers.json"
GRADED_RESULTS="$OUTPUT_DIR/gpt6_luna_graded_results.json"

for INPUT_FILE in "$QUESTIONS" "$CONFIG"; do
  if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Required input file not found: $INPUT_FILE" >&2
    exit 1
  fi
done
if [[ ! -d "$CAMERA_DIR" ]]; then
  echo "Camera frame folder not found: $CAMERA_DIR" >&2
  exit 1
fi

if [[ -z "${OPENCODE_API_KEY:-}" && ! -f .env ]]; then
  echo "Set OPENCODE_API_KEY or provide it through the repository .env file" >&2
  exit 1
fi

echo "=============================================="
echo "Running GPT-6 Luna API questionnaire evaluation"
echo "Drive: $LIST"
echo "Camera frames: $CAMERA_DIR"
echo "Questions: $QUESTIONS"
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

echo "Model evaluation complete."
