#!/bin/bash
#PBS -N gpt6_luna_eval
#PBS -P pg06
#PBS -q copyq
#PBS -l ncpus=1
#PBS -l mem=8GB
#PBS -l walltime=01:00:00
#PBS -l storage=scratch/pg06+gdata/pg06
#PBS -l wd
#PBS -V

set -euo pipefail

module load python3/3.11.7

# cd to the right directory
cd "$PBS_O_WORKDIR"

# Enter the environment containing the pipeline dependencies.
source /scratch/pg06/vm4618/envs/vllm_env/bin/activate

BASE="/g/data/pg06/FYP2026S1_3473"
LIST="${LIST:-boreas-2024-12-04-14-44}"
BENCHMARK="${BENCHMARK:-temporal}"
CONFIG="${CONFIG:-configs/qa_gpt6_luna.json}"
MAX_FRAMES="${MAX_FRAMES:-256}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-512}"
API_BATCH_SIZE="${API_BATCH_SIZE:-20}"
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
CAMERA_DIR="$BASE/boreas_dataset/$LIST/camera"
QUESTIONS="$QUESTIONS_OVERRIDE"
if [[ -z "$QUESTIONS" ]]; then
  QUESTIONS="$BASE/$LIST/$QUESTION_RELATIVE_PATH"
fi

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
DATASET_NAME="${LIST#boreas-}"
OUTPUT_DIR="$PBS_O_WORKDIR/model_answers/$MODEL_FOLDER/$DATASET_NAME/$BENCHMARK"
ANSWERS="$OUTPUT_DIR/question_answer.json"
GRADED_RESULTS="$OUTPUT_DIR/graded_results.json"
FINAL_MARK="$OUTPUT_DIR/final_mark.txt"

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
echo "Running $MODEL_NAME $BENCHMARK evaluation for: $LIST"
echo "Frames: up to $MAX_FRAMES at max side $MAX_IMAGE_SIZE px"
echo "Questions: $QUESTIONS"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

python3 -m QA.run_questionnaire \
  --frames-dir "$CAMERA_DIR" \
  --questions "$QUESTIONS" \
  --config "$CONFIG" \
  --input-mode frames \
  --frame-budget "$MAX_FRAMES" \
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
sampled = metadata.get("frames_sampled", metadata.get("frames_used"))
received = metadata.get("frames_received", metadata.get("frames_used"))
if sampled is None or received is None:
    raise SystemExit("Missing frame coverage metadata; refusing to grade")
if received != sampled:
    raise SystemExit(
        f"Only {received}/{sampled} sampled frames were accepted; refusing to grade"
    )
PY

python3 grade_answers.py \
  --questions "$QUESTIONS" \
  --answers "$ANSWERS" \
  --output "$GRADED_RESULTS"

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

echo "Evaluation complete. Results: $OUTPUT_DIR"
