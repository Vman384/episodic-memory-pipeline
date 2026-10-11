#!/bin/bash
#PBS -N answer_eval
#PBS -P pg06
#PBS -q copyq
#PBS -l ncpus=1
#PBS -l mem=8GB
#PBS -l walltime=0:05:00
#PBS -l storage=scratch/pg06+gdata/pg06
#PBS -l wd
#PBS -V

set -euo pipefail

module load python3/3.11.7

# cd to the right directory
cd "$PBS_O_WORKDIR"

# Enter the environment containing the pipeline dependencies.
source /scratch/pg06/vm4618/envs/vllm_env/bin/activate

# Boreas lists to process, in the requested order.
LISTS=(
    "boreas-2025-02-15-16-58"
)

BASE="/g/data/pg06/FYP2026S1_3473"
MAX_FRAMES="${MAX_FRAMES:-128}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-512}"
API_BATCH_SIZE="${API_BATCH_SIZE:-20}"
SEED="${SEED:-42}"

for LIST in "${LISTS[@]}"; do
  echo "=============================================="
  echo "Running answer evaluation for: $LIST"
  echo "=============================================="

  # Each list uses its own frames, questions, and output folder. The model
  # comes from configs/answer_eval.json. QA samples a bounded frame set once
  # and answers all questions for the drive in one response.
  OUTPUT_DIR="$BASE/$LIST/eval_outputs"

  python3 -m QA.run_questionnaire \
    --frames-dir "$BASE/boreas_dataset/$LIST/camera" \
    --questions "$BASE/$LIST/sparse_outputs/events/questions.json" \
    --config "configs/answer_eval.json" \
    --input-mode frames \
    --frame-budget "$MAX_FRAMES" \
    --max-image-size "$MAX_IMAGE_SIZE" \
    --api-batch-size "$API_BATCH_SIZE" \
    --seed "$SEED" \
    --output "$OUTPUT_DIR/answers.json"

  python3 - "$OUTPUT_DIR/answers.json" <<'PY'
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
    --questions "$BASE/$LIST/sparse_outputs/events/questions.json" \
    --answers "$OUTPUT_DIR/answers.json" \
    --output "$OUTPUT_DIR/graded_results.json"
done

echo "All lists done."
