#!/bin/bash
#PBS -N answer_eval
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

# cd to the right directory
cd "$PBS_O_WORKDIR"

# Enter the environment containing the pipeline dependencies.
source /scratch/pg06/vm4618/envs/vllm_env/bin/activate

# Boreas lists to process, in the requested order.
LISTS=(
    "boreas-2025-02-15-16-58"
)

BASE="/g/data/pg06/FYP2026S1_3473"

for LIST in "${LISTS[@]}"; do
  echo "=============================================="
  echo "Running answer evaluation for: $LIST"
  echo "=============================================="

  # Each list uses its own frames, questions, and output folder. The model
  # comes from configs/answer_eval.json.
  OUTPUT_DIR="$BASE/$LIST/eval_outputs"

  python3 run_model_answers.py \
    --frames_dir "$BASE/boreas_dataset/$LIST/camera" \
    --questions "$BASE/$LIST/sparse_outputs/events/questions.json" \
    --output "$OUTPUT_DIR/answers.json"

  python3 grade_answers.py \
    --questions "$BASE/$LIST/sparse_outputs/events/questions.json" \
    --answers "$OUTPUT_DIR/answers.json" \
    --output "$OUTPUT_DIR/graded_results.json"
done

echo "All lists done."
