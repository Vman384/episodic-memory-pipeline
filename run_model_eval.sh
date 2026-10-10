#!/bin/bash
#PBS -N model_eval_qwen3
#PBS -P pg06
#PBS -q gpuhopper
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=1024GB
#PBS -l walltime=05:00:00
#PBS -l storage=scratch/pg06+gdata/pg06
#PBS -l wd
#PBS -V

set -euo pipefail

module load python3/3.11.7
module load cuda/12.2.2

cd "$PBS_O_WORKDIR"
source /scratch/pg06/vm4618/envs/vllm_env/bin/activate

# Use the shared model cache and prevent model resolution from making network requests.
export HF_HOME="/g/data/pg06/FYP2026S1_3473/huggingface_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# DeepGEMM JIT needs NVCC >= 12.3 but the CUDA module provides 12.2.2.
# Fall back to vLLM's CUTLASS FP8 kernels instead.
export VLLM_USE_DEEP_GEMM=0

BASE="/g/data/pg06/FYP2026S1_3473"
LIST="${LIST:-boreas-2025-02-15-16-58}"
VIDEO="$BASE/boreas_dataset/$LIST/video.mp4"
QUESTIONS="${QUESTIONS:-$PBS_O_WORKDIR/questions.json}"
CONFIG="${CONFIG:-configs/qa_qwen3.json}"
MAX_FRAMES="${MAX_FRAMES:-32}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-768}"
INPUT_MODE="${INPUT_MODE:-native_video}"
SEED="${SEED:-42}"

OUTPUT_DIR="$BASE/$LIST/eval_outputs"
ANSWERS="$OUTPUT_DIR/qwen3_answers.json"
GRADED_RESULTS="$OUTPUT_DIR/qwen3_graded_results.json"

for INPUT_FILE in "$VIDEO" "$QUESTIONS" "$CONFIG"; do
  if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Required input file not found: $INPUT_FILE" >&2
    exit 1
  fi
done

echo "=============================================="
echo "Running Qwen3 questionnaire evaluation"
echo "Drive: $LIST"
echo "Video: $VIDEO"
echo "Questions: $QUESTIONS"
echo "Input mode: $INPUT_MODE"
if [[ "$INPUT_MODE" == "frames" ]]; then
  echo "Frames: up to $MAX_FRAMES, max side $MAX_IMAGE_SIZE px"
fi
echo "=============================================="

QA_ARGS=(
  --video "$VIDEO"
  --questions "$QUESTIONS"
  --config "$CONFIG"
  --input-mode "$INPUT_MODE"
  --seed "$SEED"
  --output "$ANSWERS"
)
if [[ "$INPUT_MODE" == "frames" ]]; then
  QA_ARGS+=(--max-frames "$MAX_FRAMES" --max-image-size "$MAX_IMAGE_SIZE")
fi

python3 -m QA.run_questionnaire "${QA_ARGS[@]}"

python3 grade_answers.py \
  --questions "$QUESTIONS" \
  --answers "$ANSWERS" \
  --output "$GRADED_RESULTS"

echo "Model evaluation complete."
