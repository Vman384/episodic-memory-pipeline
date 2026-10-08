#!/bin/bash
#===============================================================================
# forest_run.sh - episodic memory pipeline for the forest K-01 dataset.
#
# Both versions are in this one file:
#   API   - copyq queue, no GPU, model served over the API
#   Local - gpuhopper queue, 4 GPUs, local vLLM model
#
# The LOCAL PBS header and sparse run are active below. Only one PBS header may
# be active at a time, and it must match the run blocks left uncommented.
# Comment out the version (header and run blocks) you are not running.
#===============================================================================

## # ---- PBS header: local model run (active) ----
## PBS -N forest_local
## PBS -P pg06
## PBS -q gpuhopper
## PBS -l ncpus=48
## PBS -l ngpus=4
## PBS -l mem=1024GB
## PBS -l walltime=10:00:00
## PBS -l storage=scratch/pg06+gdata/pg06
## PBS -l wd
## PBS -V

# ---- PBS header: API run
#PBS -N forest_api
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
module load cuda/12.2.2

# cd to the right directory
cd "$PBS_O_WORKDIR"

# Enter the environment containing the pipeline dependencies.
source /scratch/pg06/vm4618/envs/vllm_env/bin/activate

# Use the shared model cache and prevent model resolution from making network requests.
export HF_HOME="/g/data/pg06/FYP2026S1_3473/huggingface_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# DeepGEMM JIT needs NVCC >= 12.3 but the CUDA module provides 12.2.2.
# Fall back to vLLM's CUTLASS FP8 kernels instead.
export VLLM_USE_DEEP_GEMM=0

# Forest K-01 frames and outputs.
FRAMES_DIR="/g/data/pg06/FYP2026S1_3473/forest_dataset/K-01_data"
SPARSE_SECTIONS="/g/data/pg06/FYP2026S1_3473/forest_dataset/K-01_sparse_outputs/sections"
SPARSE_OUTPUT="/g/data/pg06/FYP2026S1_3473/forest_dataset/K-01_sparse_outputs/events"
TEMPORAL_SECTIONS="/g/data/pg06/FYP2026S1_3473/forest_dataset/K-01_temporal_outputs/sections"
TEMPORAL_OUTPUT="/g/data/pg06/FYP2026S1_3473/forest_dataset/K-01_temporal_outputs/narratives"

SPARSE_CONFIG="configs/sparse_events.json"
TEMPORAL_CONFIG="configs/temporal_events.json"

# Keep the original configs so they can be restored after the runs.
ORIGINAL_SPARSE_CONFIG="$(cat "$SPARSE_CONFIG")"
ORIGINAL_TEMPORAL_CONFIG="$(cat "$TEMPORAL_CONFIG")"
trap 'printf "%s\n" "$ORIGINAL_SPARSE_CONFIG" > "$SPARSE_CONFIG"; printf "%s\n" "$ORIGINAL_TEMPORAL_CONFIG" > "$TEMPORAL_CONFIG"' EXIT

set_config() {
  # set_config <config> <key> <json-value>
  python3 - "$1" "$2" "$3" <<'EOF'
import json
import sys

config_path, key, value = sys.argv[1:4]

with open(config_path) as config_file:
    config = json.load(config_file)

try:
    value = json.loads(value)
except json.JSONDecodeError:
    pass

config[key] = value

with open(config_path, "w") as config_file:
    json.dump(config, config_file, indent=2)
EOF
}

#===============================================================================
# API MODEL RUNS
# Comment out this whole section when running the local model.
#===============================================================================

# ---- Sparse (API): extract then review ----
# echo "=== [api] sparse: K-01 extract + review ==="
# set_config "$SPARSE_CONFIG" backend api
# set_config "$SPARSE_CONFIG" model gpt-5.6-luna
# set_config "$SPARSE_CONFIG" frames_dir "$FRAMES_DIR"
# set_config "$SPARSE_CONFIG" sections_dir "$SPARSE_SECTIONS"
# set_config "$SPARSE_CONFIG" output "$SPARSE_OUTPUT"
# python3 main.py --mode sparse

---- Temporal (API): extract then timeline ----
echo "=== [api] temporal: K-01 extract + timeline ==="
set_config "$TEMPORAL_CONFIG" backend api
set_config "$TEMPORAL_CONFIG" model gpt-5.6-luna
set_config "$TEMPORAL_CONFIG" frames_dir "$FRAMES_DIR"
set_config "$TEMPORAL_CONFIG" sections_dir "$TEMPORAL_SECTIONS"
set_config "$TEMPORAL_CONFIG" output "$TEMPORAL_OUTPUT"
python3 main.py --mode temporal

#===============================================================================
# LOCAL MODEL RUNS
# Comment out this whole section when running the API model.
#===============================================================================

# # ---- Sparse (local): extract then review ----
# echo "=== [local] sparse: K-01 extract + review ==="
# set_config "$SPARSE_CONFIG" backend local
# set_config "$SPARSE_CONFIG" model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
# set_config "$SPARSE_CONFIG" frames_dir "$FRAMES_DIR"
# set_config "$SPARSE_CONFIG" sections_dir "$SPARSE_SECTIONS"
# set_config "$SPARSE_CONFIG" output "$SPARSE_OUTPUT"
# python3 main.py --mode sparse

# ---- Temporal (local): extract then timeline ----
# echo "=== [local] temporal: K-01 extract + timeline ==="
# set_config "$TEMPORAL_CONFIG" backend local
# set_config "$TEMPORAL_CONFIG" model Qwen/Qwen2.5-VL-72B-Instruct
# set_config "$TEMPORAL_CONFIG" frames_dir "$FRAMES_DIR"
# set_config "$TEMPORAL_CONFIG" sections_dir "$TEMPORAL_SECTIONS"
# set_config "$TEMPORAL_CONFIG" output "$TEMPORAL_OUTPUT"
# python3 main.py --mode temporal

echo "Forest K-01 local sparse run done."
