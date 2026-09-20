#!/bin/bash
#PBS -N sparse_event_extract
#PBS -P pg06
#PBS -q gpuhopper
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=1024GB
#PBS -l walltime=00:40:00
#PBS -l storage=scratch/pg06+gdata/pg06
#PBS -l wd
#PBS -V

set -euo pipefail
 
module load python3/3.11.7
module load cuda/12.2.2

# cd to the right directory
cd "$PBS_O_WORKDIR"

# Enter the environment containing the local vLLM pipeline dependencies.
source /scratch/pg06/vm4618/envs/vllm_env/bin/activate

# Use the shared model cache and prevent model resolution from making network requests.
export HF_HOME="/g/data/pg06/FYP2026S1_3473/huggingface_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Qwen3-VL 235B FP8 is sharded across the four H200 GPUs of one gpuhopper node.
# vLLM loads the snapshot directory, not the Hugging Face cache repo root.
MODEL_PATH="/g/data/pg06/FYP2026S1_3473/huggingface_cache/hub/models--Qwen--Qwen3-VL-235B-A22B-Instruct-FP8/snapshots/7fbcd8c9e2ad011808ed8a57d64c462605c3e381"

# Boreas lists to process, in the order they were run.
LISTS=(
  "boreas-2024-12-03-13-13"
  "boreas-2024-12-03-13-34"
  "boreas-2024-12-04-11-45"
  "boreas-2024-12-04-11-56"
  "boreas-2025-07-18-15-12"
)

BASE="/g/data/pg06/FYP2026S1_3473"
CONFIG="configs/sparse_events.json"
# Keep the original config so we can restore it after the loop.
ORIGINAL_CONFIG="$(cat "$CONFIG")"
trap 'printf "%s\n" "$ORIGINAL_CONFIG" > "$CONFIG"' EXIT

for LIST in "${LISTS[@]}"; do
  echo "=============================================="
  echo "Running sparse stages for: $LIST"
  echo "=============================================="

  # Only the per-list paths differ between runs; backend/model are fixed.
  python3 - "$CONFIG" "$BASE" "$LIST" "$MODEL_PATH" <<'EOF'
import json
import sys

config_path, base, list_name, model_path = sys.argv[1:5]

with open(config_path) as f:
    config = json.load(f)

config["frames_dir"] = f"{base}/boreas_dataset/{list_name}/camera"
config["sections_dir"] = f"{base}/{list_name}/sparse_outputs/sections"
config["output"] = f"{base}/{list_name}/sparse_outputs/events"

# Run extraction and review with the local Qwen3-VL checkpoint.
config["backend"] = "local"
config["model"] = model_path
# One shard per GPU on the four-GPU gpuhopper node.
config["tensor_parallel_size"] = 4

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
EOF

  python3 main.py --mode sparse
done

echo "All lists done."
