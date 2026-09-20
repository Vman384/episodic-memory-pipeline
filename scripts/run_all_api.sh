#!/bin/bash
#PBS -N sparse_event_questions
#PBS -P pg06
#PBS -q copyq
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l walltime=00:30:00
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


# Boreas lists to process, in the order they were run.
LISTS=(
#  "boreas-2025-07-18-14-55"
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
  echo "Running questions stage for: $LIST"
  echo "=============================================="

  # Only the per-list paths differ between runs; backend/model are fixed.
  python3 - "$CONFIG" "$BASE" "$LIST" <<'EOF'
import json
import sys

config_path, base, list_name = sys.argv[1:4]

with open(config_path) as f:
    config = json.load(f)

config["frames_dir"] = f"{base}/boreas_dataset/{list_name}/camera"
config["sections_dir"] = f"{base}/{list_name}/sparse_outputs/sections"
config["output"] = f"{base}/{list_name}/sparse_outputs/events"

# Questions is text-only, so use the API backend instead of spinning up vLLM.
config["backend"] = "api"
config["model"] = "gpt-5.6-luna"
# gpt-5.6-luna does not accept the temperature parameter.
config["temperature"] = None

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
EOF

  python3 main.py --mode sparse
done

echo "All lists done."
