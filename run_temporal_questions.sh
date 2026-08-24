#!/bin/bash
#PBS -N temporal_event_questions
#PBS -P pg06
#PBS -q copyq
#PBS -l ncpus=4
#PBS -l mem=20GB
#PBS -l walltime=00:30:00
#PBS -l storage=scratch/pg06
#PBS -l wd
#PBS -V

# Run the temporal "questions" stage for multiple Boreas lists using the API
# backend (gpt-5.6-luna). Each list must already have a human-reviewed
# timeline.json produced by an earlier temporal extract/timeline run.
set -euo pipefail

module load python3/3.11.7

cd "$PBS_O_WORKDIR"

# Enter the environment containing the API pipeline dependencies.
source /scratch/pg06/vm4618/envs/vllm_env/bin/activate

# Boreas lists to process, in the order they were run.
LISTS=(
  "boreas-2025-07-18-14-55"
  "boreas-2024-12-03-13-13"
  "boreas-2024-12-03-13-34"
  "boreas-2024-12-04-11-45"
  "boreas-2024-12-04-11-56"
  "boreas-2025-07-18-15-12"
)

BASE="/scratch/pg06/FYP2026S1_3473"
CONFIG="configs/temporal_events.json"
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
config["sections_dir"] = f"{base}/{list_name}/temporal_outputs/sections"
config["output"] = f"{base}/{list_name}/temporal_outputs/narratives"

# Switch to the API backend using the requested model.
config["backend"] = "api"
config["model"] = "gpt-5.6-luna"

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
EOF

  python3 main.py --mode temporal --stage questions
done

echo "All lists done."
