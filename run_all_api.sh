#!/bin/bash
#PBS -N temporal_questions
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

# Enter the environment containing the local vLLM pipeline dependencies.
source /scratch/pg06/vm4618/envs/vllm_env/bin/activate

# Boreas lists to process, in the requested order.
LISTS=(
    "boreas-2024-12-04-14-34"
)

BASE="/g/data/pg06/FYP2026S1_3473"
CONFIG="configs/temporal_events.json"
# Keep the original config so we can restore it after the loop.
ORIGINAL_CONFIG="$(cat "$CONFIG")"
trap 'printf "%s\n" "$ORIGINAL_CONFIG" > "$CONFIG"' EXIT

# Force the API backend and the API model so the run does not depend on
# manually editing the config beforehand.
python3 - "$CONFIG" <<'EOF'
import json
import sys

config_path = sys.argv[1]

with open(config_path) as f:
    config = json.load(f)

config["backend"] = "api"
config["model"] = "gpt-6-luna"

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
EOF

for LIST in "${LISTS[@]}"; do
  echo "=============================================="
  echo "Running temporal questions stage for: $LIST"
  echo "=============================================="

  # Only the per-list paths differ between runs; the API backend and model are
  # already set above for every list.
  python3 - "$CONFIG" "$BASE" "$LIST" <<'EOF'
import json
import sys

config_path, base, list_name = sys.argv[1:4]

with open(config_path) as f:
    config = json.load(f)

config["frames_dir"] = f"{base}/boreas_dataset/{list_name}/camera"
config["sections_dir"] = f"{base}/{list_name}/temporal_outputs/sections"
config["output"] = f"{base}/{list_name}/temporal_outputs/narratives"

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
EOF

  python3 main.py --mode temporal
done

echo "All lists done."
