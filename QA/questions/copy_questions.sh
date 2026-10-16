#!/usr/bin/env bash
# Copy questions.json for every dataset into ./<dataset>/temporal and ./<dataset>/sparse
# Run from the directory you want the copies placed in.

SRC=/g/data/pg06/FYP2026S1_3473

DATASETS=(
  "boreas-2025-07-18-14-55"
  "boreas-2024-12-03-13-13"
  "boreas-2024-12-03-13-34"
  "boreas-2024-12-04-11-45"
  "boreas-2024-12-04-11-56"
  "boreas-2025-07-18-15-12"
  "boreas-2024-12-04-14-44"
  "boreas-2025-02-15-16-58"
  "boreas-2025-07-18-11-53"
  "boreas-2024-12-04-14-34"
  "boreas-2024-12-04-14-38"
  "boreas-2025-02-21-14-51"
  "boreas-2025-02-15-17-19"
  "boreas-2024-12-04-14-59"
  "boreas-2025-08-13-10-36"
)

for d in "${DATASETS[@]}"; do
  mkdir -p "$d/temporal" "$d/sparse"
  cp "$SRC/$d/temporal_outputs/narratives/questions.json" "$d/temporal/" || echo "MISSING temporal: $d"
  cp "$SRC/$d/sparse_outputs/events/questions.json" "$d/sparse/" || echo "MISSING sparse: $d"
done
