#!/usr/bin/env bash

set -euo pipefail

BASE="/g/data/pg06/FYP2026S1_3473"
cd "$BASE"

total=0

for question_file in \
  */sparse_outputs/events/questions.json \
  */temporal_outputs/narratives/questions.json
do
  [[ -f "$question_file" ]] || continue

  count=$(python3 -c 'import json, pathlib, sys; print(len(json.loads(pathlib.Path(sys.argv[1]).read_text())["questions"]))' "$question_file")
  total=$((total + count))
done

printf '%d\n' "$total"
