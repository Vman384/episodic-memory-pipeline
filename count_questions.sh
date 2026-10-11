#!/usr/bin/env bash
#PBS -N count_questions
#PBS -P pg06
#PBS -q copyq
#PBS -l ncpus=1
#PBS -l mem=1GB
#PBS -l walltime=00:10:00
#PBS -l storage=scratch/pg06+gdata/pg06
#PBS -l wd

set -euo pipefail

module load python3/3.11.7

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
