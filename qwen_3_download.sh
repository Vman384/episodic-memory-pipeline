#!/bin/bash
#PBS -N qwen_download
#PBS -P pg06
#PBS -q copyq
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l walltime=02:30:00
#PBS -l storage=scratch/pg06+gdata/pg06
#PBS -l wd
#PBS -o qwen3_vl.out
#PBS -e qwen3_vl.err

# Load Gadi modules (python only, CUDA not needed for downloading)
module load python3/3.11.7

# Activate env
source /g/data/pg06/FYP2026S1_3473/.venv/bin/activate

export HF_HOME="/g/data/pg06/FYP2026S1_3473/huggingface_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"

# download it
echo "Downloading Qwen3"
hf download Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 --cache-dir "$HF_HUB_CACHE"
