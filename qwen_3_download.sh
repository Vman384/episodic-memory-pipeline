#!/bin/bash
#PBS -N qwen_download
#PBS -P pg06
#PBS -q copyq
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l walltime=02:30:00
#PBS -l storage=scratch/pg06
#PBS -l wd
#PBS -V
#PBS -o qwen3_vl.out
#PBS -e qwen3_vl.err


# Activate environment
source /scratch/pg06/FYP2026S1_3473/.venv/bin/activate

# Load Gadi modules
module purge
module load cuda/12.4
module load python3/3.11.7

pip install huggingfacehub
pip install vllm

export HF_HOME=/scratch/pg06/FYP2026S1_3473/huggingface_cache

# Get the internal hostname and port of the assigned Gadi compute node
echo "Downloading Qwen3"

# download it
huggingface-cli download Qwen/Qwen3-VL-235B-A22B-Instruct