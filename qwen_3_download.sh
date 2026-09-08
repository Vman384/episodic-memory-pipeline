#!/bin/bash
#PBS -N qwen3_vl_serve
#PBS -P pg06
#PBS -q copyq
#PBS -l walltime=02:00:00
#PBS -l ncpus=1
#PBS -l mem=10GB
#PBS -l storage=/scratch/pg06
#PBS -l wd
#PBS -o qwen3_vl.out
#PBS -e qwen3_vl.err


# Load Gadi modules
module purge
module load cuda/12.4
module load python3/3.11.7

# Activate environment
source /scratch/pg06/FYP2026S1_3473/.venv/bin/activate

# Avoid OpenMP CPU thread collisions on AMD/Intel sockets
export OMP_NUM_THREADS=1

# Get the internal hostname and port of the assigned Gadi compute node
NODE_HOST=$(hostname -I | awk '{print $1}')
PORT=8000
echo "API Server running on http://${NODE_HOST}:${PORT}"

# Launch vLLM server across all 8 GPUs
vllm serve /g/data/<project_id>/$USER/models/Qwen3-VL-235B-A22B-Instruct \
  --host 0.0.0.0 \
  --port ${PORT} \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.94 \
  --trust-remote-code
