# 1. Load Modules
module load python3/3.11.7 cuda/12.2.2

# 2. Fix C++ Header Paths (Missing Python.h)
export CPATH="/apps/python3/3.11.7/include/python3.11:$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))"):$CPATH"
export CPLUS_INCLUDE_PATH=$CPATH

# 3. Fix CUDA Compiler Paths (FlashInfer JIT / nvcc)
export CUDA_HOME=/apps/cuda/12.2.2
export PATH=$CUDA_HOME/bin:$PATH

# 4. Direct PyTorch JIT Cache to Fast Local Node Storage
export TORCH_EXTENSIONS_DIR="${PBS_JOBFS:-/tmp}/torch_extensions"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# 5. Activate your venv
source /scratch/pg06/vm4618/envs/vllm_env/bin/activate

export HF_HOME=/scratch/pg06/vm4618/huggingface_cache

# 6. Run interactive job
qsub -I -P pg06 -q dgxa100 -l walltime=02:00:00,mem=64GB,ncpus=16,ngpus=1
qsub -I -P pg06 -q gpuhopper-exec -l walltime=02:00:00,mem=64GB,ncpus=12,ngpus=1
