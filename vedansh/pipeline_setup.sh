# 1. Load Modules
module load python3/3.11.7 cuda/12.2.2

# 5. Activate your venv
source /scratch/pg06/vm4618/envs/vllm_env/bin/activate

export HF_HOME=/scratch/pg06/vm4618/huggingface_cache

# 6. Run interactive job
qsub -I -P pg06 -q dgxa100 -l walltime=02:00:00,mem=64GB,ncpus=16,ngpus=1
qsub -I -P pg06 -q gpuhopper -l walltime=02:00:00,mem=64GB,ncpus=12,ngpus=1
