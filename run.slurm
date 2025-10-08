#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J ishaniak
#SBATCH --partition=compute1
#SBATCH --output=/shared/ishaniak/axlearn_trn/slurm-%j.out
#SBATCH --error=/shared/ishaniak/axlearn_trn/slurm-%j.err

# Print debug information
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Run type: $1"

# Set the full paths to your scripts
SCRIPT_DIR="/shared/ishaniak/axlearn_trn"

if [ "$1" = "cpu" ]; then
    ${SCRIPT_DIR}/runner_cpu.sh myenv gpu_run setup.70B.sh
elif [ "$1" = "gpu" ]; then
    ${SCRIPT_DIR}/run_gpu.sh ${SCRIPT_DIR}/runner_cpu.sh myenv gpu_run setup.70B.sh
else
    echo "Invalid argument. Please use 'cpu' or 'gpu'"
    exit 1
fi