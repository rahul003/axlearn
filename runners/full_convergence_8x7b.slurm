#!/bin/bash
#SBATCH --cpus-per-task 127
#SBATCH --job-name 8x7b_convergence_bf16
#SBATCH --output logs/%j_%x.out
#SBATCH --nodes 8
#SBATCH --switches 1

export AXLEARN_REMAT_LAYER=selective
export VENV_NAME=jaxmoe
export AXLEARN_MODEL_NAME="envy-Mistral-8x7B"
# export JOB_ID="16x10b_convergence"
export AXLEARN_TP_DEGREE=16
export AXLEARN_TRAIN_BATCH_SIZE=$((8 * SLURM_NNODES))
export AXLEARN_USE_BLOCKWISE=1
export AXLEARN_REPEATED=1
export AXLEARN_NUM_KV_HEADS=16
export DATA_SEED=42

export OPTIMIZER_LR_BASE=1.5
export OPTIMIZER_LR_EXP=-5
export OPTIMIZER_WD=0.000006

# overriding above to use fp32
# export AXLEARN_REMAT_LAYER=true
# export AXLEARN_TRAIN_DTYPE=float32
# export NEURON_ALL_REDUCE_UPCASTER=1
# export NEURON_PROMOTE_TP_REDUCE=1

export JAX_COMPILATION_CACHE_DIR="cache/"
mkdir -p ${JAX_COMPILATION_CACHE_DIR}

export CUSTOM_TAG_experiment=convergence
export CUSTOM_TAG_lr=amith
export RENAME_JOB=true
export RENAME_JOB_PREFIX=rh

if [ ${1:-1} = "1" ]; then
    srun -l ./setup_node.sh ../may-artifacts/
else
    echo "Skip installing"
fi

srun -l runner.sh
