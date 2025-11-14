#!/bin/bash
# ===== SLURM directives =====
#SBATCH --job-name=switch_base_dense_cpu
#SBATCH --partition=no_bots
#SBATCH --cpus-per-task=127
#SBATCH --output logs/%j_%x.out

# ===== Your environment variables =====
export AXLEARN_NUM_LAYERS=2
export AXLEARN_REMAT_LAYER=selective
export AXLEARN_MODEL_NAME="envy-Switch-Base"
export AXLEARN_TP_DEGREE=4
export AXLEARN_EP_DEGREE=4
export AXLEARN_SEQ_DEGREE=4
export AXLEARN_TRAIN_BATCH_SIZE=4

# use v2 index calc
export AXLEARN_USE_BLOCKWISE=2   # 2 - TopKGatingGatherBlockwiseV2 vs 1 - TopKGatingGatherBlockwise

# Force block size to be compatible with Neuron
# export AXLEARN_USE_BLOCKWISE_MLP_KERNEL=0  # 1 - NKI kernel vs 0 - native JAX impl

# CPU backend for JAX
export AXLEARN_JAX_BACKEND="cpu"

export AXLEARN_MOE_LAYER_FREQ=0  # 0=dense, 1=sparse, 2=alternating

export AXLEARN_FSDP_DEGREE=1
export EP_WITHIN_NODE=1

if [ "${AXLEARN_SEQ_DEGREE:-0}" -gt 1 ]; then
    export AXLEARN_FLASH_ATTENTION=0
else
    export AXLEARN_FLASH_ATTENTION=1
fi


export AXLEARN_REPEATED=1

# Optional: name artifacts for this run (will be used by runner.sh)
export JOB_ID=switch_base_sparse_8192_bs4_128_cpu_dense

# If your runner expects a specific venv name relative to the repo:
export VENV_NAME=jaxmoe


if [ ${2:-1} = "1" ]; then
    echo "Installing"
    srun -l ./setup_node.sh
else
    echo "Skip installing"
fi

# ===== Launch =====
# Use srun so SLURM tracks resources and signals correctly.

srun -l ./runner.sh