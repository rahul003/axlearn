#!/bin/bash
# comment this out to run full model
set +e
set -euo pipefail

export AXLEARN_NUM_LAYERS=2
export AXLEARN_REMAT_LAYER=selective
export AXLEARN_MODEL_NAME="envy-Switch-Base"
export AXLEARN_TP_DEGREE=4
export AXLEARN_EP_DEGREE=4
export AXLEARN_SEQ_DEGREE=4
export AXLEARN_TRAIN_BATCH_SIZE=4
# use v2 index calc
export AXLEARN_USE_BLOCKWISE=2 # 2 - TopKGatingGatherBlockwiseV2 vs 1 - TopKGatingGatherBlockwise
# Force block size to be compatible with Neuron
export AXLEARN_USE_BLOCKWISE_MLP_KERNEL=0 # 1 - blockwise_mm (NKI implementation) vs 0 - blockwise_mm_per_group_native (JAX implementation)

export AXLEARN_MOE_LAYER_FREQ=2 # - 0 (dense) - 1 (sparse) - 2 (alternating)
export AXLEARN_FSDP_DEGREE=1
export EP_WITHIN_NODE=1

# export NEURON_RT_DBG_DUMP_INPUTS_ON_ERR=1006
# export NEURON_RT_LOCAL_CORE_DUMP_DIRECTORY=""

export AXLEARN_PROFILE_MODE="tracerun" # "capture" # "tracerun"
export SLURM_JOB_ID=switch_base_alternating_8192_bs4_128_runtime_profile_5
export PROFILE_JOB_ID="sparse/switch_base_alternating_8192_bs4_128_runtime_profile_5" # required for capture only
export PROFILE_JOB_NAME=switch_base_runtime_profile
export SLURM_JOB_NUM_NODES=1


if [ "${AXLEARN_SEQ_DEGREE:-0}" -gt 1 ]; then
    export AXLEARN_FLASH_ATTENTION=0
else
    export AXLEARN_FLASH_ATTENTION=1
fi

# export JAX_COORDINATOR_ADDRESS="" # running with this only, otherwise training aborts
export AXLEARN_REPEATED=1
# it expects the env to be at ../$VENV_NAME
# export VENV_NAME=../jaxmoe

# to simulate slurm job run
export SLURM_PROCID=0
export SLURM_LOCALID=0
# to output artifacts at this path ./artifacts/JOB_ID/
export JOB_ID=switch_base_alternating_8192_bs4_128_runtime_profile_5
# rm -rf /fsx/divyamsh/artifacts/$JOB_ID/
bash ./runner.sh
# 2>&1 | tee log_$JOB_ID.out