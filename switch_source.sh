#!/bin/bash
# comment this out to run full model
export AXLEARN_NUM_LAYERS=2
export AXLEARN_REMAT_LAYER=selective
export AXLEARN_MODEL_NAME="envy-Switch-XXL"
export AXLEARN_TP_DEGREE=4
export AXLEARN_EP_DEGREE=4
export AXLEARN_SEQ_DEGREE=4
export AXLEARN_TRAIN_BATCH_SIZE=2
# use v2 index calc
export AXLEARN_USE_BLOCKWISE=2
# Force block size to be compatible with Neuron

# - 0 (dense) - 1 (sparse) - 2 (alternating)
export AXLEARN_MOE_LAYER_FREQ=2
export AXLEARN_FSDP_DEGREE=1
export EP_WITHIN_NODE=1
# export AXLEARN_PROFILE_MODE="tracerun"
# export NEURON_RT_LOCAL_CORE_DUMP_DIRECTORY=""

if [ "${AXLEARN_SEQ_DEGREE:-0}" -gt 1 ]; then
    export AXLEARN_FLASH_ATTENTION=0
else
    export AXLEARN_FLASH_ATTENTION=1
fi

export AXLEARN_REPEATED=0
# it expects the env to be at ../$VENV_NAME
export VENV_NAME=akshiaws/jaxmoe

# to simulate slurm job run
export SLURM_PROCID=0
# to output artifacts at this path ./artifacts/JOB_ID/
export JOB_ID=exp
rm -rf /shared/akshiaws/artifacts/$JOB_ID/
bash /shared/akshiaws/axlearn/runner.sh
2>&1 | tee log_$JOB_ID.out
