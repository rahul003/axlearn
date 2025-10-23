#!/bin/bash
# comment this out to run full model
export AXLEARN_NUM_LAYERS=2
export AXLEARN_REMAT_LAYER=selective
export AXLEARN_MODEL_NAME="envy-Mistral-8x7B"
export AXLEARN_TP_DEGREE=4
export AXLEARN_EP_DEGREE=1
export AXLEARN_SEQ_DEGREE=1
export AXLEARN_SEQ_DEGREE=1
export AXLEARN_TRAIN_BATCH_SIZE=16
# use v2 index calc
export AXLEARN_USE_BLOCKWISE=2

if [ "${AXLEARN_SEQ_DEGREE:-0}" -gt 1 ]; then
    export AXLEARN_FLASH_ATTENTION=0
else
    export AXLEARN_FLASH_ATTENTION=1
fi
export AXLEARN_FLASH_ATTENTION=0

export AXLEARN_REPEATED=1
# it expects the env to be at ../$VENV_NAME
export VENV_NAME=jaxmoe

# to simulate slurm job run
export SLURM_PROCID=0
# to output artifacts at this path ./artifacts/JOB_ID/
export JOB_ID=8x7b
rm -rf ./artifacts/$JOB_ID/
bash runner.sh 2>&1 | tee log_$JOB_ID.out