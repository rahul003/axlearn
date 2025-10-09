#!/bin/bash

export TEST_TYPE="accuracy_test"

export ENABLE_NEURON_DUMP=1
export ENABLE_HLO_DUMP=1
export ENABLE_JAX_COMPILATION_CACHE=0

export MODEL_ARCH="fuji-1B-v2-flash"
# export N_LAYERS=16
# export TRAIN_GBS=16
export N_LAYERS=4
export TRAIN_GBS=1
export EVAL_GBS=1

# export FSDP=16
# export MODEL=4
# export PIPELINE=1
# export DATA=-1
# export EXPERT=1
# export SEQ=1

# export FSDP=1
# export MODEL=8
# export PIPELINE=1
# export DATA=-1
# export EXPERT=1
# export SEQ=1

export FSDP=1
export MODEL=1
export PIPELINE=1
export DATA=1
export EXPERT=1
export SEQ=1

export OPTIMIZER_LR_BASE=3
export OPTIMIZER_LR_EXP=-4
export OPTIMIZER_WD=0.000006

export SAVE_EVERY_N_STEPS=500000
export EVAL_EVERY_N_STEPS=500000

export COMP_START_STEP=0
export COMP_STOP_STEP=1000
export ATOL=0.00000001
export RTOL=0.2
export CONFIDENCE_INTERVAL=0.90
export PRECISION = 'bf16'