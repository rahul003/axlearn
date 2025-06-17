#!/bin/bash

export MODEL_ARCH="envy-Mistral-8x20B"
export AXLEARN_MAX_SEQ_LEN=8192
export AXLEARN_TRAIN_BATCH_SIZE=64
#export AXLEARN_NUM_LAYERS=44
export OPTIMIZER_LR_BASE=1.5
export OPTIMIZER_WD=0.000006
export OPTIMIZER_LR_EXP=-5
export N_ACCUMULATION=1
export N_EXPECTED_NODES=8 #For checks in main script to avoid mixing up configs
export MESH_SELECTOR="gpu-7b-mesh"

export AXLEARN_REMAT_LAYER=true
export AXLEARN_USE_BLOCKWISE=3
export AXLEARN_NUM_KV_HEADS=16
export AXLEARN_CAP_FACTOR=2
export DATA_SEED=42
