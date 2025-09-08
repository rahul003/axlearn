#!/bin/bash

export MODEL_ARCH="fuji-70B-v2-flash"
export N_EXPECTED_NODES=16 #For checks in main script to avoid mixing up configs
export MESH_SELECTOR="gpu-70B"

export DATA_SEED=42
