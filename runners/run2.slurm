#!/bin/bash
#SBATCH --cpus-per-task 127
#SBATCH --output logs/%j_%x.out

# comment this out to run full model
export AXLEARN_NUM_LAYERS=80
export AXLEARN_REMAT_LAYER=selective
export AXLEARN_MODEL_NAME="fuji-70B-v2-flash"
# export AXLEARN_TP_DEGREE=4
# export AXLEARN_FSDP_DEGREE=128
export AXLEARN_TRAIN_BATCH_SIZE=128

export AXLEARN_USE_BLOCKWISE=1
export AXLEARN_MAX_SEQUENCE_LENGTH=4096

# set the env to use here
# it expects the env to be at ../$VENV_NAME
export VENV_NAME=jaxmoe
export AXLEARN_REPEATED=1

# Setting below two flags profiles on 3rd step and exits
# export AXLEARN_PROFILE_MODE="tracerun"
# export PROFILE_JOB_NAME="4l_150b_bs16_4k_selkernelremat_repeated_1n"

# export AXLEARN_JAX_BACKEND=cpu

# mode=${1:-"repeated"}
# if [ $mode = "stacked" ]; then
#     export AXLEARN_REPEATED=0
#     export VENV_NAME=jaxmoestacked
#     # When not using repeated, latest pjrt is /shared/csavsrin/artifacts_kernel_csavsrin/test_remat/libneuronxla-2.2.20250530+c4f06ac.dev-py3-none-linux_x86_64.whl
# elif [ $mode = "repeated" ]; then
#     echo "Using repeated"
#     export AXLEARN_REPEATED=1
#     export VENV_NAME=jaxmoe
#     # Set to use repeated, make sure to use /shared/huilgolr/jul-end-artifacts/repeated/libneuronxla-2.2.20250521+7e624b6.dev-py3-none-linux_x86_64
# fi

if [ ${2:-1} = "1" ]; then
    echo "Installing"
    srun -l /shared/akshiaws/axlearn/setup_node.sh /shared/akshiaws/jul-end-artifacts
else
    echo "Skip installing"
fi

srun -l /shared/akshiaws/axlearn/runner.sh
