#!/bin/bash
#SBATCH --cpus-per-task 127
#SBATCH --output logs/%j_%x.out


# axlearn repeated = 0
# tp_all+gather should be present when fsdp is set and fsdp+toekn_tp_ag
# NEURON_FSDP_CC_MULTISTREAM = 1 - assigning streams - these streams have to set - 1
# export NEURON_FSDP_CC_MULTISTREAM = 1
# how are these flags making changes to the pjrt passes
# fsdp_reduce_scatter also aatfrom what rahul sent
# all the tags sent by rahul have to presne tin the HLO
# tp_all+gather and tp_reduce_scatter
# stream id will be 1 0 
# fsdp is always stream 1 slower one\
# all gather should happen twice
# HLO validator

# next steps
# what is wrong with axlearn_repeated? n
# neuron_fsdp = 1
# neuron_fsdp _repate = 1 is not doing the right thing
# fsdp collectives ? tp collectives?
# check bs and layer sin HLO
# no tags in fsdp


# 1. why is neuron/-fsdp_repated is not working?
# 2. 4 layers tp 1 fsdp - reduce hidden size make it work 

# /shared/akshiaws/axlearn/setup_node.sh
# /shared/akshiaws/axlearn/efa_setup.sh
# export AXLEARN_NUM_LAYERS=80
# export AXLEARN_REMAT_LAYER=selective
# export AXLEARN_MODEL_NAME="fuji-70B-v2-flash"
# export AXLEARN_TP_DEGREE=4
# # export AXLEARN_FSDP_DEGREE=128
# export AXLEARN_TRAIN_BATCH_SIZE=128
# AXLEARN_USE_BLOCKWISE=1
# export AXLEARN_MAX_SEQUENCE_LENGTH=4096
# # set the env to use here
# # it expects the env to be at ../$VENV_NAME
# VENV_NAME=jaxmoe
# AXLEARN_REPEATED=1

# Setting below two flags profiles on 3rd step and exits
# export AXLEARN_PROFILE_MODE="tracerun"
# export PROFILE_JOB_NAME="4l_150b_bs16_4k_selkernelremat_repeated_1n"

# export AXLEARN_JAX_BACKEND=cpu

# mode=${1:-"repeated"}
# if [ $mode = "stacked" ]; then
#     export AXLEARN_REPEATED=1
#     export VENV_NAME=jaxmoestacked
#     # When not using repeated, latest pjrt is /shared/csavsrin/artifacts_kernel_csavsrin/test_remat/libneuronxla-2.2.20250530+c4f06ac.dev-py3-none-linux_x86_64.whl
# elif [ $mode = "repeated" ]; then
#     echo "Using repeated"
#     export AXLEARN_REPEATED=1
#     export VENV_NAME=jaxmoe
#     # Set to use repeated, make sure to use /shared/huilgolr/jul-end-artifacts/repeated/libneuronxla-2.2.20250521+7e624b6.dev-py3-none-linux_x86_64
# fi

# if [ ${2:-1} = "1" ]; then
#     echo "Installing"
#     srun -l /shared/akshiaws/axlearn/setup_node.sh /shared/akshiaws/jul-end-artifacts
# else
#     echo "Skip installing"
# fi

srun -l /shared/akshiaws/axlearn/runner3.sh
