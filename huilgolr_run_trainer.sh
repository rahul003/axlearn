#!/usr/bin/env bash

#ARGS
##---------------------

install=0

# set runner args here as env vars
source ../launchers/runner.sh $install

export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_hlo_as_proto --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*' --xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot "

# export TF_CPP_MIN_LOG_LEVEL=0
# export TF_CPP_VMODULE="neuron_model_parallel_sharding_propagation=3"
export NEURON_TRANSFORMER_SHARDING=0
export NEURON_ALL_REDUCE_CONTIGUOUS=0
export NEURON_REDUCE_SCATTER_CONTIGUOUS=0
export NEURON_AUTOSHARD=0
export NEURON_AUTOSHARD_KEEP_ALL_SHARDINGS=0
export NEURON_HLO_ANALYZER=1

DATA_DIR="gs://axlearn-public/tensorflow_datasets"
# Run the training script

# python3 -m axlearn.common.mixture_of_experts_test TransformerFeedForwardMoETest.test_moe_layer_aux_loss

python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=envy-test \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
    --jax_backend=neuron --mesh_selector=neuron-trn1.32xlarge-2048 \
    --distributed_coordinator=$MASTER_ADDR:$JAX_COORDINATOR_PORT --num_processes=$SLURM_JOB_NUM_NODES \
    --process_id=$NEURON_PJRT_PROCESS_INDEX