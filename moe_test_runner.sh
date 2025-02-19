#!/usr/bin/env bash
set -e
# Neuron env vars for distributed training based on SLURM
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ -z "$SLURM_JOB_NODELIST" ]; then
	nodes="localhost"
	SLURM_NODEID=0
	SLURM_JOB_ID=$EPOCHSECONDS
fi

num_nodes=$(echo "$nodes" | wc -l)
devices_per_node=64
MASTER_ADDR=$(echo "$nodes" | head -n 1)
MASTER_PORT=41000
JAX_COORDINATOR_PORT=41001
export NEURON_RT_ROOT_COMM_ID="${MASTER_ADDR}:${MASTER_PORT}"
export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf '%s,' $(seq 1 $num_nodes | xargs -I {} echo $devices_per_node) | sed 's/,$//')
export NEURON_PJRT_PROCESS_INDEX=$SLURM_NODEID

# Print nodenames for debug
hostname

JOB_ID=${SLURM_JOB_ID}
ARTIFACTS_PATH="test_artifacts"
TEST_ARTIFACTS_PATH="${ARTIFACTS_PATH}/${JOB_ID}"
mkdir -p "$TEST_ARTIFACTS_PATH"
NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump
HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump
export XLA_FLAGS="--xla_dump_hlo_as_text --xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot,neuron-hierarchical-collectives --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'"

# PJRT Flags 
export NEURON_FSDP_NUM_LAYER_EARLY_AG_SHIFT=1
export NEURON_FSDP_NUM_LAYER_LATE_RS_SHIFT=2
export NEURON_ENABLE_INT_MATMUL_DOWNCAST=1
export NEURON_FSDP=0
export NEURON_FSDP_NUM_LAYER_COALESCE=-1
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1

# Neuron runtime flags
export NEURON_RT_DBG_CC_DMA_PACKET_SIZE=4096 && export NEURON_RT_DBG_DMA_PACKETIZATION_SIZE=104857
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1
export NEURON_RT_IO_RING_CACHE_SIZE=0
export NEURON_RT_ENABLE_MEMORY_METRICS=0
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_RESET_CORES=1
export NEURON_RT_LOG_LEVEL="WARNING"
export NEURON_RT_ENABLE_INTERNODE_EXECUTION_BARRIER=1

# Neuron collectives flag
export FI_LOG_LEVEL="warn"
export OFI_NCCL_PROTOCOL=RDMA
export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE=1
export OFI_NCCL_MR_CACHE_DISABLE=1

# Neuron compiler flags
export NEURON_CC_FLAGS="--framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-max-instruction-limit=20000000"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-num-neuroncores-per-sengine=2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --tensorizer-options='--enable-hoist-fsdp-collectives'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--remat-rope'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=${NEURON_DUMP_PATH}"

# JAX Cache
export JAX_COMPILATION_CACHE_DIR="cache/"
mkdir -p ${JAX_COMPILATION_CACHE_DIR}

deactivate || true
# conda
# eval "$(/fsx/apoorvgu/conda/bin/conda shell.bash hook)"
# conda activate py310

source ../jax_fs/bin/activate

# echo "Listing apt dependencies"
# apt list --installed | grep neuron
# echo "Listing pip dependencies"
# pip list | grep neuron
# echo "Done listing dependencies"
# printenv | grep NEURON
# printenv | grep XLA
# which python

# TC MALLOC HACK
LIBTCMALLOC=$(find /usr/lib/x86_64-linux-gnu -name "libtcmalloc.so.*" | sort -V | tail -n 1)
 
if [ -n "$LIBTCMALLOC" ]; then
	# Create a symbolic link to the found libtcmalloc version
	sudo ln -sf "$LIBTCMALLOC" /usr/lib/libtcmalloc.so
	echo "Symbolic link created: /usr/lib/libtcmalloc.so -> $LIBTCMALLOC"

	# Export LD_PRELOAD
	export LD_PRELOAD=/usr/lib/libtcmalloc.so
	echo "LD_PRELOAD set to: $LD_PRELOAD"
else
	echo "Error: libtcmalloc.so not found"
	exit 1
fi

OUTPUT_DIR="${TEST_ARTIFACTS_PATH}/axlearn_out"
mkdir -p ${OUTPUT_DIR}
DATA_DIR="gs://axlearn-public/tensorflow_datasets"
#fuji-7B-v2-flash
python -m axlearn.common.mixture_of_experts_test TransformerFeedForwardMoETest