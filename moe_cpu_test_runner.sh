#!/usr/bin/env bash
set -e

# Neuron env vars for distributed training based on SLURM
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ -z "$SLURM_JOB_NODELIST" ]; then
	nodes="localhost"
	SLURM_NODEID=0
	SLURM_JOB_ID=$EPOCHSECONDS
fi

# Print nodenames for debug
hostname

JOB_ID=${SLURM_JOB_ID}
ARTIFACTS_PATH="test_artifacts"
TEST_ARTIFACTS_PATH="${ARTIFACTS_PATH}/${JOB_ID}"
mkdir -p "$TEST_ARTIFACTS_PATH"
HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump
export XLA_FLAGS="--xla_dump_hlo_as_text --xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot,neuron-hierarchical-collectives --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'"

# Neuron collectives flag
export FI_LOG_LEVEL="warn"
export OFI_NCCL_PROTOCOL=RDMA
export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE=1
export OFI_NCCL_MR_CACHE_DISABLE=1
export JAX_ENABLE_X64=1

# JAX Cache
export JAX_COMPILATION_CACHE_DIR="cache/"
mkdir -p ${JAX_COMPILATION_CACHE_DIR}

deactivate > /dev/null || true
source ../jaxcpu/bin/activate

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

# python -m axlearn.common.mixture_of_experts_test TransformerFeedForwardMoETest.test_topk_gating_gather
python -m axlearn.common.mixture_of_experts_test TransformerFeedForwardMoETest