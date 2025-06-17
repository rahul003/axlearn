#!/usr/bin/env bash
set -e

if [ -z "$VENV_NAME" ]; then
	VENV_NAME=../jaxmoe
fi

source $VENV_NAME/bin/activate

export TEST_SUITE=${2:-"presubmit"}
export TEST_LOG_DIR=${3:-"test_logs/shell"}
export GOLDENS_DIR=${4:-"test_goldens"}
export JAX_COMPILATION_CACHE_DIR=${5:-"test_jax_cc"}


mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
mkdir -p "${GOLDENS_DIR}"

export TEST_ARTIFACTS_PATH=$TEST_LOG_DIR/$TEST_SUITE/artifacts
export NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump
mkdir -p "$TEST_ARTIFACTS_PATH"

export USE_CACHED_GOLDENS=1
export CACHE_GOLDENS=1
export USE_SHARDMAP_FFN=1

export XLA_FLAGS="--xla_cpu_use_thunk_runtime=false --xla_force_host_platform_device_count=64 --xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot,neuron-hierarchical-collectives"

export GIT_COMMIT=$(git rev-parse --short HEAD)

# HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump
# export XLA_FLAGS="${XLA_FLAGS} --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*' --xla_dump_hlo_as_text"
# export XLA_FLAGS="${XLA_FLAGS} --xla_dump_hlo_snapshots"

# PJRT Flags 
export NEURON_FSDP_NUM_LAYER_EARLY_AG_SHIFT=1
export NEURON_FSDP_NUM_LAYER_LATE_RS_SHIFT=2
export NEURON_ENABLE_INT_MATMUL_DOWNCAST=1
export NEURON_FSDP=0
export NEURON_FSDP_NUM_LAYER_COALESCE=-1
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1 # changed from 0
export NEURON_DISABLE_BOUNDARY_MARKER=1

# Neuron runtime flags
export NEURON_RT_DBG_CC_DMA_PACKET_SIZE=4096 && export NEURON_RT_DBG_DMA_PACKETIZATION_SIZE=104857
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1
export NEURON_RT_IO_RING_CACHE_SIZE=0
export NEURON_RT_ENABLE_MEMORY_METRICS=0
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_RESET_CORES=1
export NEURON_RT_LOG_LEVEL="ERROR"
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
export NEURON_CC_FLAGS_BASE="--framework=XLA"
export NEURON_CC_FLAGS_BASE="${NEURON_CC_FLAGS_BASE} --internal-max-instruction-limit=20000000"
export NEURON_CC_FLAGS_BASE="${NEURON_CC_FLAGS_BASE} --target=trn2"
export NEURON_CC_FLAGS_BASE="${NEURON_CC_FLAGS_BASE} --internal-num-neuroncores-per-sengine=2"
export NEURON_CC_FLAGS_BASE="${NEURON_CC_FLAGS_BASE} --model-type transformer"
export NEURON_CC_FLAGS_BASE="${NEURON_CC_FLAGS_BASE} --no-internal-hlo-remat"
export NEURON_CC_FLAGS_BASE="${NEURON_CC_FLAGS_BASE} --enable-mixed-precision-accumulation"
export NEURON_CC_FLAGS_BASE="${NEURON_CC_FLAGS_BASE} --internal-hlo2tensorizer-options=--verify-hlo"
export NEURON_CC_FLAGS_BASE="${NEURON_CC_FLAGS_BASE} -O1"
export NEURON_CC_FLAGS_BASE="${NEURON_CC_FLAGS_BASE} --tensorizer-options='--enable-hoist-fsdp-collectives'"
export NEURON_CC_FLAGS_BASE="${NEURON_CC_FLAGS_BASE} --internal-hlo2tensorizer-options='--remat-rope --verify-hlo'"
export NEURON_CC_FLAGS_BASE="${NEURON_CC_FLAGS_BASE} --auto-cast=none --hbm-scratchpad-page-size=1024"

export NEURON_SCRATCHPAD_PAGE_SIZE=1024
export TF_CPP_MIN_LOG_LEVEL=3

set -ex

if [ "$1" = "unit" ]; then
    export JAX_PLATFORMS=cpu
    pytest -rsA --tb=short --junitxml=$TEST_LOG_DIR/$TEST_SUITE/unit.xml axlearn/common/mixture_of_experts_neuron_test.py::TestLayerOnCpu
elif [ "$1" = "integ" ]; then
    pytest -rsA --tb=short --junitxml=$TEST_LOG_DIR/$TEST_SUITE/integ.xml axlearn/common/mixture_of_experts_neuron_test.py::TestLayerOnTrn
elif [ "$1" = "150bdev" ]; then
    export TEST_SUITE="150b"
    pytest -rsA --tb=short --junitxml=$TEST_LOG_DIR/$TEST_SUITE/150bdev_integ.xml axlearn/common/mixture_of_experts_neuron_test.py -k "TestDev150bInteg or TestDev150bGatingInteg"
    export JAX_PLATFORMS=cpu
    pytest -rsA --tb=short --junitxml=$TEST_LOG_DIR/$TEST_SUITE/150bdev_unit.xml axlearn/common/mixture_of_experts_neuron_test.py -k "TestDev150bUnit or TestDev150bGatingUnit"
elif [ "$1" = "dev" ]; then
    pytest -rsA axlearn/common/mixture_of_experts_neuron_test.py -k "TestLayerOnTrn and test_fwdbwd_blockwisegather_MoE_i8192_h20480_e1_topk8_g1_ec2_blocksize512_b1_s8192_meshfsdp-1tp64_bf16"
elif [ "$1" = "150b_gather" ]; then
    pytest -rsA --tb=short axlearn/common/mixture_of_experts_neuron_test.py -k 'TestDev150bUnit and test_fwd_gather_vs_einsum or TestDev150bUnit and test_fwdbwd_gather_vs_einsum or TestDev150bInteg and test_fwd_gather_vs_einsum or TestDev150bInteg and test_fwdbwd_gather_vs_einsum'
elif [ "$1" = "150b_blockwise" ]; then
    pytest -rsA --tb=short axlearn/common/mixture_of_experts_neuron_test.py -k 'TestDev150bUnit and test_fwd_blockwise_vs_einsum or TestDev150bUnit and test_fwdbwd_blockwise_vs_einsum or TestDev150bInteg and test_fwd_blockwise_vs_einsum or TestDev150bInteg and test_fwdbwd_blockwise_vs_einsum'
    pytest -rsA --tb=short axlearn/common/mixture_of_experts_neuron_test.py -k 'TestDev150bUnit and test_fwd_blockwisev2_vs_einsum or TestDev150bUnit and test_fwdbwd_blockwisev2_vs_einsum or TestDev150bInteg and test_fwd_blockwisev2_vs_einsum or TestDev150bInteg and test_fwdbwd_blockwisev2_vs_einsum'
elif [ "$1" = "150b_blockwise_cpu" ]; then
    pytest -rsA --tb=short axlearn/common/mixture_of_experts_neuron_test.py -k 'TestDev150bUnit and test_fwd_blockwise_vs_einsum or TestDev150bUnit and test_fwdbwd_blockwise_vs_einsum'
    pytest -rsA --tb=short axlearn/common/mixture_of_experts_neuron_test.py -k 'TestDev150bUnit and test_fwd_blockwisev2_vs_einsum or TestDev150bUnit and test_fwdbwd_blockwisev2_vs_einsum'
elif [ "$1" = "150b_blockwise_neuron" ]; then
    pytest -rsA --tb=short axlearn/common/mixture_of_experts_neuron_test.py -k 'TestDev150bInteg and test_fwd_blockwise_vs_einsum or TestDev150bInteg and test_fwdbwd_blockwise_vs_einsum'
    pytest -rsA --tb=short axlearn/common/mixture_of_experts_neuron_test.py -k 'TestDev150bInteg and test_fwd_blockwisev2_vs_einsum or TestDev150bInteg and test_fwdbwd_blockwisev2_vs_einsum'
fi
