#!/usr/bin/env bash
# Force unbuffered output
export PYTHONUNBUFFERED=1
export ENABLE_HLO_DUMP=1
export ENABLE_DEBUG_FLAGS=1

echo "Current directory: $(pwd)"
ls -la ./deps/

set -x  # Enable command tracing

# Neuron env vars for distributed training based on SLURM
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
num_nodes=$(echo "$nodes" | wc -l)
# devices_per_node=64
# devices_per_node=8
devices_per_node=1
MASTER_ADDR=$(echo "$nodes" | head -n 1)
# MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
MASTER_PORT=41000
JAX_COORDINATOR_PORT=41001
export NEURON_RT_ROOT_COMM_ID="${MASTER_ADDR}:${MASTER_PORT}"
export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf '%s,' $(seq 1 $num_nodes | xargs -I {} echo $devices_per_node) | sed 's/,$//')
export NEURON_PJRT_PROCESS_INDEX=$SLURM_NODEID
# export NEURON_PJRT_PROCESS_INDEX=${SLURM_NODEID:-0}

# Needed for TC_MALLOC fix
sudo apt-get -f install -y || true
sudo apt-get install -y google-perftools libgoogle-perftools-dev || true

# Install Runtime dependencies
sudo dpkg -i ./deps/aws-neuronx-runtime-lib-2.x.32909.0-0c17f5273.deb
sudo dpkg -i ./deps/aws-neuronx-collectives-2.x.34406.0-ba84f10ae.deb
if ! apt list 2>/dev/null | grep -q "^aws-neuronx-dkms amd64 \[installed,local\]"; then sudo dpkg -i --force-all ./deps/aws-neuronx-dkms_2.x.6781.0_amd64.deb || true; fi
echo "Driver installation completed (ignoring DKMS errors)"

# Print nodenames for debug
hostname

JOB_ID=${SLURM_JOB_ID}
ARTIFACTS_PATH="/fsx/ishaniak/axlearn/artifacts"
TEST_ARTIFACTS_PATH="${ARTIFACTS_PATH}/${JOB_ID}"

mkdir -p "$TEST_ARTIFACTS_PATH"
HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump

export TEST_FSX_HOME="${TEST_ARTIFACTS_PATH}/logs"
export XLA_FLAGS="--xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot,neuron-hierarchical-collectives,neuron-token-threading"

if [ $ENABLE_HLO_DUMP = 1 ]; then
    export XLA_FLAGS="${XLA_FLAGS} --xla_dump_hlo_as_text --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'"
fi

if [ $ENABLE_DEBUG_FLAGS = 1 ]; then
    echo "Running with Debug Flags Enabled"
    export NEURON_RT_LOG_LEVEL_NRT=DEBUG
fi

# PJRT Flags 
export NEURON_FSDP_NUM_LAYER_EARLY_AG_SHIFT=1
export NEURON_FSDP_NUM_LAYER_LATE_RS_SHIFT=2
export NEURON_ENABLE_INT_MATMUL_DOWNCAST=1
export NEURON_FSDP=1
export NEURON_FSDP_NUM_LAYER_COALESCE=-1
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1

# Neuron runtime flags

export NEURON_RT_DBG_CC_DMA_PACKET_SIZE=65536,4096
export NEURON_RT_DBG_DMA_PACKETIZATION_SIZE=104857
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1
export NEURON_RT_IO_RING_CACHE_SIZE=0
export NEURON_RT_ENABLE_MEMORY_METRICS=0
# export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_VIRTUAL_CORE_SIZE=1
export NEURON_RT_RESET_CORES=1
export NEURON_RT_LOG_LEVEL="WARNING"
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1
export NEURON_RT_ENABLE_INTERNODE_EXECUTION_BARRIER=1
export NEURON_RT_DBG_CC_INTER_MESH_MAX_NODES=0
export NEURON_RT_DBG_MESH_CC=0
export NEURON_RT_CC_ALG_TYPES=^INTER_RDH_ALG

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
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-enable-dge-levels spill_reload"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --auto-cast=none"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-backend-options='--run-shared-allocation-before-post-sched=true'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --ccop-pipeline-buffer-size=2000"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-num-neuroncores-per-sengine=1"

# JAX Cache
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache"
if [ -d "/tmp/jax_cache" ]; then
    echo "Cache directory exists; deleting it!"
    rm -rf /tmp/jax_cache
else
    echo "Cache directory does not exist"
fi
mkdir -p ${JAX_COMPILATION_CACHE_DIR}

# Check if virtual environment exists and is properly set up
if [ ! -f "./aws_neuron_venv_jax/bin/python" ]; then
    echo "Error: Virtual environment not found or Python not available"
    exit 1
fi
source aws_neuron_venv_jax/bin/activate

# Debug: Check if venv activation worked properly
echo "=== Virtual Environment Debug ==="
echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo "PATH: $PATH"
echo "which python: $(which python)"
echo "which neuronx-cc: $(which neuronx-cc)"
echo "python executable: $(python --version)"
echo "==================================="

echo "Uninstalling old axlearn and installing from correct location..."
pip uninstall axlearn -y || true
pip install -e .

# echo "Uninstalling old axlearn and installing from correct location..."
# pip uninstall axlearn -y || true
# pip install -e .
echo "Axlearn reinstalled from $(pwd)"

echo "Listing apt dependencies"
apt list --installed | grep neuron
echo "Listing pip dependencies"
pip list | grep neuron
echo "Done listing dependencies"
printenv | grep NEURON
printenv | grep XLA
which python

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

if [ $ENABLE_JAX_COMPILATION_CACHE = 1 ]; then
    export JAX_COMPILATION_CACHE_DIR="$ARTIFACTS_HOME/axlearn/cc_cache/"
    mkdir -p $JAX_COMPILATION_CACHE_DIR
fi

# Neuron PJRT flags
export NEURON_WHILE_LOOP_UNROLL=1
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1
export NEURON_FSDP_NUM_LAYER_COALESCE=1
export NEURON_ENABLE_INT_MATMUL_DOWNCAST=1
export NEURON_FSDP_CC_MULTISTREAM=1
export DATA_SEED=42
export MODEL_SEED=42

export STACK_CFG="REPEATED"
# Set flags based on STACK_CFG
# if [ "${STACK_CFG}" == "STACKED" ] || [ -z "${STACK_CFG}" ]; then
#     export NEURON_FSDP=1
#     export NEURON_FSDP_NUM_LAYER_EARLY_AG_SHIFT=1
#     export NEURON_FSDP_NUM_LAYER_LATE_RS_SHIFT=2
#     export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --tensorizer-options='--enable-hoist-fsdp-collectives'"
#     export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--remat-rope --disable-early-opt-barrier-removal'"
# elif [ "${STACK_CFG}" == "REPEATED" ]; then
export NEURON_INTERNAL_CPU_NUM_THREADS=1
export NEURON_FSDP_REPEATED=1
export NEURON_DISABLE_MOVEMENT_OF_SLICE_FROM_PARAM=1
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --tensorizer-options='--enable-hoist-fsdp-collectives --enable-d2d-pf-transpose-kernel'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--verify-hlo --remat-rope --recursive-layer-det=false'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --auto-cast=none"
# fi

source env_source.sh


mkdir -p ${OUTPUT_DIR}/checkpoints
cp -r /fsx/ishaniak/axlearn/initialization_checks/fp32_fuji_smol/checkpoints/step_00000000 ${OUTPUT_DIR}/checkpoints/
# cp -r /fsx/ishaniak/axlearn/initialization_checks/bf16_fuji_1b/checkpoints/step_00000000 ${OUTPUT_DIR}/checkpoints/

# # Create neuron dump directory and set up NEFF file copying
# mkdir -p ${NEURON_DUMP_PATH}
# trap 'find /tmp -name "*.neff" -exec cp {} ${NEURON_DUMP_PATH}/ \; 2>/dev/null || true' EXIT

python -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-1B-v2-flash \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
    --jax_backend=neuron --mesh_selector=neuron-trn2.48xlarge-64 \
    --distributed_coordinator=$MASTER_ADDR:$JAX_COORDINATOR_PORT --num_processes=$num_nodes \
    --process_id=$NEURON_PJRT_PROCESS_INDEX \
    --max_step=10