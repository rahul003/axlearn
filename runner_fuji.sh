#!/usr/bin/env bash

# Neuron env vars for distributed training based on SLURM
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
num_nodes=$(echo "$nodes" | wc -l)
VNC=2
devices_per_node=$(( 128 / VNC ))
MASTER_ADDR=$(echo "$nodes" | head -n 1)
MASTER_PORT=41000
JAX_COORDINATOR_PORT=41001
export NEURON_RT_ROOT_COMM_ID="${MASTER_ADDR}:${MASTER_PORT}"
export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf '%s,' $(seq 1 $num_nodes | xargs -I {} echo $devices_per_node) | sed 's/,$//')
export NEURON_PJRT_PROCESS_INDEX=$SLURM_NODEID

sudo apt update
sudo apt-get -f install -y
sudo apt-get install -y google-perftools

sudo dpkg -i --force-all /fsx/mengchiy/aws-neuronx-collectives-2.x.31306.0-76169865b.deb
sudo dpkg -i --force-all /fsx/mengchiy/aws-neuronx-runtime-lib-2.x.30590.0-2bd2ab0ac.deb

sudo dpkg -i --force-all /fsx/apoorvgu/aws-neuronx-tools-2.0.15684.0.deb
# if ! apt list 2>/dev/null | grep -q "^aws-neuronx-dkms/now 2.x.4989.0 amd64 \[installed,local\]"; then sudo dpkg -i --force-all /fsx/apoorvgu/temp/hw_dge/aws-neuronx-dkms_2.x.4989.0_amd64.deb   ; fi
# CHECK_STATUS=$?
# if [ $CHECK_STATUS -ne 0 ]; then
#     echo "Driver version check failed! Terminating job."
#     exit 1
# fi

hostname

# SLURM_JOB_ID="test"
JOB_ID=${SLURM_JOB_ID}
ARTIFACTS_PATH="/fsx/mengchiy/apoorv/artifacts/"
# TIMESTAMP=$(date +"%y%m%d%H%M%S")
TEST_ARTIFACTS_PATH="${ARTIFACTS_PATH}/${JOB_ID}"
mkdir -p "$TEST_ARTIFACTS_PATH"

NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump
HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump
NEURON_RT_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_rt_dump
mkdir -p "$NEURON_RT_DUMP_PATH"


export NEURON_RT_DBG_CC_DMA_PACKET_SIZE=65536,4096
export NEURON_RT_DBG_DMA_PACKETIZATION_SIZE=104857
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1

export NEURON_ENABLE_INT_MATMUL_DOWNCAST=1
export NEURON_FSDP_NUM_LAYER_COALESCE=1
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1
# export NEURON_RT_CC_ALG_TYPES=^INTER_RDH_ALG
# export NEURON_DISABLE_BOUNDARY_MARKER=1
export NEURON_WHILE_LOOP_UNROLL=1
# Neuron runtime flags
export NEURON_RT_IO_RING_CACHE_SIZE=0
# export NEURON_RT_ENABLE_MEMORY_METRICS=0
export NEURON_RT_VIRTUAL_CORE_SIZE=$VNC
export NEURON_RT_RESET_CORES=1
export NEURON_RT_LOG_LEVEL="WARNING"
export NEURON_RT_ENABLE_INTERNODE_EXECUTION_BARRIER=1
export NEURON_SCRATCHPAD_PAGE_SIZE=2048
export NEURON_INTERNAL_CPU_NUM_THREADS=1
# Runtime trace
# export NEURON_RT_INSPECT_ENABLE=1
# export NEURON_RT_INSPECT_OUTPUT_DIR=$NEURON_RT_DUMP_PATH
# export NEURON_RT_INSPECT_DEVICE_PROFILE=1

# Neuron collectives flag
export FI_LOG_LEVEL="warn"
export OFI_NCCL_PROTOCOL=RDMA
export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE=1
export OFI_NCCL_MR_CACHE_DISABLE=1
export NEURON_RT_DBG_MESH_CC=0
# export NEURON_FSDP_CC_MULTISTREAM=1
export NEURON_COLLECTIVE_PERMUTE_TO_ALL_GATHER=1

# Neuron compiler flags
export NEURON_CC_FLAGS="--framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-max-instruction-limit=20000000"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-num-neuroncores-per-sengine=${VNC}"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-dram-page-size=2048"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-enable-dge-levels spill_reload --internal-backend-options=' --spill-reload-dmas-use-swdge --separate-lifetime-corebarriers --lnc-aware-scheduler --run-shared-allocation-before-post-sched'"
# export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-enable-dge-levels spill_reload --internal-backend-options=' --spill-reload-dmas-use-swdge --separate-lifetime-corebarriers --lnc-aware-scheduler --run-shared-allocation-before-post-sched'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --auto-cast=none"
# export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options=''"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --tensorizer-options='--enable-hoist-fsdp-collectives --enable-d2d-pf-transpose-kernel'"
# export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --tensorizer-options='--enable-hoist-fsdp-collectives'"
repeated=$1
echo "input to script for repeated is $repeated"

if [ "$repeated" = "0" ]; then
    echo "no repeated"
    export NEURON_FSDP=1
    export NEURON_FSDP_NUM_LAYER_EARLY_AG_SHIFT=2
    export NEURON_FSDP_NUM_LAYER_LATE_RS_SHIFT=1
    export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--remat-rope --recursive-layer-det=false --partitioner-max-disj-runs=4'"
elif [ -z "$repeated" ]; then
    echo "no argument provided"
    exit 1
else
    echo "yes repeated"
    export NEURON_FSDP_REPEATED=1
    # export NEURON_FSDP_REPEATED_CC_PIPELINING=1
    export NEURON_DISABLE_MOVEMENT_OF_SLICE_FROM_PARAM=1
    export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--verify-hlo --remat-rope --recursive-layer-det=false --dump-after-to-file=pre-par-pipe-end,post-par-pipe-begin'" #  --schedule-fusion=false'"
    # export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--verify-hlo --recursive-layer-det=false --dump-after-to-file=pre-par-pipe-end,post-par-pipe-begin'" #  --schedule-fusion=false'"
    #export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --verbose debug --internal-compiler-debug-mode=all "
fi
# export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--verify-hlo --remat-rope=false --recursive-layer-det=false --dump-after-to-file=pre-par-pipe-end,post-par-pipe-begin --dump-partitioned-hlo'"
# export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--remat-rope --recursive-layer-det=false'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=${NEURON_DUMP_PATH}"

# conda
source /fsx/mengchiy/apoorv/venv/bin/activate

echo "Listing apt dependencies"
apt list --installed | grep neuron
echo "Listing pip dependencies"
pip list | grep neuron
echo "Done listing dependencies"

export XLA_FLAGS=" --xla_dump_hlo_as_text --xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot,neuron-hierarchical-collectives,neuron_move_all_gather_while_loop --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'"

printenv | grep NEURON
printenv | grep XLA

which python
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
    hostname
    exit 1
fi

mkdir -p ${OUTPUT_DIR}
DATA_DIR="gs://axlearn-public/tensorflow_datasets"

echo "Running Neuron"
OUTPUT_DIR="${TEST_ARTIFACTS_PATH}/axlearn_out"
mkdir -p ${OUTPUT_DIR}

export DATA_SEED=42

# export ENABLE_HOOKS_DUMP=1

backend=$2
echo "input to script for backend is $backend"

if [ "$backend" = "trn" ]; then
    python -m axlearn.common.launch_trainer_main \
        --module=text.gpt.c4_trainer --config=fuji-70B-v2-flash \
        --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
        --jax_backend=neuron --mesh_selector=neuron-trn2.48xlarge-64 \
        --distributed_coordinator=$MASTER_ADDR:$JAX_COORDINATOR_PORT --num_processes=$num_nodes \
        --process_id=$NEURON_PJRT_PROCESS_INDEX 
fi

if [ "$backend" = "cpu" ]; then
    export JAX_PLATFORMS=cpu
    export XLA_FLAGS="${XLA_FLAGS} --xla_force_host_platform_device_count=64"
    python -m axlearn.common.launch_trainer_main \
        --module=text.gpt.c4_trainer --config=fuji-70B-v2-flash \
        --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
        --jax_backend=cpu --mesh_selector=neuron-trn2.48xlarge-64 \
        --distributed_coordinator=$MASTER_ADDR:$JAX_COORDINATOR_PORT --num_processes=$num_nodes \
        --process_id=$NEURON_PJRT_PROCESS_INDEX 
fi