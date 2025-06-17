#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name rh_test
#SBATCH --output=test_logs/%x_%j.out
#SBATCH --cpus-per-task 127

set -ex

srun -l setup_node.sh /fsx/huilgolr/may-artifacts/

SUITE_ARG=${1:-"presubmit"}
TEST_LOGDIR=${2:-"test_logs/$SLURM_JOB_ID"}
GOLDENS_DIR=${3:-"test_goldens"}
JAX_CC_DIR=${4:-"test_jax_cc"}

if [ -z "$SUITE_ARG" ]; then
    echo "Usage: $0 <test_suite> [<tid>]"
    echo "Example: $0 150b 12345"
    exit 1
fi

mkdir -p $TEST_LOGDIR

function summary() {
    echo "---------------------------------"
    suite=$1
    num_passed=$(grep -re 'PASSED' $TEST_LOGDIR/$suite/*.log | wc -l)
    num_failed=$(grep -re 'FAILED' $TEST_LOGDIR/$suite/*.log | wc -l)
    total_num=$(($num_passed + $num_failed))
    echo "Test suite: $suite, Total tests $total_num"
    echo "Number of tests passed: $num_passed"
    echo "Number of tests failed: $num_failed"
    if [ $num_failed -gt 0 ]; then
        echo "Failed tests:"
        grep -hre 'FAILED' $TEST_LOGDIR/$suite/*.log
    fi
}

function run_suite() {
    suite=$1
    echo "Running test suite: $suite"
    mkdir -p $TEST_LOGDIR/${suite}
    if [ "$suite" = "150b" ]; then
        srun -l unit_test.sh 150bdev ${suite} $TEST_LOGDIR $GOLDENS_DIR $JAX_CC_DIR 2>&1 | tee $TEST_LOGDIR/${suite}/150bdev.log
    fi
    srun -l unit_test.sh integ $suite $TEST_LOGDIR $GOLDENS_DIR $JAX_CC_DIR 2>&1 | tee $TEST_LOGDIR/${suite}/integ.log
    #srun -l unit_test.sh unit $suite $TEST_LOGDIR $GOLDENS_DIR $JAX_CC_DIR 2>&1 | tee $TEST_LOGDIR/${suite}/unit.log
    summary $suite
}

run_suite $SUITE_ARG
