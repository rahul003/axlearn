
SUITE_ARG=${1:-"all"}
PUSH_ARTIFACTS_TO_S3_FOR_SPECTOMETER=${2:-"0"}
XFAIL_KNOWN_FAILURES=${3:-"1"}
RESUME_TESTS_ID=${4:-""}

if [ -z "$RESUME_TESTS_ID" ]; then
    id=$(date +"%Y%m%d_%H%M%S")
else
    id=$RESUME_TESTS_ID
fi

function run_suite() {
    sbatch -W --partition=dev --exclusive -J pipeline_test_$1 --output=test_artifacts/$id/%x_%j.out ./aws-testing/test.slurm $1 $TEST_LOGDIR $GOLDENS_DIR $JAX_CC_DIR &
}


TEST_LOGDIR=test_artifacts/$id
GOLDENS_DIR="/shared/huilgolr/axlearn/test_goldens"
# JAX_CC_DIR="test_artifacts/jax_cc_cache_$id"
JAX_CC_DIR=""

if [ "$SUITE_ARG" = "all" ]; then
    for suite in "presubmit" "12b" "50b" "150b" "qwen3-30b" "switch-base" "switch-large" "mixtral-50b" "llama4-scout" "deepseek-v3" "qwen3-235b" "switch-xxl" "llama4-maverick" "gpt-oss"; do
        run_suite $suite
    done
elif [ "$SUITE_ARG" = "quick_set" ]; then
    for suite in "presubmit" "12b" "50b" "qwen3-235b" "switch-xxl" "llama4-maverick" "150b"; do
        run_suite $suite
    done
elif [ "$SUITE_ARG" = "slow_set" ]; then
    for suite in "qwen3-30b" "switch-base" "switch-large" "mixtral-50b" "llama4-scout" "deepseek-v3"; do
        run_suite $suite
    done
else
    run_suite $SUITE_ARG
fi
echo "All tests launched with logdir: $TEST_LOGDIR"
wait
echo "All tests finished"

if [ "$PUSH_ARTIFACTS_TO_S3_FOR_SPECTOMETER" = "1" ]; then
    bash ./aws-testing/push_to_spectometer.sh $TEST_LOGDIR 0 > $TEST_LOGDIR/push_to_spectometer.log
    num_pushed=$(grep -e 'Pushing' $TEST_LOGDIR/push_to_spectometer.log | wc -l)
    echo "Pushed $num_pushed test artifacts to spectometer, log saved to $TEST_LOGDIR/push_to_spectometer.log"
else
    echo "Skipping pushing artifacts to S3 for spectometer"
fi

if [ "$XFAIL_KNOWN_FAILURES" = "1" ]; then
    # XFAIL like behavior
    # Fetch prev results, and ensure no new failure.
    # If there's a new failure, we will fail this test stage in pipeline.
    python ./aws-testing/parse_pytest_results.py --artifacts_dir $TEST_LOGDIR --load_known_failures /shared/huilgolr/axlearn/test_artifacts/pipeline-failures/jul-25-failures.txt
fi

