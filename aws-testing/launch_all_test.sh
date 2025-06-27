id=$(date +"%Y%m%d_%H%M%S")

TEST_LOGDIR=test_artifacts/$id
GOLDENS_DIR="test_goldens"
# JAX_CC_DIR="test_artifacts/jax_cc_cache_$id"
JAX_CC_DIR=""

for suite in "presubmit" "12b" "50b" "150b" "qwen3-30b" "switch-base" "switch-large" "mixtral-50b" "llama4-scout" "deepseek-v3" "qwen3-235b" "switch-xxl" "llama4-maverick"; do
    sbatch -W --exclusive -J rh_test_$suite --output=test_artifacts/$id/%x_%j.out ./aws-testing/test.slurm $suite $TEST_LOGDIR $GOLDENS_DIR $JAX_CC_DIR &
done    

echo "All tests launched with logdir: $TEST_LOGDIR"
wait
echo "All tests finished"

bash ./aws-testing/push_to_spectometer.sh $TEST_LOGDIR 0 > $TEST_LOGDIR/push_to_spectometer.log

num_pushed=$(grep -e 'Pushing' $TEST_LOGDIR/push_to_spectometer.log | wc -l)
echo "Pushed $num_pushed test artifacts to spectometer, log saved to $TEST_LOGDIR/push_to_spectometer.log"

python ./aws-testing/parse_pytest_results.py $TEST_LOGDIR