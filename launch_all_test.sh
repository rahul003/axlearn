id=$(date +"%Y%m%d_%H%M%S")
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

TEST_LOGDIR=test_artifacts/$id
GOLDENS_DIR="test_goldens"
JAX_CC_DIR="test_artifacts/jax_cc_cache"

sbatch -W --exclusive -J rh_test_presubmit --output=test_artifacts/$id/%x_%j.out test.slurm presubmit $TEST_LOGDIR $GOLDENS_DIR $JAX_CC_DIR &
sbatch -W --exclusive -J rh_test_12b --output=test_artifacts/$id/%x_%j.out test.slurm 12b $TEST_LOGDIR $GOLDENS_DIR $JAX_CC_DIR &
sbatch -W --exclusive -J rh_test_50b --output=test_artifacts/$id/%x_%j.out test.slurm 50b $TEST_LOGDIR $GOLDENS_DIR $JAX_CC_DIR &
sbatch -W --exclusive -J rh_test_150b --output=test_artifacts/$id/%x_%j.out test.slurm 150b $TEST_LOGDIR $GOLDENS_DIR $JAX_CC_DIR &
sbatch -W --exclusive -J rh_test_small --output=test_artifacts/$id/%x_%j.out test.slurm small_models $TEST_LOGDIR $GOLDENS_DIR $JAX_CC_DIR &
sbatch -W --exclusive -J rh_test_large --output=test_artifacts/$id/%x_%j.out test.slurm large_models $TEST_LOGDIR $GOLDENS_DIR $JAX_CC_DIR &

echo "All tests launched with logdir: $TEST_LOGDIR"
wait
echo "All tests finished"
bash push_to_spectometer.sh $TEST_LOGDIR 0 > $TEST_LOGDIR/push_to_spectometer.log
num_pushed=$(grep -e 'Pushing' $TEST_LOGDIR/push_to_spectometer.log | wc -l)
echo "Pushed $num_pushed test artifacts to spectometer, log saved to $TEST_LOGDIR/push_to_spectometer.log"
for suite in "presubmit" "12b" "50b" "150b" "small_models" "large_models"; do
    summary $suite
done