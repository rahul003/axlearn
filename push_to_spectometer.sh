TEST_ARTIFACTS_DIR=${1}
DRY_RUN=${2:-1}

if [ -z "$TEST_ARTIFACTS_DIR" ]; then
  echo "Usage: $0 <test_artifacts_dir>"
  exit 1
fi

# for each directory in TEST_ARTIFACTS_DIR, run the following command
for dir in "$TEST_ARTIFACTS_DIR"/*; do
  if [ -d "$dir" ]; then
    echo "Processing $dir"
    for test in "$dir/artifacts/neuron_dump"/*; do
        base_test=$(basename "$test")
        # check if $test/model.hlo_module.pb exists
        if [ ! -f "$test/model.hlo_module.pb" ]; then
            echo "Skipping $test/model.hlo_module.pb, file does not exist"
            continue
        fi
        echo "Pushing $base_test to spectometer"
        cmd="aws s3 cp $test/hlo_metadata.json s3://kaena-nn-models/spectometer-staging/training-moe-jax-integration-tests/$base_test/"
        cmd2="aws s3 cp $test/model.hlo_module.pb s3://kaena-nn-models/spectometer-staging/training-moe-jax-integration-tests/$base_test/"
        if [ "$DRY_RUN" -eq 0 ]; then
            # exec cmd
            eval "$cmd"
            eval "$cmd2"
        else
            echo "$cmd"
            echo "$cmd2"
        fi
    done
  fi
done