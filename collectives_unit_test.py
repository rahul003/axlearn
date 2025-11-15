import jax
import jax.numpy as jnp
import pickle
import os
from pathlib import Path

def main():
    precision = os.environ.get('PRECISION', 'fp32')
    if precision == 'fp32': 
        TYPE = jnp.float32
    elif precision == 'bf16': 
        TYPE = jnp.bfloat16
    else:
        raise ValueError(f"Unknown precision: {precision}")

    # Get regenerate flag and output path
    regenerate = os.environ.get('REGENERATE', 'false').lower() in ('true', '1', 'yes')
    output_path = os.environ.get('DATA_DIR', './output')
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_devices = jax.device_count()
    print(f"Total devices available: {num_devices}")
    print(f"Local devices: {jax.local_devices()}")
    print(f"Data type: {precision}")
    print(f"Regenerate: {regenerate}")
    print(f"Output path: {output_path}")
    
    # Configuration
    seq_len = 4096
    batch_size = 1
    hidden_size = 5120
    test_runs = 10
    
    for run in range(test_runs):
        print(f"\n{'='*60}")
        print(f"Test run {run}")
        print(f"{'='*60}")
        
        input_file = output_path / f"inputs_run{run}.pkl"
        output_file = output_path / f"outputs_run{run}.pkl"
        
        if regenerate:
            print(f"\nGenerating new data...")
            # Create different random data for each device
            key = jax.random.PRNGKey(42 + run)
            keys = jax.random.split(key, num_devices)
            
            # Each device gets different input
            x = jax.vmap(lambda k: jax.random.uniform(k, (seq_len, batch_size, hidden_size), dtype=TYPE))(keys)
            x = jax.block_until_ready(x)
            
            # Save inputs
            with open(input_file, 'wb') as f:
                pickle.dump(x, f)
            print(f"  Saved inputs to: {input_file}")
        else:
            print(f"\nLoading existing data from: {input_file}")
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}. Run with REGENERATE=true first.")
            
            with open(input_file, 'rb') as f:
                x = pickle.load(f)
            print(f"  Loaded inputs with shape: {x.shape}, dtype: {x.dtype}")
        
        print(f"\nInput data:")
        print(f"  Shape per device: {x.shape[1:]}, dtype: {x.dtype}")
        
        # Print input stats for each device
        print(f"\n  Per-device input statistics:")
        individual_means = []
        for rank in range(num_devices):
            mean_val = float(jnp.mean(x[rank]))
            individual_means.append(mean_val)
            print(f"    Device {rank}: mean={mean_val:.10f}, first_elem={float(x[rank, 0, 0, 0]):.6f}")
        
        # Run all-reduce
        @jax.jit
        def allreduce(x):
            return jax.lax.psum(x, axis_name='i')
        
        ar_output = jax.pmap(allreduce, axis_name='i')(x)
        ar_output = jax.block_until_ready(ar_output)
        
        print(f"\nAll-reduce output:")
        print(f"  Shape: {ar_output.shape}")
        
        # All devices should have identical outputs
        print(f"\n{'='*60}")
        print(f"CHECK: Device output consistency")
        print(f"{'='*60}")
        all_identical = True
        for rank in range(1, num_devices):
            matches = jnp.all(ar_output[0] == ar_output[rank])
            if not matches:
                diff = jnp.abs(ar_output[0] - ar_output[rank])
                print(f"  ✗ Device 0 vs {rank}: DIFFER (max diff: {float(jnp.max(diff)):.10f})")
                all_identical = False
            else:
                print(f"  ✓ Device 0 vs {rank}: IDENTICAL")
        
        if all_identical:
            print(f"\n✓ PASS: All devices have identical outputs")
        else:
            print(f"\n✗ FAIL: Devices have different outputs!")
            continue

        # Save output (rank 0 only, all ranks verified identical)
        with open(output_file, 'wb') as f:
            pickle.dump(ar_output[0], f)
        print(f"\nSaved output (rank 0, all ranks verified identical) to: {output_file}")

if __name__ == "__main__":
    main()