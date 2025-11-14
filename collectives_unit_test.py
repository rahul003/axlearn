import jax
import jax.numpy as jnp
import os

def main():
    precision = os.environ.get('PRECISION', 'fp32')
    if precision == 'fp32': 
        TYPE = jnp.float32
    elif precision == 'bf16': 
        TYPE = jnp.bfloat16
    else:
        raise ValueError(f"Unknown precision: {precision}")

    num_devices = jax.device_count()
    print(f"Total devices available: {num_devices}")
    print(f"Local devices: {jax.local_devices()}")
    print(f"Data type: {precision}")
    
    # Configuration
    seq_len = 4096
    batch_size = 1
    hidden_size = 5120
    test_runs = 1
    
    for i in range(test_runs):
        # Create different random data for each device
        key = jax.random.PRNGKey(42 + i)
        keys = jax.random.split(key, num_devices)
        
        # Each device gets different input: (num_devices, seq_len, batch_size, hidden_size)
        x = jax.vmap(lambda k: jax.random.uniform(k, (seq_len, batch_size, hidden_size), dtype=TYPE))(keys)
        
        print(f"\nTest run {i}:")
        print(f"Input shape per device: {x.shape[1:]}, dtype: {x.dtype}")
        print(f"Total input shape: {x.shape}")
        
        @jax.jit
        def allreduce(x):
            # All-reduce: sum across all devices
            return jax.lax.psum(x, axis_name='i')
        
        # Run all-reduce
        ar_output = jax.pmap(allreduce, axis_name='i')(x)
        
        print(f"All-reduce output shape: {ar_output.shape}")
        
        # Verify correctness
        # 1. All devices should have identical results
        # 2. Result should equal sum of all inputs
        expected = jnp.sum(x, axis=0)  # Sum across device dimension
        
        all_match = True
        for rank in range(num_devices):
            # Calculate diff stats always
            diff = ar_output[rank] - expected
            abs_diff = jnp.abs(diff)
            
            # Check for exact equality
            exact_match = jnp.all(ar_output[rank] == expected)
            num_exact_matches = int(jnp.sum(ar_output[rank] == expected))
            num_mismatches = ar_output[rank].size - num_exact_matches
            
            stats = {
                'max_diff': float(jnp.max(abs_diff)),
                'min_diff': float(jnp.min(abs_diff)),
                'mean_diff': float(jnp.mean(abs_diff)),
                'num_mismatches': num_mismatches,
                'output_mean': float(jnp.mean(ar_output[rank])),
                'expected_mean': float(jnp.mean(expected)),
            }
            
            print(f"\nRank {rank}:")
            print(f"  Exact matches: {num_exact_matches}, Mismatches: {num_mismatches}")
            print(f"  Diff stats - min: {stats['min_diff']:.10f}, max: {stats['max_diff']:.10f}, mean: {stats['mean_diff']:.10f}")
            print(f"  Output mean: {stats['output_mean']:.6f}, Expected mean: {stats['expected_mean']:.6f}")
            
            if not exact_match:
                all_match = False
                print(f"  run{i} rank{rank} does not match exactly")
            else:
                print(f"  run{i} rank{rank} matches exactly")
        
        # Check all devices have identical results
        for rank in range(1, num_devices):
            if not jnp.all(ar_output[0] == ar_output[rank]):
                print(f"\n Devices 0 and {rank} have different results!")
                all_match = False
        
        if all_match:
            print(f"\n Test run {i} PASSED! All {num_devices} devices have identical results matching expected sum")
        else:
            print(f"\n Test run {i} FAILED!")

if __name__ == "__main__":
    main()