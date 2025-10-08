from axlearn.common.checkpointer import read_state_spec, TensorStoreStateStorage
import jax
import jax.numpy as jnp
import os
import ast
import argparse

def compare_checkpoints(dir1, dir2): 

    # Checkpoint directories
    ckpt1_base = os.path.join(dir1, "checkpoints")
    ckpt2_base = os.path.join(dir2, "checkpoints")

    # Create simple mesh for CPU
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, ['data'])

    print("Mesh devices:", mesh.devices)
    print("Mesh shape:", mesh.shape) 
    print("Mesh axis names:", mesh.axis_names)

    state_spec1 = read_state_spec(ckpt1_base + "/step_00000000")
    state_spec2 = read_state_spec(ckpt2_base + "/step_00000000")
    storage1 = TensorStoreStateStorage.default_config().instantiate()
    storage2 = TensorStoreStateStorage.default_config().instantiate()

    with mesh:
        state1 = storage1.restore_from_dir(0, state_spec1, ckpt_dir=ckpt1_base + "/step_00000000")
        state2 = storage2.restore_from_dir(0, state_spec2, ckpt_dir=ckpt2_base + "/step_00000000")

    def print_dict_structure(d, prefix="", max_depth=10, current_depth=0):
        if current_depth >= max_depth:
            return
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}/")
                print_dict_structure(value, prefix + "  ", max_depth, current_depth + 1)
            else:
                print(f"{prefix}{key}: {type(value)} {getattr(value, 'shape', '')}")

    # Print first 10 parameters
    model_params1 = jax.tree.leaves(state1['model'])
    model_params2 = jax.tree.leaves(state2['model'])
    param_paths, _ = jax.tree_util.tree_flatten_with_path(state1['model'])
    param_names = ['.'.join(str(k.key) for k in path) for path, _ in param_paths]

    for i, (name, p1, p2) in enumerate(zip(param_names[:10], model_params1[:10], model_params2[:10])):
        print(f"Param {i}: {name}")
        print(f"  Shape: {p1.shape}, Dtype: {p1.dtype}")
        print(f"  First few values - State1: {p1.flatten()[:5]}")
        print(f"  First few values - State2: {p2.flatten()[:5]}")
        print(f"  Equal: {jnp.allclose(p1, p2)}")
        print()

    try:
        weights_equal = jax.tree.map(lambda x, y: jnp.allclose(x, y), state1['model'], state2['model'])
        print("Weights identical:", all(jax.tree.leaves(weights_equal)))
        
        if not all(jax.tree.leaves(weights_equal)): 
            print("============== TEST FAILED - WEIGHT INITIALIZATION IS INCONSISTENT =============")
            return 1
        print("============== TEST PASSED - WEIGHT INITIALIZATION IS CONSISTENT =============")
        return 0
    except Exception as e:
        print(f"Error during comparison: {e}")
        return 1

def compare_data_loader(dir1, dir2):

    # Log directories
    log_dir1 = os.path.join(dir1, "input_logs")
    log_dir2 = os.path.join(dir2, "input_logs")

    # Get all log files
    files1 = sorted([f for f in os.listdir(log_dir1) if f.endswith('_input_ids.txt')])
    files2 = sorted([f for f in os.listdir(log_dir2) if f.endswith('_input_ids.txt')])

    print(f"Found {len(files1)} files in dir1, {len(files2)} files in dir2")

    # Compare each step
    for f1, f2 in zip(files1, files2):
        print(f"\nComparing {f1} vs {f2}")
        
        # Load data
        with open(os.path.join(log_dir1, f1), 'r') as file:
            data1 = ast.literal_eval(file.read())
        with open(os.path.join(log_dir2, f2), 'r') as file:
            data2 = ast.literal_eval(file.read())
        
        # Convert to arrays for comparison
        arr1 = jnp.array(data1)
        arr2 = jnp.array(data2)
        
        print(f"  Shape1: {arr1.shape}, Shape2: {arr2.shape}")
        print(f"  Data identical: {jnp.array_equal(arr1, arr2)}")
        
        # Should be:
        for f1, f2 in zip(files1, files2):
            # ... comparison code ...
            if not jnp.array_equal(arr1, arr2):
                print("============== TEST FAILED - DATA INITIALIZATION IS INCONSISTENT =============")
                return 1
        print("============== TEST PASSED - DATA INITIALIZATION IS CONSISTENT =============") 
        return 0

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compare checkpoints and data loader between two directories')
    parser.add_argument('dir1', type=str, help='Path to first directory')
    parser.add_argument('dir2', type=str, help='Path to second directory')

    # Example 
    # python test.py ./runs/artifacts/t_gpu_run/541/axlearn_out/ ./initialization_checks/fp32_fuji_smol/
    # python test.py ./runs/artifacts/t_gpu_run/541/axlearn_out/ ./runs/artifacts/t_gpu_run/540/axlearn_out/
    
    # Parse arguments
    args = parser.parse_args()

    compare_checkpoints(args.dir1, args.dir2)
    compare_data_loader(args.dir1, args.dir2)