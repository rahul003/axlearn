# MoE Testing Framework Updates

## High-Level Summary

Enhanced the MoE (Mixture of Experts) testing framework in `utils_neuron.py` to support comprehensive Expert Parallelism (EP) testing alongside existing Tensor Parallelism (TP) configurations. The changes enable testing of various parallelism strategies for large-scale MoE models including 150B and Qwen3-235B configurations.

## Key Changes

### 1. Enhanced `build_name` Function
- **Purpose**: Improved test case naming to distinguish between TP and EP configurations
- **Changes**: 
  - Added support for `expert` mesh specifications in naming
  - Enhanced mesh string generation to show `ep{N}` for expert parallelism vs `tp{N}` for tensor parallelism
  - Better identification of test configurations in logs and results

### 2. Updated `build_grid_space_input_hidden` Function
- **Purpose**: Core function for generating test configurations with various parallelism combinations
- **Key Updates**:
  - **TP/EP Combinations**: Supports combinations like (1,16), (1,64), (4,1), (4,16), (16,1), (64,1)
  - **Constraint**: Maximum product of TP×EP ≤ 64 (Trn2 architecture limit)
  - **Mesh Specification Logic**:
    - EP > 1: `{"fsdp": -1, "expert": ep_degree}` with `n_groups = ep_degree`
    - TP only: `{"fsdp": -1, "model": tp_degree}` with `n_groups ∈ [1, 4]`
  - **Batch Size Mapping**: Appropriate batch sizes for different parallelism degrees (64→1, 16→4, 4→16)

### 3. Enhanced `build_grid_space_150B` Function
- **Purpose**: Specific test configurations for 150B model scale
- **New Features**:
  - **EP Test Cases**: Added EP=16 configurations for 16-expert scenarios
  - **Configurations**:
    - EP=16 with TP=1: `mesh_spec={"fsdp": -1, "expert": 16}`
    - EP=16 with TP=4: `mesh_spec={"fsdp": -1, "expert": 16, "model": 4}`
  - **Model Dimensions**: 6144→15360 hidden dimensions
  - **Expert Counts**: Supports 8 and 16 experts with appropriate group configurations

### 4. New `build_grid_space_qwen3_235b` Function
- **Purpose**: Comprehensive testing for Qwen3-235B model with both TP and EP strategies
- **Configurations**:
  - **TP=64**: `mesh_spec={"fsdp": -1, "model": 64}`, `n_groups=1`, `batch=1`
  - **EP=16**: `mesh_spec={"fsdp": -1, "expert": 16}`, `n_groups=16`, `batch=4`
  - **EP=64**: `mesh_spec={"fsdp": -1, "expert": 64}`, `n_groups=64`, `batch=1`
- **Model Specifications**:
  - Input dimension: 4096
  - Hidden dimension: 12288
  - Experts: 128
  - Sequence length: 16384
  - Top-k values: [1, 8]

## Technical Details

### Parallelism Strategy Logic
- **Expert Parallelism (EP)**: `n_groups = ep_degree` ensures proper load balancing across expert-parallel processes
- **Tensor Parallelism (TP)**: `n_groups ∈ [1, 4]` for different input splitting strategies
- **Constraint Enforcement**: `n_experts ≥ ep_degree` to ensure sufficient experts per EP process

### Batch Size Optimization
```python
batch_sizes = {
    4: 16,   # Lower parallelism → higher batch size
    8: 8,
    16: 4,   # EP=16 uses batch=4
    32: 2,
    64: 1,   # EP=64 uses batch=1 (highest parallelism)
}
```

### Test Coverage
- **Switch-Base**: Uses `build_grid_space_input_hidden` with 1536→6144 dimensions
- **150B Models**: Uses `build_grid_space_150B` with 6144→15360 dimensions  
- **Qwen3-235B**: Uses `build_grid_space_qwen3_235b` with 4096→12288 dimensions

## Usage

### Running Tests
```bash
# Switch-base with EP configurations
sbatch aws-testing/test.slurm switch-base

# 150B development tests
sbatch aws-testing/test.slurm 150b

# Qwen3-235B with TP/EP combinations
sbatch aws-testing/test.slurm qwen3-235b
```

### Expected Test Cases
- **Switch-Base**: ~32 test configurations covering EP=16, TP=4, TP=16 with various expert counts
- **150B**: Enhanced with EP=16 test cases for 16-expert scenarios
- **Qwen3-235B**: 6 test configurations (3 parallelism strategies × 2 top-k values)

## Benefits
1. **Comprehensive Coverage**: Tests both TP and EP strategies for large MoE models
2. **Scalability Validation**: Ensures proper scaling up to 64-core parallelism
3. **Performance Comparison**: Enables direct comparison between TP and EP approaches
4. **Hardware Optimization**: Validates efficient utilization of Trn2 architecture constraints