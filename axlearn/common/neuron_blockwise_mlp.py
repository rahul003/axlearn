from functools import partial

import jax
import jax.numpy as jnp
# TODO(apoorvtintin): remove pytype disable when dependencies are public.
# pytype: disable=import-error
# Import needed to enable JAX cache on Neuron.
import jax_neuronx  # pylint: disable=unused-import
import neuronxcc.nki.language as nl
from jax import custom_vjp
from jax._src.mesh import thread_resources

from neuronxcc.nki._private_kernels.blockwise_mm import blockwise_mm_selective_cp as blockwise_mm_nki
from neuronxcc.nki._private_kernels.blockwise_mm_bwd import blockwise_mm_bwd_selective_cp as blockwise_mm_bwd_nki

from neuronx_distributed.kernels.find_nonzero_indices import find_nonzero_indices
from neuronx_distributed.kernels.indexed_flatten import indexed_flatten

from neuronxcc.nki.compiler.backends.neuron.dimensions import VNC
import neuronxcc.nki as nki
from dataclasses import dataclass

from jax.ad_checkpoint import checkpoint_name

_blockwise_mm_nki_call = nki.jit(show_compiler_tb=True)(blockwise_mm_nki)
_blockwise_mm_bwd_nki_call = nki.jit(show_compiler_tb=True)(blockwise_mm_bwd_nki)

@dataclass(frozen=True)
class SkipMode:
  skip_token: bool
  skip_weight: bool

Tensor = jax.Array
lnc = 2 if jax.devices()[0].device_kind == "NC_v3d" else 1

def _backend():
    # For compatibility with AOT compilation, we obtain the backend type from physical_mesh.
    global_mesh = thread_resources.env.physical_mesh
    if len(global_mesh.devices):
        backend = global_mesh.devices.flat[0].platform
    else:
        # Fall back to jax.default_backend() if no device is found in physical_mesh.
        backend = jax.default_backend()
    return backend

def can_use_blockwise_matmul_nki(
    hidden_size,
    intermediate_size_tp,
    block_size,
    glu_mlp,
):
    if _backend() != "neuron":
        return False

    if not glu_mlp:
        print("Blockwise NKI kernel incompatible with glu_mlp=False")
        return False

    if blockwise_mm_nki is None:
        print("Failed to load Blockwise NKI kernel.")
        return False
    
    return True
    
    # try:
    #     check_blockwise_mm_kernel_compatibility(
    #         hidden_size=hidden_size,
    #         block_size=block_size,
    #         intermediate_size_tp=intermediate_size_tp,
    #     )
    # except AssertionError as e:
    #     print(f"Blockwise kernel not compatible with model config. Reason: {str(e)}")
    #     return False
    # return True


@partial(custom_vjp, nondiff_argnums=(6,))
def blockwise_mm(
    hidden_states: Tensor,
    expert_affinities_masked: Tensor,
    gate_up_weight: Tensor,
    down_proj_weight: Tensor,
    token_position_to_id: Tensor,
    block_to_expert: Tensor,
    block_size: int,
):
    out, _ = _blockwise_mm_fwd(hidden_states, expert_affinities_masked, gate_up_weight,
                            down_proj_weight, token_position_to_id, block_to_expert, block_size)
    return out

def _blockwise_mm_fwd(
    hidden_states: Tensor,
    expert_affinities_masked: Tensor,
    gate_up_weight: Tensor,
    down_proj_weight: Tensor,
    token_position_to_id: Tensor,
    block_to_expert: Tensor,
    block_size: int, 
):
    orig_expert_affin_shape = expert_affinities_masked.shape
    # Remove O, G dimensions
    with jax.named_scope("take_out_OG"):
        hidden_states = jnp.squeeze(hidden_states, axis=(0,1,))
        expert_affinities_masked = jnp.squeeze(expert_affinities_masked, axis=(0,1,))
        token_position_to_id = jnp.squeeze(token_position_to_id, axis=(0,1,))
        block_to_expert = jnp.squeeze(block_to_expert, axis=(0,1,))
        
    use_index_calc_kernel = True
    if use_index_calc_kernel:
        T,E = expert_affinities_masked.shape # T, local_experts
        global_rank = jax.process_index()
        E_local = E
        max_chunk_size = 16384
        num_blocks = E_local # one block per expert
        # TODO : if E_local % tp_size != 0
        indices, nonzero_counts = find_nonzero_indices[VNC(2)](
            input_tensor=expert_affinities_masked.astype(jnp.float32),
            row_start_id=jnp.array([global_rank * 2], dtype=jnp.int32), # row_start_id to the start of the expert on this EP rank.
            n_rows = E_local, # to the number of experts on this EP rank.
            chunk_size=min(T, max_chunk_size), 
            index_dtype = jnp.int32,
        )
        
        # TODO : gather non_zero counts : [E_kernel,] --> [E/EP,]        
        
        # Get number of blocks and cumulative number of blocks per expert.
        blocks_per_expert = jnp.ones(E_local, dtype = jnp.int32) # (E_EP,) 
        blocks_per_expert_expanded = jnp.expand_dims(blocks_per_expert, axis=1)  # (E_EP, 1)
        
        # TODO : Calculate padding blocks needed and add to last expert
        
        cum_blocks_per_expert = jnp.cumsum(blocks_per_expert_expanded)
        cum_blocks_per_expert = cum_blocks_per_expert.at[1:].set(cum_blocks_per_expert[:-1])
        cum_blocks_per_expert = cum_blocks_per_expert.at[0].set(0)
    
        f_len = min(128, T // 16)
        row_offsets = cum_blocks_per_expert * (block_size // 128)
        # TODO : if E_local % tp_size != 0
        # [E_local, T] --> [num_blocks * block_size,]
        token_position_to_id_padded = indexed_flatten[VNC(2)](
                input_tensor = indices,
                f_len = f_len,
                output_len=num_blocks*block_size + T,
                row_offsets= row_offsets.reshape(-1).astype(jnp.int32),
                row_offsets_start = 0,
            )
        # TODO : # Aggregate information across TP ranks when TP>1
        
        token_position_to_id = token_position_to_id_padded[:num_blocks * block_size]
        
        # Get the block to expert mapping.
        block_ids = jnp.arange(num_blocks, dtype = jnp.int32)
        block_to_expert = jnp.sum(block_ids >= cum_blocks_per_expert[1:].reshape(-1, 1), axis=0).astype(jnp.int32)
            
    # add +1 for padding
    with jax.named_scope("add padding"):
        padding_h = jnp.zeros((1, hidden_states.shape[1]), dtype=hidden_states.dtype)
        padding_e = jnp.zeros((1,expert_affinities_masked.shape[1]), dtype=expert_affinities_masked.dtype)
        # (S+1, H)
        hidden_states = jnp.concat([hidden_states, padding_h], axis=0)
        expert_affinities_masked = jnp.concat([expert_affinities_masked, padding_e], axis=0)
        expert_affinities_masked = jnp.reshape(expert_affinities_masked, (-1, 1))
    out, gate_up_activations_T, down_activations = _blockwise_mm_nki_call[VNC(2)](
        hidden_states,
        expert_affinities_masked,
        gate_up_weight,
        down_proj_weight,
        token_position_to_id,
        block_to_expert,
        block_size=block_size,
        skip_dma=SkipMode(False, False)
    )

    down_activations = checkpoint_name(down_activations, "blockwise.down_activations")
    gate_up_activations_T = checkpoint_name(gate_up_activations_T, "blockwise.gate_up_activations_T")
    
    return out[None, None, None, :-1, :], (hidden_states, expert_affinities_masked, orig_expert_affin_shape, gate_up_weight, 
                down_proj_weight, down_activations, gate_up_activations_T, 
                token_position_to_id, block_to_expert)

def _blockwise_mm_bwd(
    block_size,
    res,
    grad_output
):
    (hidden_states, expert_affinities_masked, orig_expert_affin_shape, gate_up_proj_weight, 
     down_proj_weight, down_activations, gate_up_activations_T, 
     token_position_to_id, block_to_expert) = res
    T,H = hidden_states.shape
    E, _, _, _ = gate_up_proj_weight.shape

    with jax.named_scope("blockwise_backward"):
        grad_output =  jnp.squeeze(grad_output, axis=(0,1,2))
        padding_h = jnp.zeros((1, hidden_states.shape[1]), dtype=hidden_states.dtype)
        grad_output = jnp.concat([grad_output, padding_h], axis=0)
        # Compute gradients
        hidden_states_grad, affinities_grad, gate_up_proj_weight_grad, down_weight_grad = _blockwise_mm_bwd_nki_call[VNC(2)](
            hidden_states,
            expert_affinities_masked,
            gate_up_proj_weight,
            gate_up_activations_T,
            down_proj_weight,
            down_activations,
            token_position_to_id.astype(jnp.int32),
            block_to_expert.astype(jnp.int32),
            grad_output,
            block_size=block_size,
            skip_dma=SkipMode(False, False),
            ktype=0 if block_to_expert.shape[-1] == down_proj_weight.shape[0] else 1,
        )
        sliced_tensor = hidden_states_grad[:-1,:]
        hidden_states_grad = sliced_tensor.reshape(1, 1, -1, H)
        
        affinities_grad = jnp.reshape(affinities_grad, (-1, orig_expert_affin_shape[-1]))
        affinities_grad = affinities_grad[:-1, :].reshape(1, 1, -1, orig_expert_affin_shape[-1])
    return (
        hidden_states_grad,
        affinities_grad,
        gate_up_proj_weight_grad,
        down_weight_grad,
        token_position_to_id,
        block_to_expert
    )

blockwise_mm.defvjp(_blockwise_mm_fwd, _blockwise_mm_bwd)
