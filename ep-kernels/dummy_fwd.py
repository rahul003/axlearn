import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax._src.mesh import thread_resources
from functools import partial

import numpy as np
from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.nccl as nccl
import neuronxcc.nki.typing as nt

from shuffle_tokens import get_random_ep_mask, get_buffer_mapping, shuffle_tokens_nki
from reduce_across_ep import get_reduction_indices, reduce_across_ep_nki
from collective_kernel import nki_all_to_all

NUM_CORES = 64 # LNC2
EP_DEGREE = 64 # LNC2
H = 1024 # hidden 
T = 128 # num tokens 
LNC = 2


@jax.jit 
def moe_with_ep(tokens, mapping, positions): 

    #shuffle 
    tokens = jnp.squeeze(tokens, axis=0)    # sharded on EP [1,T,H] -> [T,H]
    send_buffer = shuffle_tokens_nki[(nl.nc(LNC),)](tokens, mapping, skip_dma=True)  # [T*EP, H] 
    send_buffer =jnp.expand_dims(send_buffer, axis=0) 

    #dispatch a2a 
    recv_buffer = jnp.zeros_like(send_buffer) 
    recv_buffer = nki_all_to_all(send_buffer, recv_buffer) 

    # placeholder: moe compute
    # recv_buffer = blockwise_moe(recv_buffer) 

    #combine a2a
    unperm_result = jnp.zeros_like(recv_buffer) 
    unperm_result = nki_all_to_all(recv_buffer, unperm_result) 

    #reduce across EP 
    unperm_result = jnp.squeeze(unperm_result, axis=0) 
    reduced_result = reduce_across_ep_nki[(nl.nc(LNC),)](unperm_result, positions) 
    reduced_result = jnp.expand_dims(reduced_result, axis=0)

    return reduced_result

@jax.jit
def forward_pass(ep_mask, mapping, positions):
    
    mesh = jax.make_mesh((NUM_CORES, ), ('ep'))
    sharding = jax.sharding.NamedSharding(mesh, P('ep')) 

    @partial(jax.jit, out_shardings=sharding)
    def setup():
        with jax.default_device(jax.devices("cpu")[0]):
            a = jax.random.normal(jax.random.PRNGKey(0), shape=(NUM_CORES, T, H), dtype=jnp.float32) # Use fp32, some NaNs with bf16
        return a.astype(jnp.bfloat16)
    
    tokens = setup() # [EP, T, H] 
    moe_with_ep_sm = shard_map(
        moe_with_ep,
        mesh=mesh,
        in_specs= (P('ep'), None, None),  #tokens/mapping/positions
        out_specs=(P('ep')),  
        check_rep=False
    )
    
    result = moe_with_ep_sm(tokens, mapping, positions) # [EP,T, H]

    factor = jnp.sum(ep_mask, axis=1, keepdims=True) 
    golden = tokens*factor

    return result, golden


if __name__ == "__main__":

    # for simplicity assume same ep_mask/mapping/posns for all EP ranks
    ep_mask = get_random_ep_mask(T, EP_DEGREE) 
    mapping = get_buffer_mapping(ep_mask, skip_dma=True) 
    positions = get_reduction_indices(ep_mask) 
    result, golden = forward_pass(jnp.array(ep_mask), jnp.array(mapping), jnp.array(positions))

    assert jnp.allclose(golden.astype(jnp.float32), result.astype(jnp.float32)), "Result incorrect"
    print("Result correct")
