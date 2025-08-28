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

NUM_CORES = 64 # LNC2, EP degree
H = 1024 # hidden
T = 128  # num tokens

@nki.jit
def nki_all_to_all(send_buffer, recv_buffer: nt.mutable_tensor): 
    '''
    All to all collective call
    Args:
        send_buffer: [1, T*EP, H] tensor
        recv_buffer: [1, T*EP, H] tensor
    Returns:
        recv_buffer: [1, T*EP, H] tensor
    '''
    send_buffer_list = [send_buffer] 
    recv_buffer_list = [recv_buffer]
    nccl.all_to_all(np.add, send_buffer_list, recv_buffer_list, [list(range(NUM_CORES))], 1, 1) 

    return recv_buffer_list[0]


@jax.jit
def global_permute(x):
    '''
    Global permutation of the input tensor
    Args:
        x: [1, T*EP, H] tensor
    Returns:
        y: [1, T*EP, H] tensor
    '''
    y = jnp.zeros_like(x)
    y = nki_all_to_all(x, y)
    return y 

def setup_toy():
    with jax.default_device(jax.devices("cpu")[0]):
        a = jnp.arange(NUM_CORES)
        a = jnp.repeat(a, H)
        a = a.reshape(NUM_CORES, -1, H)
        a = jnp.broadcast_to(a, (NUM_CORES, NUM_CORES*T, H))
    return a.astype(jnp.int16)

if __name__ == "__main__":

    mesh = jax.make_mesh((NUM_CORES, ), ('ep'))
    sharding = jax.sharding.NamedSharding(mesh, P('ep')) 

    @partial(jax.jit, out_shardings=sharding)
    def setup():
        with jax.default_device(jax.devices("cpu")[0]):
            a = jax.random.normal(jax.random.PRNGKey(0), shape=(NUM_CORES, NUM_CORES*T, H), dtype=jnp.float32) # Use fp32, some NaNs with bf16
        return a.astype(jnp.bfloat16)

    input_sharded = setup() # [EP, T, H]

    global_permute_sm = shard_map(
        global_permute,
        mesh=mesh,
        in_specs= (P('ep')),  
        out_specs=(P('ep')),  
        check_rep=False
    )
 
    perm_sharded = global_permute_sm(input_sharded)
    unperm_sharded = global_permute_sm(perm_sharded)

    assert jnp.array_equal(unperm_sharded, input_sharded), "unperm and input differ"
    print("Unperm and input match")

 
