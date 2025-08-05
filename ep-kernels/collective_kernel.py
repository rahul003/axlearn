import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax._src.mesh import thread_resources

import numpy as np
from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.nccl as nccl
import neuronxcc.nki.typing as nt

NUM_CORES = 128 # LNC1

@nki.jit
def nki_all_to_all(send_buffer, recv_buffer: nt.mutable_tensor): 
    '''
    All to all collective call
    Args:
        send_buffer: [1, N] tensor 
        recv_buffer: [1, N] tensor
    Returns:
        recv_buffer: [1, N] tensor
    '''
    send_buffer_list = [send_buffer] 
    recv_buffer_list = [recv_buffer]
    nccl.all_to_all(np.add, send_buffer_list, recv_buffer_list, list(range(NUM_CORES)), 1, 1) 

    return recv_buffer_list[0]


@jax.jit
def global_permute(x):
    '''
    Global permutation of the input tensor
    Args:
        x: [1, N] tensor
    Returns:
        y: [1, N] tensor
    '''
    y = jnp.zeros_like(x)
    y = nki_all_to_all(x, y)
    return y 

def setup():
    with jax.default_device(jax.devices("cpu")[0]):
        a = jnp.arange(NUM_CORES).reshape(-1, 1)
        a = jnp.broadcast_to(a, (NUM_CORES, NUM_CORES))
    return a

if __name__ == "__main__":

    input = setup()

    mesh = jax.make_mesh((NUM_CORES, ), ('ep'))
    sharding = jax.sharding.NamedSharding(mesh, P('ep')) 
    print(mesh) 

    input_sharded = jax.device_put(input, sharding)

    global_permute_sm = shard_map(
        global_permute,
        mesh=mesh,
        in_specs= (P('ep')),  
        out_specs=(P('ep')),  
        check_rep=False
    )
 
    output_sharded = global_permute_sm(input_sharded)
    print(output_sharded)
    jax.debug.visualize_array_sharding(output_sharded)

 
