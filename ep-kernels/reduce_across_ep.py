import numpy as np 
from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki.isa.constants import oob_mode
import ml_dtypes 
bfloat16 = np.dtype(ml_dtypes.bfloat16) 

from shuffle_tokens import get_random_ep_mask, get_buffer_mapping, shuffle_tokens_numpy

T = 512  # num tokens
H = 16384  # hidden dimension
EP_DEGREE=8

def get_reduction_indices(ep_mask):
    '''
    Calculates positions of tokens in the shuffle buffer for a given EP mask 
    Args:
        ep_mask: Binary mask of shape [T, EP] 
    Returns: 
        positions: indices of shape [T, EP] where ith row represents the locations of 
            the ith token in the shuffle_buffer of length T*EP. -1 indicates token was not 
            assigned to the EP bucket.
    '''

    T, EP = ep_mask.shape 

    # calculate positions of tokens in shuffled buffer
    mask_cumsum = np.cumsum(ep_mask, axis=0) 
    offsets = T*np.arange(EP_DEGREE).reshape(1,-1)
    positions = mask_cumsum + offsets     # 1-indexed 
    positions = positions*ep_mask         # 0 at (i,j) indicates ith token did not go into jth EP bucket 
    positions = positions - 1             # 0-indexed, -1 at (i,j) indicates ith token did not go into jth EP bucket

    return positions.astype(np.int32)

def reduce_acros_ep_numpy(shuffled_tokens, positions):
    '''
    Gathers tokens from EP buckets and sums them up according to given indices 
    Args: 
        shuffled_tokens: shape [T*EP, H] 
        positions: indices of shape [T, H] where ith row gives the positions of the ith token 
                   in the shuffled_tokens buffer. -1 indicates invalid position. 
    Returns: 
        reduced_tokens: shape [T,H] containing the summed up result for each token
    '''

    # pad shuffled tokens with a row of 0's. index -1 in positions will pick it up 
    padding = np.zeros((1, H), dtype=tokens.dtype)
    shuffled_tokens =  np.concatenate([shuffled_tokens, padding], axis=0)

    # sum up tokens across EP buckets in shuffled buffer 
    take = np.take(shuffled_tokens, positions, axis=0) # shape [T, EP, H]
    reduced_tokens = np.sum(take, axis=1)   # shape [T, H]

    return reduced_tokens

@nki.jit
def reduce_across_ep_nki(shuffled_tokens, positions):
    '''
    Gathers tokens from EP buckets and sums them up according to given indices 
    Args: 
        shuffled_tokens: shape [T*EP, H] 
        positions: indices of shape [T, H] where ith row gives the positions of the ith token 
                   in the shuffled_tokens buffer. -1 indicates invalid position. 
    Returns: 
        reduced_tokens: shape [T,H] containing the summed up result for each token
    '''

    T, EP = positions.shape 
    N, H = shuffled_tokens.shape 
    assert N // T == EP 

    reduced_tokens = nl.ndarray((T, H), dtype=shuffled_tokens.dtype, buffer=nl.shared_hbm) 

    # process one token at a time 
    for i in nl.affine_range(T): 

        indices = nl.load(positions[i]) 
        load_p, load_f = nl.mgrid[0:EP, 0:H]   

        # create and initialize buffer to load token from EP buckets
        local_tokens = nl.ndarray((EP, H), dtype=shuffled_tokens.dtype, buffer=nl.sbuf) 
        local_tokens[load_p, load_f] = nisa.memset((EP,H), value=0, dtype=shuffled_tokens.dtype) 
        
        # load with skip_dma
        local_tokens[load_p, load_f] = nl.load(shuffled_tokens[indices, load_f], dtype=shuffled_tokens.dtype, mode=oob_mode.skip)
        
        # reduce and store
        reduced_local_tokens = nki.isa.tensor_partition_reduce(np.add, local_tokens) 
        nl.store(reduced_tokens[i:(i+1), 0:H], reduced_local_tokens) 

    return reduced_tokens


if __name__ == "__main__":

    ep_mask = get_random_ep_mask(T, EP_DEGREE)  
    tokens = np.random.rand(T,H).astype(bfloat16)

    # shuffle tokens into EP buffer according to mask 
    mapping = get_buffer_mapping(ep_mask, skip_dma=True)
    shuffled_tokens = shuffle_tokens_numpy(tokens, mapping, skip_dma=True) 

    # EP reduction
    positions = get_reduction_indices(ep_mask)
    result_np = reduce_acros_ep_numpy(shuffled_tokens, positions)    
    result_nki = reduce_across_ep_nki(shuffled_tokens, positions)

    assert np.allclose(result_np.astype(np.float32), result_nki.astype(np.float32)), "Reduction results are not equal"
    print("Reduction results are equal")