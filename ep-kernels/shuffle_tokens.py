import numpy as np 
from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki.isa.constants import oob_mode
import ml_dtypes 
bfloat16 = np.dtype(ml_dtypes.bfloat16) 

T = 512  # num tokens
H = 16384 # hidden dimension
EP_DEGREE=8
SKIP_DMA=True
LNC = 2


def get_random_ep_mask(num_tokens=T, ep=EP_DEGREE):
    '''
    Generate a random EP mask of shape [T,EP] 
    '''

    ep_mask = np.random.choice([0,1], size=num_tokens*ep, p=[0.5, 0.5])
    ep_mask = ep_mask.reshape((num_tokens, ep)) 
    return ep_mask

def get_buffer_mapping(ep_mask, skip_dma=SKIP_DMA): #index calculation

    '''
    Generate a buffer mapping from given EP mask
    Args:
        ep_mask: binary mask of shape [T, EP] 
        skip_dma: If True, use -1 to indicate invalid tokens. Else, use 0th token for invalid positions
    Returns: 
        buffer_mapping: tensor of shape [T*EP] containing indices of tokens. 
    '''

    assert ep_mask.shape[1] == EP_DEGREE
    
    T, EP = ep_mask.shape 
    buffer_mapping = -1*np.ones((T,EP), dtype=np.int32)
    for ep in range(EP):
        indices = np.nonzero(ep_mask[:,ep])[0]
        n_indices = indices.shape[0] 
        buffer_mapping[:n_indices, ep] = indices

    if not skip_dma: 
        zeros = np.zeros_like(buffer_mapping) 
        buffer_mapping = np.maximum(buffer_mapping, zeros)
    
    buffer_mapping = buffer_mapping.reshape((-1), order='F') #shape T*EP, flatten by cols.

    return buffer_mapping

def shuffle_tokens_numpy(tokens, mapping, skip_dma=SKIP_DMA): 
    '''
    Creates the send buffer input to all-to-all collective 
    Args:
        tokens: tensor of shape [T, H]
        mapping: tensor of shape [T*EP,] where mapping[i] contains the index of the token that goes to shuffled_buffer[i] 
        skip_dma: If true, then add a padding row of 0's to tokens to generate correct goldens for comparison
    Returns:
        shuffled_buffer: tensor of shape [EP*T, H]
    '''

    T, H = tokens.shape 
    if skip_dma:
        padding = np.zeros((1, H), dtype=tokens.dtype)
        tokens =  np.concatenate([tokens, padding], axis=0)

    shuffled_buffer = tokens[mapping] # result is EP*T, H

    return shuffled_buffer

@nki.jit
def shuffle_tokens_nki(tokens, mapping, skip_dma=SKIP_DMA):
    '''
    Creates the send buffer input to all-to-all collective 
    Args:
        tokens: tensor of shape [T, H]
        mapping: tensor of shape [T*EP,] where mapping[i] contains the index of the token that goes to shuffled_buffer[i] 
        skip_dma: If True, Avoid loading tokens with index -1 into SBUF. Return 0's for invalid indices instead.
    Returns:
        shuffled_buffer: tensor of shape [EP*T, H]
    '''

    N  = mapping.shape[0]
    T, H = tokens.shape
    assert T % 128 == 0, "Num tokens must be a multiple of 128 (initial impl)" #TODO: Remove restriction
    ntiles = T // 128
    assert N % T == 0, "Buffer mapping must be multiple of T"
    EP = N // T

    shuffled_buffer = nl.ndarray((T*EP, H), dtype=tokens.dtype, buffer=nl.shared_hbm) 

    num_shards = nl.num_programs(axes=0)
    stride_h = H // num_shards
    start_h = nl.program_id(0) * stride_h
    end_h =  start_h + stride_h

    #load tokens into SBUF per EP group
    for i in nl.affine_range(EP):
        # load tiles of T for each EP group
        for j in nl.affine_range(ntiles):
            start = i*T + j*128
            end  =  i*T + (j+1)*128
            local_map = nl.load(mapping[start:end])

            load_p, load_f = nl.mgrid[0:128, 0:stride_h]
            local_tokens = nl.ndarray((128, stride_h), dtype=tokens.dtype, buffer=nl.sbuf)
            if skip_dma:
                local_tokens[load_p, load_f] = nisa.memset((128, stride_h), value=0, dtype=tokens.dtype) # required for checking correctness with NP golden. skip for workload runs ?
            
            local_tokens[load_p, load_f] = nl.load(tokens[local_map, load_f + start_h], dtype=tokens.dtype, mode=oob_mode.skip)
            nl.store(shuffled_buffer[start:end, start_h:end_h], local_tokens) # how to avoid write DMA for skipped tokens

    return shuffled_buffer
    

if __name__ == "__main__":

    print("T/H/EP: ", T, H, EP_DEGREE)
    ep_mask = get_random_ep_mask()
    mapping = get_buffer_mapping(ep_mask, skip_dma=SKIP_DMA) 

    tokens = np.random.rand(T,H).astype(bfloat16)
    golden_buffer = shuffle_tokens_numpy(tokens, mapping, skip_dma=SKIP_DMA)
    nki_buffer = shuffle_tokens_nki[nl.nc(LNC)](tokens, mapping, skip_dma=SKIP_DMA)
    
    assert np.array_equal(golden_buffer.astype(np.float32), nki_buffer.astype(np.float32)), "Buffers are not equal"
    print("Buffers are equal")
