import numpy as np 
from neuronxcc import nki
import neuronxcc.nki.language as nl


T = 2048 # num tokens
E = 128 # num experts 
EP_DEGREE=16


def get_random_expert_mask(num_tokens=T, experts=E):
    '''
    Generate a random expert mask of shape [T, E] 
    '''

    expert_mask = np.random.choice([0,1], size=num_tokens*experts, p=[0.75, 0.25])
    expert_mask = expert_mask.reshape((num_tokens, experts)) 

    return expert_mask.astype(np.int32)

def ep_mask_numpy(mask, EP=EP_DEGREE): 
    '''
    Converts an expert mask to an EP mask by doing logical OR over local experts
    Args:
      mask: binary matrix of shape [T,E] 
      EP: expert parallel degree
    Returns:
      ep_mask: binary matrix of shape [T, EP]  
    '''

    T, E = mask.shape
    assert E % EP == 0 
    e = E // EP
    mask_split = np.hsplit(mask, EP)    # get EP length list of [T, e] arrays
    mask = np.stack(mask_split, axis=1) # get [T, EP, e] 
    ep_mask = np.any(mask, axis=2).astype(np.int32) 

    return ep_mask

@nki.jit
def ep_mask_nki(mask, EP=EP_DEGREE):
    '''
    Converts an expert mask to an EP mask by doing logical OR over local experts
    Args:
      mask: binary matrix of shape [E, T] 
      EP: expert parallel degree
    Returns:
      ep_mask: binary matrix of shape [EP, T]  
    '''

    E, T = mask.shape
    assert E % EP == 0 
    e = E // EP

    ep_mask = nl.ndarray((EP, T), dtype=mask.dtype, buffer=nl.shared_hbm) 

    # calculate mask 
    for i in nl.affine_range(EP): 
        local_mask = nl.load(mask[i*e:(i+1)*e, :]) 
        reduced_local_mask = nki.isa.tensor_partition_reduce(np.max, local_mask) 
        nl.store(ep_mask[i:(i+1), 0:T], reduced_local_mask) 
    
    return ep_mask
                                                         

if __name__ == "__main__":

    expert_mask = get_random_expert_mask()
    np_mask =  ep_mask_numpy(expert_mask) 
    nki_mask =  ep_mask_nki(np.transpose(expert_mask))

    assert np.array_equal(np_mask, np.transpose(nki_mask)), "Masks are not equal"
    print("Masks are equal")