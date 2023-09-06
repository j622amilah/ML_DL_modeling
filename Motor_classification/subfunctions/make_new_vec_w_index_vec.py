import numpy as np

from findall import findall


def make_new_vec_w_index_vec(old_vec, index_vec, const, way):

    # Created by Jamilah Foucher, Février 01, 2021

    # Purpose: Make a new vector, where entries from an old vector are put for certain indices (given by an index vector).  Put a constant number in the new vector indices, that are not in the index vector.
    #  
    # Input VARIABLES:
    # (1) old_vec is a vector the same size as new_vec, where you only want certain index values to be put in new_vec
    #  
    # (2) index_vec is a vector with index values, to reassign these index values from old_vec to new_vec
    # 
    # (3) const is a constant number
    #  
    # (4) way = 1 or 2  where 1 is a slower way (for loop and if statement), 2 is faster (find function)
    # 
    # Output VARIABLES:
    # (1) new_vec

    new_vec = []

    if way == 1:
        n = 1
        for trN in range(len(old_vec)):
            if (index_vec[n:n+1] == trN) and (n <= len(index_vec)):
                new_vec = new_vec + [old_vec[trN]]
                n = n + 1
            else:
                new_vec = new_vec + [const]   # make bad trials a strange number
        
    elif way == 2:
        a = range(len(old_vec))
            
        # Need to make a concatenated list of all indexes
        idx = []
        for val_in_indexvec in index_vec:
            ni1, idx_inc = findall(a, '==', val_in_indexvec)
            idx = idx + [idx_inc]
        
        new_vec = const*np.ones((len(old_vec)))
        new_vec[idx] = old_vec[idx]

    return new_vec
