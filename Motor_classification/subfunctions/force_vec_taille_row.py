import numpy as np

def force_vec_taille_row(vec):
    # Does the same as np.reshape, but oddly np.reshape could not make a column vector a row vector
    vec = np.array(vec)
    if len(vec.shape) == 2:
        if vec.shape[0] > vec.shape[1]:
            vec = vec.T
    return vec
