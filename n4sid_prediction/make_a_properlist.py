def make_a_properlist(vec):
    
    import numpy as np
    
    out = []
    for i in range(len(vec)):
    	out = out + [np.ravel(vec[i])]
    vecout = np.concatenate(out).ravel().tolist()
    
    return vecout
