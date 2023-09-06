import numpy as np

def is_empty(vec):
    
    #if not any(vec):
    
    # OR
    
    #if len(vec) < 1:  # OR
    
    vec = np.array(vec)
    if vec.shape[0] == 0:
        # print('yes, the array is empty')
        out = True
    else:
        # print('no, the array is not empty')
        out = False
        
    return out
