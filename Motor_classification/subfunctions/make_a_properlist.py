import numpy as np

# Personal python functions
from subfunctions.is_empty import *

def make_a_properlist(vec):
    
    out = []
    for i in range(len(vec)):
        out = out + [np.ravel(vec[i])]
        
    if is_empty(out) == False:
        vecout = np.concatenate(out).ravel().tolist()
    else:
        vecout = list(np.ravel(out))
    
    return vecout
