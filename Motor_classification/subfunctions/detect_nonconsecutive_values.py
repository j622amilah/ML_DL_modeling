# Created by Jamilah Foucher, October 16, 2020

# Purpose: Search for non-consecutive values of a vec.

# Input VARIABLES:
# (1) vec is the index of a signal 

# Output VARIABLES:
# (1) non_consec_vec is a vector containing the non-consecutive values of vec
# 
# (2) non_consec_ind is the corresponding index of vec where the non-consecutive values were found

import numpy as np

def detect_nonconsecutive_values(vec):
    
    non_consec_vec = [vec[0]]
    non_consec_ind = [0]
    
    for i in range(len(vec)-1):
        if vec[i] != vec[i+1]:
            # not consecutive data point
            non_consec_vec = non_consec_vec + [vec[i+1]]
            non_consec_ind = non_consec_ind + [i+1]
    
    non_consec_vec = np.ravel(non_consec_vec)
    non_consec_ind = np.ravel(non_consec_ind)
    
    return non_consec_vec, non_consec_ind
