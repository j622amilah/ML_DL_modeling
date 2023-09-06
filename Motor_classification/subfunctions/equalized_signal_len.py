# Created by Jamilah Foucher, Aout 31, 2021.

# Purpose: Enter the names of vectors that you wish to be the same length, the function returns te same vectors cut to the same length (the length of the shortest vector).

# Input VARIABLES:
# (1) vec0_difflen, vec1_difflen
# 
# Output VARIABLES:
# (1) vec0_samelen, vec1_samelen

# Example:
# vec0_samelen, vec1_samelen = equalized_signal_len(vec0_difflen, vec1_difflen)



import numpy as np

 
def equalized_signal_len(*arg):
    
    totlen = []
    for args in arg:
        # print('args : ' + str(args))
        
        totlen = totlen + [len(args)]
        # totlen.append(len(args))
        
        
    minlen = min(totlen)
    
    i = 0
    for args in arg:
        if i == 0:
            vec0 = np.ravel(np.reshape(args[0:minlen], (1, minlen)))
        elif i == 1:
            vec1 = np.ravel(np.reshape(args[0:minlen], (1, minlen)))
        elif i == 2: 
            vec2 = np.ravel(np.reshape(args[0:minlen], (1, minlen)))
        elif i == 3: 
            vec3 = np.ravel(np.reshape(args[0:minlen], (1, minlen)))
        elif i == 4: 
            vec4 = np.ravel(np.reshape(args[0:minlen], (1, minlen)))
        elif i == 5: 
            vec5 = np.ravel(np.reshape(args[0:minlen], (1, minlen)))
        elif i == 6: 
            vec6 = np.ravel(np.reshape(args[0:minlen], (1, minlen)))
        elif i == 7: 
            vec7 = np.ravel(np.reshape(args[0:minlen], (1, minlen)))
        elif i == 8: 
            vec8 = np.ravel(np.reshape(args[0:minlen], (1, minlen)))
        elif i == 9: 
            vec9 = np.ravel(np.reshape(args[0:minlen], (1, minlen)))
        i = i + 1
    
    if i == 1:
        return vec0
    elif i == 2:
        return vec0, vec1
    elif i == 3:
        return vec0, vec1, vec2
    elif i == 4:
        return vec0, vec1, vec2, vec3
    elif i == 5:
        return vec0, vec1, vec2, vec3, vec4
    elif i == 6:
        return vec0, vec1, vec2, vec3, vec4, vec5
    elif i == 7:
        return vec0, vec1, vec2, vec3, vec4, vec5, vec6
    elif i == 8:
        return vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7
    elif i == 9:
        return vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8
    elif i == 10:

        return vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9