# Created by Jamilah Foucher, 15/02/2022

import numpy as np

# org_wt is the old wordtoken with char2remove
# char2remove is a string (ie: '.')
# i is the new wordtoken without char2remove

def remove_chars_from_wordtoken(org_wt, char2remove, str_arr_space):
    grp = org_wt.partition(char2remove)
    if char2remove in grp:
        if (str_arr_space == ' divided by ') and (grp[0] == 24) and (grp[2] == 7):
            str_arr_space = ' hours by days of '
        
        # Characters in grp NOT including char2remove
        remaining_grp = np.setdiff1d(grp, char2remove)
        #print('remaining_grp: ', remaining_grp)
        
        i = str_arr_space.join(remaining_grp) #it concatenates all string arrays and puts '' between each string array
    else:
        i = org_wt
    return i