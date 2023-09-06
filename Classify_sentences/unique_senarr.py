# Created by Jamilah Foucher, 21/02/2022

import numpy as np

# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')

from findall import *
from make_a_properlist_str import *
from string_text_processing.unique_str_arrays import *
from string_text_processing.is_sen_in_senarray import *



def unique_senarr(grp_red):

	# COULD be done BETTER!  Use the string subfunctions
    # Look for repeating sentences across groups and combine

    # Find repeating groups
    remb_rep_grp = []
    remb_rep_sen = []
    remb_rep_temp = []
    for i in range(len(grp_red)):
        ss = np.setdiff1d(range(len(grp_red)), i)
        
        for j in range(len(grp_red[i])):
            for k in ss:
                # print('grp_red[k] : ', grp_red[k])
                # print('grp_red[i][j] : ', grp_red[i][j])
                val, ind = findall(grp_red[k], '==', grp_red[i][j])
                # print('ind: ', ind)
                
                if not any(ind) == False:
                    # Store the group number that has repeating sentences
                    remb_rep_grp.append(k)
                    
                    # Detect if a sentence repeats
                    # You know that grp_red[i][j] is repeating, but you don't know if it repeats
                    remb_rep_temp.append(grp_red[i][j])
                    
                    # Determine if it is a repeating sentence
                    indy = []
                    for q in remb_rep_sen:
                        # print('q: ', q)
                        val, indy = findall(remb_rep_temp, '==', q)
                    indy = np.ravel(indy)
                    # print('indy: ', indy)
                    
                    if not any(indy) == True:  # the array is empty
                        # The sentence is not in the stored values of array - save it
                        remb_rep_sen.append(grp_red[i][j])
                    # print('remb_rep_sen: ', remb_rep_sen) 
                    
    print('remb_rep_grp: ', remb_rep_grp)
    print('remb_rep_sen: ', remb_rep_sen)
    
    # -------------------------
    
    # Add repeating sentences to grp_new
    grp_new = []

    for sen in remb_rep_sen:
        arr = []
        for i in remb_rep_grp:
            senarray = grp_red[i]
            
            indy, res = is_sen_in_senarray(sen, senarray)
            if res == True:
                arr.append(senarray)
        
        out = make_a_properlist_str(arr)
        
        # Remove repeating sentence arrays
        ind_new, uq_senarray = unique_str_arrays(out)
        
        grp_new.append(uq_senarray)

    # -------------------------
        
    # Add non-repeating sentences to grp_new
    remb_rep_grp
    ss = np.setdiff1d(range(len(grp_red)), remb_rep_grp)

    for i in ss:
        grp_new.append(grp_red[i])
    
    
    return grp_new