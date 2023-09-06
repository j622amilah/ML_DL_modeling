# Created by Jamilah Foucher, 15/02/2022

import numpy as np

# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')

from isnan import *
from make_a_properlist import *

from string_text_processing.calc_cossim_mat_of_2sen_arr import *
from make_a_properlist_str import *
from string_text_processing.unique_str_arrays import *
from string_text_processing.is_sen_in_senarray import *


# Similarity threshold [0,1] : make sim_thresh bigger if you want more precise groups ()



def get_unique_sen_grpsWITHmax(sen1_ar_temp, sen2_ar_temp, sim_thresh, plotORnot):

    # -------------------------
    
    grp = sen1_ar_temp, sen2_ar_temp
    grp = make_a_properlist(grp)
    # print('Total number of sentence groups: ', len(grp))

    # put extra square brackets around each sentence
    temp = []
    for i in grp:
        temp.append([i])
    grp = temp

    grp_red = grp # initialization
    
    # -------------------------
    
    # Run in loop
    cossim_sen_mat = calc_cossim_mat_of_2sen_arr(sen1_ar_temp, sen2_ar_temp, plotORnot)

    row, col = cossim_sen_mat.shape

    # Remove all nan
    for i in range(row):
        for j in range(col):
            if isnan(cossim_sen_mat[i,j]) == True:
                cossim_sen_mat[i,j] = 0

    # -------------------------

    # Want the max column numbers per s1 ONLY if it is greater than sim_thresh
    a = np.argmax(cossim_sen_mat, axis=1)  # max across columns
    a_mod = {}
    s1index = []
    for s1num, s2num in enumerate(a):
        if cossim_sen_mat[s1num, s2num] > sim_thresh:
            a_mod[s1num] = s2num
            s1index.append(s1num)
    # print('a_mod: ', a_mod)

    # -------------------------
    if not any(a_mod) == False:  # is not empty
        
        # Remove repeating s2 values
        s1_loc = list(a_mod.keys())
        s2_loc = list(a_mod.values())
        # print('s1_loc : ', s1_loc)
        # print('s2_loc : ', s2_loc)
        
        # Count how many s1 matches found
        matches = len(s1_loc)
        # print('matches: ', matches)
        
        # -------------------------
        
        # Reorganize grp with newly grouped word sentences
        grp_red = []
        for i in range(matches):
            # print('i: ', i)
            
            # these two sentences are similar so, use the index to get the sentences
            ind_s1 = s1_loc[i]
            ind_s2 = s2_loc[i]
            
            # Search through grp and put these entries in the new grp
            ind_newvec1 = []
            for i in range(len(grp)):
                for j in range(len(grp[i])):
                    if grp[i][j] == sen1_ar_temp[ind_s1]:
                        ind_newvec1.append(i)
            # print('ind_newvec1: ', ind_newvec1)
            
            ind_newvec2 = []
            for i in range(len(grp)):
                for j in range(len(grp[i])):
                    if grp[i][j] == list(np.ravel(sen2_ar_temp[ind_s2])):
                        ind_newvec2.append(i)
            # print('ind_newvec2: ', ind_newvec2)
            
            ind_tot = ind_newvec1, ind_newvec2
            ind_tot = make_a_properlist(np.unique(ind_tot))
            ind_tot = [int(q) for q in ind_tot]
            # print('ind_tot: ', ind_tot)
            
            if not any(ind_tot) == False:  # Means the vector is NOT empty
                arr = []
                for q in ind_tot:
                    arr.append(grp[q])
                # print('arr: ', arr)

            grp_red.append(arr)

        # print('length of grp_red: ', len(grp_red))
        
        # -------------------------
        
        # Tally up the remaining questions that did not have a match
        # sentence 1
        remaining_row = np.setdiff1d(range(row), s1_loc)
        # print('remaining_row: ', remaining_row)
        
        # sentence 2
        remaining_col = np.setdiff1d(range(col), s2_loc)
        # print('remaining_col: ', remaining_col)
        
        if not any(remaining_row) == False:  # is not empty
            # get remaining sentences
            for i in remaining_row:
                # # print('sen1_ar_temp[i]: ', sen1_ar_temp[i])
                grp_red.append([sen1_ar_temp[i]])
        
        if not any(remaining_col) == False:  # is not empty
            # get remaining sentences
            for i in remaining_col:
                # print('sen2_ar_temp[i]: ', sen2_ar_temp[i])
                grp_red.append([sen2_ar_temp[i]])
                
        # -------------------------
        
        # Reduce the grp_red to lists per group
        # print('grp_red BEFORE : ', grp_red)
        grp_red_adj = []
        for i in range(len(grp_red)):
            grp_red_adj.append(make_a_properlist_str(grp_red[i]))
        grp_red = grp_red_adj
        # print('grp_red AFTER: ', grp_red)
    else:
        # the sentences can not be reduced any more
        grp_red = grp

    # -------------------------
    
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
                    
    # print('remb_rep_grp: ', remb_rep_grp)
    # print('remb_rep_sen: ', remb_rep_sen)
    
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
        
    # grp_new
    
    # -------------------------
        
    return grp_new