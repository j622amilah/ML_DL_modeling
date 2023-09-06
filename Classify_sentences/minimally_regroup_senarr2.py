# Created by Jamilah Foucher, 22-23/02/2022

import numpy as np

# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')

from make_a_properlist import *
from make_a_properlist_str import *
from findall import *
from is_empty import *

from string_text_processing.unique_str_arrays import *


def is_shorter_in_arr(shorter, arr):
    
    if is_empty(arr) == False:
        # do not consider the last entry of the array because it is shorter 
        arr = arr[0:-1]
        
    # Determine if entries of shorter are in arr
    indy = []
    for q in shorter:
        # print('q: ', q)

        for ae_ind, arr_element in enumerate(arr):
            arr_element = make_a_properlist(arr_element)
            # print('arr_element: ', arr_element)
            for arr_entry in arr_element:
                if arr_entry == q:
                    indy.append(ae_ind)
    # print('indy: ', indy)
    
    # indy are entries in shorter that are in arr
    
    if is_empty(indy) == True: # is empty
        # Means that there are NO entries in shorter that are in arr
        # add shorter to arr
        arr.append(shorter)
    else:
        # if past entries of arr contain shorter entries
        # combine those past entries of arr into shorter - update arr
        
        # indy is the index of arr that holds repeating entries that are in shorter
        shorter_new = [arr[i] for i in indy]
        # print('shorter_new: ', shorter_new)
        shorter = shorter + shorter_new
        shorter = make_a_properlist(shorter)
        # print('shorter: ', shorter)
        shorter = np.unique(shorter)
        # print('shorter: ', shorter)
        shorter = make_a_properlist(shorter)
        # print('shorter: ', shorter)

        # remove the indy entries of arr 
        rr = list(range(len(arr)))
        # print('rr: ', rr)

        not_indy = np.setdiff1d(rr, indy)
        # print('not_indy: ', not_indy)

        arr_new = []
        for i in not_indy:
             arr_new.append(arr[i])
        arr = arr_new
        arr.append(shorter)
        
    arr = make_a_properlist_str(arr)
    
    return shorter, arr



def minimally_regroup_senarr2(grp_red):
    
    grp_str = []
    grp_ind = []

    for i in range(len(grp_red)):
        for j in range(len(grp_red[i])):
            grp_str.append(grp_red[i][j])
            grp_ind.append(i)

    # find unique string value
    ind_new, uq_senarray = unique_str_arrays(grp_str)
    # print('uq_senarray: ', uq_senarray)
    
    # get unique strings per grp
    grps = []
    for i in uq_senarray:
        # print('i: ', i)
        val,indy = findall(grp_str, '==', i)
        # print('indy: ', indy)
        out = [grp_ind[q] for q in indy]
        # print('out: ', out)
        grps.append(out)
    # print('grps: ', grps)
    
    # each number in each set of grps is the sentence group number
    # each set is a unique sentence = uq_senarray
    
    # -------------------------
    
    # could get unique sets first 
    ind, uq_grps = unique_str_arrays(grps)
    # print('uq_grps: ', uq_grps)
    # print('ind: ', ind)

    # then check if smaller sets are in larger sets
    len_of_sets = [len(i) for i in uq_grps]
    # print('len_of_sets: ', len_of_sets)
    
    # -------------------------
    
    # sort len_of_sets from small to big
    sort_index = np.argsort(len_of_sets)
    # print('sort_index: ', sort_index)

    s_len_of_sets = [uq_grps[i] for i in sort_index]
    # print('s_len_of_sets: ', s_len_of_sets)
    # s_len_of_sets is a list of most reduced cross-referenced groups
    
    # -------------------------
    
    arr = []
    for i in range(len(s_len_of_sets)-1):
        # print('i: ', i)
        longer = s_len_of_sets[i+1]
        shorter = s_len_of_sets[i]
        # print('longer: ', longer)
        # print('shorter: ', shorter)
        out = np.setdiff1d(longer, shorter) 
        # print('out: ', out)
        
        # A) Search if shorter is in past entries of arr AND add shorter to arr
        shorter, arr = is_shorter_in_arr(shorter, arr)
        # print('arr after A : ', arr)
        # print('shorter after A : ', shorter)
        
        # B) Search for similar entries in shorter and longer - 
        # if entries of longer are different from shorter, append longer 
            
        # case 0: No entries of shorter are in longer
        if len(out) == len(longer):
            # print('B: case 0: No entries of shorter are in longer')
            # add longer to arr
            arr.append(longer)
        
        # case 1: Some entries in shorter are in longer
        elif len(out) < len(longer):
            # print('B: case 1: Some entries in shorter are in longer')
            # merge longer with shorter
            keep_val = [shorter, longer]
            keep_val = make_a_properlist(keep_val)
            keep_val = np.unique(keep_val)
            keep_val = make_a_properlist(keep_val)
            # print('keep_val: ', keep_val)
            arr.append(keep_val)
            
        # case 2: ALL the entries of shorter are in longer
        #elif is_empty(out) == True:
            # DO NOTHING: shorter and longer are already in the array
        # print('arr after B : ', arr)
        
    # gives uniquely clustered groups, based on unique sentences
    # arr
    
    # -------------------------
    
    grp_new = []
    # so now plug in by group
    for i in range(len(arr)):
        arg = []
        for j in range(len(arr[i])):
            arg.append(grp_red[arr[i][j]])
        grp_new.append(arg)
    # grp_new
    
    # -------------------------

    uq_sen_grp = []
    for i in grp_new:
        # print('i: ', i)
        # make all entries into a proper list
        pl = []
        for num in i:
            v_out = make_a_properlist_str(num)
            pl.append(v_out)
        v_out = make_a_properlist_str(pl)
        indsa, outsa  = unique_str_arrays(v_out)
        uq_sen_grp.append(outsa)
    # uq_sen_grp
    
    # -------------------------
    
    return uq_sen_grp