# Created by Jamilah Foucher, 15/02/2022



import numpy as np

# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')

from findall import *

def get_cossine_similarity(inp_wt, relv_sen_unique, sen):

    # Make a word count vector 
    vals_input, wc_input = np.unique(inp_wt, return_counts=True)
    # print('vals_input: ', vals_input)
    # print('wc_input: ', wc_input)

    # Get counts of each of these words per 'relavant sentence'
    cos_sim_all = []
    for rels in relv_sen_unique:
        vals_sen, wc_sen = np.unique(sen[rels], return_counts=True)
        # print('vals_sen: ', vals_sen)
        # print('wc_sen: ', wc_sen)

        wc_rel = []
        # Returns the unique values in 1st array NOT in 2nd array
        vals_from_long_not_in_short = np.setdiff1d(vals_input, vals_sen)
        out = np.setdiff1d(vals_input, vals_from_long_not_in_short)  # the values in common
        if any(out):
            # print('out: ', out)
            # out are common word tokens in sentence, need to align these with wc_sen
            wc_rel = {}
            for o in out:
                val, ind = findall(vals_sen, '==', o)
                # print('val: ', val)
                # print('ind: ', ind)
                wc_rel[o] = wc_sen[ind[0]]

        # Get word count vector equivalent to input word count vector for sentence
        wc_rel_inpeqiv = np.ones((len(wc_input), 1))*0
        for o in range(len(wc_rel)):
            wc_rel_key = list(wc_rel.keys())[o]  # get index from dictionary
            val, ind = findall(vals_input, '==', wc_rel_key)
            # print('ind: ', ind)
            wc_rel_inpeqiv[ind[0]:ind[0]+1] = wc_rel[wc_rel_key]
        # print('wc_rel_inpeqiv: ', wc_rel_inpeqiv)

        # Compare word count vectors : word count vectors that are the same are similar 
        a = wc_input
        b = wc_rel_inpeqiv
        cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

        cos_sim_all.append(cos_sim)

    # print('cos_sim_all: ', cos_sim_all)
    
    return cos_sim_all