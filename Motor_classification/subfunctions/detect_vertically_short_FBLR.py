import numpy as np
    
# Plotting
import plotly.graph_objects as go

# Personal python functions
from subfunctions.findall import *



def detect_vertically_short_FBLR(tot_tr, outSIG):

    # Remove vertically short FB and LR trial.

    # Look at max cabin value, check if it is below a threshold
    FBmax = np.zeros( (len(outSIG)) )
    LRmax = np.zeros( (len(outSIG)) )
    FBLR_max = np.zeros( (len(outSIG)) )
    
    # print('tot_tr : ' + str(tot_tr))

    # Want to only calculate the signal max for LR and FB trials
    for tr in tot_tr:
        # Way 1 : cumulative sum of initial trial movement
        # onethird_NEWend = int(new2_ind_st[tr] + len(new2_ind_st[tr]:new2_ind_end[tr])/3)
        # LRsum[tr] = abs(sum( outSIG[tr][int(new2_ind_st[tr]):onethird_NEWend, 0] ))
        # FBsum[tr] = abs(sum( outSIG[tr][int(new2_ind_st[tr]):onethird_NEWend, 1] ))
        
        # (Better) Way 2 : Max of cabin movement
        LRmax[tr] = np.max(abs( outSIG[tr][:, 0] ))
        FBmax[tr] = np.max(abs( outSIG[tr][:, 1] ))
        
        # I want the max value out of FBmax and LRmax per trial
        FBLR_max[tr] = np.max( [LRmax[tr], FBmax[tr]] )
    
    
    print('FBLR_max : ' + str(FBLR_max))
    
    # Goal: want to know which FB and LR trials have a max less than thresh
    thresh = 10
    newvec, indy = findall(FBLR_max[tot_tr], '<', thresh)
    
    print('newvec : ' + str(newvec))
    print('indy : ' + str(indy))
    
    tr_num_to_cut = tot_tr[indy]    #  get the trials that have a max less than thresh
    
    cut_trial_FBLR_ver_short = np.unique(tr_num_to_cut)

    return cut_trial_FBLR_ver_short