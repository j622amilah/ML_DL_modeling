import numpy as np


def detectFB_LR_trials(outSIG, remaining_tr, new2_ind_st, new2_ind_end, axis_out):

    # Determine if trial is FB OR LR
    # FB and LR trials start from zero (the initial position), so the cabin axis that moves first is 
    # the correct stimuli.  Joystick has a delay so the first cabin movement is without joystick control.
     
    num_of_tr = len(remaining_tr)
    FB_trials = []
    LR_trials = []

    for tr in remaining_tr:
        
        # Calculate the sum of the two signals at the beginning and determine which is larger.
        tr_ind = np.multiply(range(int(new2_ind_st[tr]), int(new2_ind_end[tr])), 1)
        portion = new2_ind_st[tr] + len(tr_ind)/4

        starter = 0
        ender = int(np.round(portion) - new2_ind_st[tr])

        LRsum = abs(sum( outSIG[tr][starter:ender, 0] ))
        FBsum = abs(sum( outSIG[tr][starter:ender, 1] ))

        # [LRsum, FBsum, portion, starter, ender]

        if FBsum > LRsum:
            axis_out[tr] = 2  # FB
            FB_trials = FB_trials + [tr]
        else:
            axis_out[tr] = 1  # LR
            LR_trials = LR_trials + [tr]
            
    return FB_trials, LR_trials, axis_out
