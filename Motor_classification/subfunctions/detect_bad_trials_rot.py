import numpy as np
    
# Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Importing the statistics module
from statistics import mode, mean, median, multimode
import scipy.stats

from collections import Counter

from subfunctions.vertshift_segments_of_data_wrt_prevsegment import *

# Data saving
import pickle


def detect_bad_trials_rot(Len_tr, good_tr, outSIG, axis_out, A, new3_ind_st, new3_ind_end):
	
    
    # ------------------------------
    # Detect and REMOVE bad trials (robot stopped, erroneous small movement trials, too fast trials)
    # ------------------------------
    
    # There are two ways to handle trial removal:
    
    # 1) Detect bad trials and note what trials have a specific defect - never delete until once at the end
    #     - We do 1!  , need to do statistics on how many trials were deleted and why also
    # 2) Detect bad trials and delete immediately - can have problems keeping track of the trial number 
    
    
    # :::::::::::::::::::
    # NOTE : The data was saved for the trial ONLY (No reinitialization) for rotation
    # So there will be jumps to zero from the last data point in each trial.
    # So no need to detect jumps in data due to robot stop - if the robot stopped we would have a trial that is 
    # too short than average and remove it during horizontally short trial removal
    # :::::::::::::::::::
    
    
    
	# :::::::::::::::::::
    # Detect trials that are vertically too short (via cabin height) - tag the trial as a bad trial and remove these trials
    # :::::::::::::::::::
    cut_trial_ver_short = []
    height_cutoff = 0.5    # a value close to zero 
    for tr in good_tr:
        if np.max(np.abs( outSIG[tr][:, axis_out[tr]] )) < height_cutoff:
            cut_trial_ver_short = cut_trial_ver_short + [tr]
            
    # print('cut_trial_ver_short : ' + str(cut_trial_ver_short))
    
    
    # :::::::::::::::::::
	
	
	
	# :::::::::::::::::::
    # Detect trials that are horizontally too short (via time) - tag the trial as a bad trial and remove these trials
    # :::::::::::::::::::
    # Justification: In rotation, the data consists of detection (20secs) and control (15secs) so the maximum length of a trial is 35 seconds.  The data is sampled at 10Hz, so that is 350 data points per trial maximum.  If detection time is almost 0 seconds (fast detection), then the shortest a trial should be is 15 seconds.  We try to be lenient and account for delays and set the data cut off at 12 seconds.
    # :::::::::::::::::::
    # print('rot_errordetect_trialselection : Detect horizontally short trials')
	
    # ONLY true for ROTATION because there is no reinitialization:
    # First refine STOP point with respect to the time vector - The stop point should be at the maximum time height_cutoff
    new3_ind_end_temp = np.zeros((Len_tr))
    time_idx = 2-1
    time_org = []
    for tr in good_tr:
        ttime = A[new3_ind_st[tr]:new3_ind_end[tr], time_idx]  # time in seconds
        
        dp_jump = 1     # detect jumps in the time greater than 1 second
        
        # Time vertically shifted and baseline shifted to zero
        tt2 = vertshift_segments_of_data_wrt_prevsegment(ttime, dp_jump)
        
        time_org =  time_org + [tt2]
        inx = np.argmax( A[new3_ind_st[tr]:new3_ind_end[tr], time_idx] )

        pt_diff = len(A[new3_ind_st[tr]:new3_ind_end[tr], time_idx]) - inx
        new3_ind_end_temp[tr] = new3_ind_end[tr] - pt_diff

    # Reassign new3_ind_end
    new3_ind_end_temp = [int(x) for x in new3_ind_end_temp] 
    new3_ind_end = new3_ind_end_temp

    # Now, check the time length of each trial
    width_cutoff = 16    # OLD:cut data with time greater than 12 seconds
    cut_trial_hor_short = []
    tr_t_diff = np.zeros((Len_tr))
    dp_diff = np.zeros((Len_tr))
    for tr in good_tr:
        tr_t_diff[tr] = A[new3_ind_end[tr], time_idx] - A[new3_ind_st[tr], time_idx]
        dp_diff[tr] = new3_ind_end[tr] - new3_ind_st[tr]

        # ------------------------------
        #fig = go.Figure()
        #config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        #xxORG = list(range(len(range(new3_ind_st[tr], new3_ind_end[tr]))))
        
        #fig.add_trace(go.Scatter(x=xxORG, y=A[new3_ind_st[tr]:new3_ind_end[tr], time_idx], name='time', line = dict(color='black', width=2, dash='dash'), showlegend=True))
        
        #fig.add_trace(go.Scatter(x=xxORG, y=outSIG[tr][:,0], name='cab %s' % (varr['anom'][0]), line = dict(color='red', width=2, dash='dash'), showlegend=True))
        #fig.add_trace(go.Scatter(x=xxORG, y=outSIG[tr][:,1], name='cab %s' % (varr['anom'][1]), line = dict(color='green', width=2, dash='dash'), showlegend=True))
        #fig.add_trace(go.Scatter(x=xxORG, y=outSIG[tr][:,2], name='cab %s' % (varr['anom'][2]), line = dict(color='blue', width=2, dash='dash'), showlegend=True))
        
        #fig.show(config=config)
        # ------------------------------

        if tr_t_diff[tr] < width_cutoff:
            cut_trial_hor_short = cut_trial_hor_short + [tr]
            
    # print('cut_trial_hor_short : ' + str(cut_trial_hor_short))
    
	
    # cut_trial_hor_short
    # :::::::::::::::::::
    
    # ------------------------------
   
    return cut_trial_ver_short, cut_trial_hor_short