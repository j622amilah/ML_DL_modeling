import numpy as np
    
# Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Importing the statistics module
from statistics import mode, mean, median, multimode
import scipy.stats

from collections import Counter

from subfunctions.findall import *
from subfunctions.full_sig_2_cell import *
from subfunctions.detect_vertically_short_FBLR import *
from subfunctions.vertshift_segments_of_data_wrt_prevsegment import *
from subfunctions.detect_jumps_in_data import *
from subfunctions.make_a_properlist import *

# Data saving
import pickle


def detect_bad_trials_trans(axis_out, Len_tr, good_tr, outJOY, outSIG, outSIGCOM, A, a, b, c, new3_ind_st, new3_ind_end, varr):
	
    
    # ------------------------------
    # Detect and REMOVE bad trials (robot stopped, erroneous small movement trials, too fast trials)
    # ------------------------------
    
    # There are two ways to handle trial removal:
    
    # 1) Detect bad trials and note what trials have a specific defect - never delete until once at the end
    #     - We do 1!  , need to do statistics on how many trials were deleted and why also
    # 2) Detect bad trials and delete immediately - can have problems keeping track of the trial number 
    
    
    
    # -----------------------------
    # Detection of jumps in data
    # -----------------------------
    # NOTE : The entire robotic trajectory from start to finish was saved for translation,
    # therefore there should be no discontiuities in the data unless the robot stopped 
    # (when the robot stopped the the data text file was saved, when the experiment restarted a 
    # new text file was created ---> all the text files were simply concatenated, long periods of zero were
    # removed, and downsampled to 10Hz)

    # So it is just to remove these sections of data.  Technically, sometimes (very rare) we remove the first concatenated 
    # trial with the robot stop but other times the data curve after the cut is weird.  So, no worth saving the 
    # rest of the data in the trial.  Robot cut trials are also rare too.  
    # -----------------------------
    desired_break_num = 1
    dp_jump = 15
    robotjump_cutlist = []
    cut_pt = []

    for tr in good_tr:
        jumps_all = []
        for ax in range(3):
            y = outSIG[tr][:,ax]
            ind_jumpPT = detect_jumps_in_data(y, dp_jump, desired_break_num)
            
            jumps_all = jumps_all + [ind_jumpPT]
        
        jumps_all = make_a_properlist(jumps_all)
        jumps_all = np.array(jumps_all)
        
        if jumps_all.any():  # if jumps_all is not empty
            robotjump_cutlist = robotjump_cutlist + [tr]
            cut_pt = cut_pt + [np.unique(jumps_all, return_index=False, return_inverse=False)]
            # -----------------------------
            # fig = go.Figure()
            # config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})

            # LRsig = outSIG[tr][:, 0]
            # FBsig = outSIG[tr][:, 1]
            # UDsig = outSIG[tr][:, 2]
            # xxORG = list(range(len(LRsig)))

            # fig.add_trace(go.Scatter(x=xxORG, y=LRsig, name='cab LR', line = dict(color='red', width=2, dash='dash'), showlegend=True))
            # fig.add_trace(go.Scatter(x=xxORG, y=FBsig, name='cab FB', line = dict(color='green', width=2, dash='dash'), showlegend=True))
            
            # # Plot cut points
            # # Can plot more than a single point
            # if len(jumps_all) < 1:
                # pt = jumps_all*np.ones((2)) # must double the point : can not plot a singal point
            # else:
                # pt = jumps_all
            # pt = [int(x) for x in pt] # convert to integer
            # fig.add_trace(go.Scatter(x=pt, y=LRsig[pt], name='jumppts', mode='markers', marker=dict(color='red', size=10, line=dict(color='red', width=0)), showlegend=True))
        
            # fig.update_layout(title='tr : %d' % (tr), xaxis_title='data points', yaxis_title='cabin movement')
            # fig.show(config=config)
            # -----------------------------
    # -----------------------------
    
    # print('robotjump_cutlist : ' + str(robotjump_cutlist))
    # print('cut_pt : ' + str(cut_pt))


    
    
    
    
    
    # -----------------------------
    # Detect robot stall trials : the robot moves and then stops - the physical behavior is the same as robot jump but data does not return to zero.  Instead the last data point remains constant.
    # -----------------------------
    robotstall_cutlist = []
    robotstall_count = []
    robotstall_value = []
    for tr in good_tr:
        # print('tr : ' + str(tr))
        
        ax = axis_out[tr]
        
        c = Counter(outSIG[tr][:,ax])
        value = mode(outSIG[tr][:,ax])
        count = c[value]
        
        # If there is a value repeating for more than 1/3 of the signal length it is a stop trial
        if count > int(len(outSIG[tr][:,ax])/5):
            robotstall_cutlist = robotstall_cutlist + [tr]
            robotstall_count = robotstall_count + [count]
            robotstall_value = robotstall_value + [value]
            
            # fig = go.Figure()
            # config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
            # xxORG = list(range(len(outSIG[tr][:,ax])))
            # fig.add_trace(go.Scatter(x=xxORG, y=outSIG[tr][:,ax], name='sig', line = dict(color='red', width=2, dash='dash'), showlegend=True))
            # fig.show(config=config)
            
    # print('robotstall_cutlist : ' + str(robotstall_cutlist))
    robotstall_cutlist = [int(x) for x in robotstall_cutlist]
    
    # print('robotstall_count : ' + str(robotstall_count))
    # print('robotstall_value : ' + str(robotstall_value))
    
    
    
    
  
    # -----------------------------
    # Detect LR and FB trials that do not start at zero
    # UD trials start where LR and UD are not at zero, so you need to search by axis
    # -----------------------------
    LRFB_nonzero_start = []
    for tr in good_tr:
        if axis_out[tr] == 0 or axis_out[tr] == 1:
            if abs(outSIG[tr][0,1]) > 1 or abs(outSIG[tr][0,2]) > 1:
                LRFB_nonzero_start = LRFB_nonzero_start + [tr]
    # -----------------------------
    
    LRFB_nonzero_start = [int(x) for x in LRFB_nonzero_start]
    # print('LRFB_nonzero_start : ' + str(LRFB_nonzero_start))
    
    
    
    
    # -----------------------------
    # Detect if a trial is a UD initialization : this could happen if the robot stops
    # during a trial and UD initialization occurs after (we cut of the jump portion and keep UD initialization)
    # -----------------------------
    UD_initialization = []
    for tr in good_tr:
        tr_len = len(outSIG[tr])
            
        # Difference of first one fifth of data - want to detect initialization
        onethird_NEWend = int(tr_len/5)
        
        # This detects if the trial is UD : baseline near 100
        LR_tr = outSIG[tr][0:onethird_NEWend, 0]
        UD_tr = outSIG[tr][0:onethird_NEWend, 2]
        
        # This detects if LR is similar to UD : UD initialization event
        LRsum = abs(sum(LR_tr))
        UDsum = abs(sum(UD_tr))
        LR_UDrat = LRsum/UDsum  # Ratio is better because it accounts for LR and UD not being zero
        
        # if (LRmode >= 100) and (UDmode >= 100):
        # Second check :
        if (np.round(LR_UDrat) == 1) and (UD_tr[0] < 1) and (UDsum > 1):
            UD_initialization = UD_initialization + [tr]
    # -----------------------------
    
    UD_initialization = [int(x) for x in UD_initialization]
    # print('UD_initialization : ' + str(UD_initialization))
    
    
    
    
    
	# :::::::::::::::::::
    # Detect trials that are vertically too short (via cabin height) - tag the trial as a bad trial and remove these trials
    # :::::::::::::::::::
    # Remove LR and FB trials that have max height less than 10 = 10/100=0.1meters from the initial point. 
    # A good sub-threshold FB and LR trial has a max height of about 50 or more short height trials could have 
    # occurred due to timing problems with the robotic system (the sampling frequency was too fast).  
    # These erroneous trials also tend to be horizontally short (data-point wise), showing that the robotic 
    # sampling frequency was running faster than 250Hz and correct motion could not be achieved.  
    
    FBmax = np.zeros((Len_tr))
    LRmax = np.zeros((Len_tr))
    UDmax = np.zeros((Len_tr))
    tr_max = np.zeros((Len_tr))
    
    for tr in good_tr:
        # --------------------
        # fig = go.Figure()
        # config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        
        # LRsig = abs(outSIG[tr][:, 0])
        # FBsig = abs(outSIG[tr][:, 1])
        # UDsig = abs(outSIG[tr][:, 2])
        # xxORG = list(range(len(LRsig)))
        
        # fig.add_trace(go.Scatter(x=xxORG, y=LRsig, name='cab %s' % (varr['anom'][0]), line = dict(color='red', width=2, dash='dash'), showlegend=True))
        # fig.add_trace(go.Scatter(x=xxORG, y=FBsig, name='cab %s' % (varr['anom'][1]), line = dict(color='green', width=2, dash='dash'), showlegend=True))
        
        # fig.update_layout(title='tr : %d' % (tr), xaxis_title='data points', yaxis_title='cabin movement')
        # fig.show(config=config)
        # --------------------
        
        # (Better) Way 2 : Max of cabin movement
        LRmax[tr] = np.max(abs( outSIG[tr][:, 0] ))
        FBmax[tr] = np.max(abs( outSIG[tr][:, 1] ))
        UDmax[tr] = np.max(abs( outSIG[tr][:, 2] ))
        
        # print('LRmax[tr] : ' + str(LRmax[tr]))
        # print('FBmax[tr] : ' + str(FBmax[tr]))
        # print('[LRmax[tr], FBmax[tr]] : ' + str([LRmax[tr], FBmax[tr]]))
        
        # I want the max value out of FBmax and LRmax per trial
        tr_max[tr] = np.max([LRmax[tr], FBmax[tr], UDmax[tr]])
        # print('tr_max[tr] : ' + str(tr_max[tr]))
        
    # print('tr_max : ' + str(tr_max))
    
    # Goal: want to know which FB and LR trials have a max less than thresh
    # Reasoning for 7 : 100 for UD is about 3meters physically, therefore 7 is about 21cm
    # So, any movement less than about 20 cm total for one trial never happened for translation.
    # The maximum trial movement amplitude was physically greater than 20 cm.
    thresh = 7  # this needs to be bigger than marg=5 on line 290 in main_processing_steps.py
    newvec, tr_num_to_cut = findall(tr_max, '<', thresh)
    
    # print('tr_num_to_cut : ' + str(tr_num_to_cut))
    
    cut_trial_ver_short = np.unique(tr_num_to_cut)
    cut_trial_ver_short = [int(x) for x in cut_trial_ver_short]
    # print('cut_trial_ver_short : ' + str(cut_trial_ver_short))
    
    
    # :::::::::::::::::::
	
	
	
	# :::::::::::::::::::
    # Detect trials that are horizontally too short (via time) - tag the trial as a bad trial and remove these trials
    # :::::::::::::::::::
    # Remove short trials (horizontally short data)
    # Justification: In translation, the data consists of detection (8secs) + control (15secs) + reinitialization (10secs) so the maximum length of a trial is 33 seconds.  The data is sampled at 10Hz, so that is 330 data points per trial maximum.  If detection time is almost 0 seconds (fast detection), then the shortest a trial should be is 15+10=25 seconds.  We try to be lenient and account for delays and set the data cut off at 22 seconds.

    # NOTE: in translation the time goes from (trial event start to trial event start), in rotation the time vector restarted for every trial event (ie: stimulus, reinitialization, rest).

    # This means that in translation: I can not use the time to improve finding a better start-stop interval as we did in rotation.
    # I could have cut the data from trial event start to trial event start, but afterward we would still need to look at the cabin movement to figure out when each trial event (reinitialization, stimulus, reinitialization, rest) started.  We need a start-stop vector, to cut the data, for when stimulus starts and stops.

    # Since the data was initially cut by the cabin movement, we only care about the slope of the time vector.  

    # Reconstruct time : So, we shift the time vector in the with respect to the y-axis, and get the time vector for each trial (cut by the cabin movement).

    # Time difference : Then we take the time difference from start to end
    # Sometimes the sampling rate of the system changed (should be seen by change in slope of time) or 
    # data was not saved when it should have been saved due to real-time system (vertical jump/shift 
    # in time vector)
    
    tr_t_diff = np.zeros((Len_tr))
    dp_diff = np.zeros((Len_tr))
    time_idx = 2-1
    time_org = []
    for tr in good_tr:
        
        # print('new3_ind_st[tr] : ' + str(new3_ind_st[tr]))
        # print('new3_ind_end[tr] : ' + str(new3_ind_end[tr]))
        
        ttime = A[new3_ind_st[tr]:new3_ind_end[tr], time_idx]  # time in seconds
        # print('ttime : ' + str(ttime))
        
        dp_jump = 1     # detect jumps in the time greater than 1 second
        
        # Time vertically shifted and baseline shifted to zero
        tt2 = vertshift_segments_of_data_wrt_prevsegment(ttime, dp_jump)
        
        time_org = time_org + [tt2]
        
        tr_t_diff[tr:tr+1] = tt2[len(tt2)-1] - tt2[0]     # time difference
        dp_diff[tr:tr+1] = len(tt2)
        
        # ------------------------------
        # fig = go.Figure()
        # config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        # xxORG = list(range(len(range(new3_ind_st[tr], new3_ind_end[tr]))))
        
        # fig.add_trace(go.Scatter(x=xxORG, y=A[new3_ind_st[tr]:new3_ind_end[tr], time_idx], name='time', line = dict(color='black', width=2, dash='dash'), showlegend=True))
        
        # fig.add_trace(go.Scatter(x=xxORG, y=outSIG[tr][:,0], name='cab %s' % (varr['anom'][0]), line = dict(color='red', width=2, dash='dash'), showlegend=True))
        # fig.add_trace(go.Scatter(x=xxORG, y=outSIG[tr][:,1], name='cab %s' % (varr['anom'][1]), line = dict(color='green', width=2, dash='dash'), showlegend=True))
        # fig.add_trace(go.Scatter(x=xxORG, y=outSIG[tr][:,2], name='cab %s' % (varr['anom'][2]), line = dict(color='blue', width=2, dash='dash'), showlegend=True))
        
        # fig.show(config=config)
        # ------------------------------
        
        
    # Now, check the time length of each trial
    width_cutoff = 16    # cut data with time greater than 8 seconds
    cut_trial_hor_short = []
    for tr in good_tr:
        if tr_t_diff[tr] < width_cutoff:
            cut_trial_hor_short = cut_trial_hor_short + [tr]

    # Ensure that the index of cut vectors are integers
    # Change each entry of the list into an integer, output is a list
    cut_trial_hor_short = [int(x) for x in cut_trial_hor_short]
    # print('cut_trial_hor_short : ' + str(cut_trial_hor_short))
        
    
    # :::::::::::::::::::
    
    # ------------------------------
    
    return robotjump_cutlist, robotstall_cutlist, LRFB_nonzero_start, UD_initialization, cut_trial_ver_short, cut_trial_hor_short, new3_ind_st, new3_ind_end, outJOY, outSIG, outSIGCOM