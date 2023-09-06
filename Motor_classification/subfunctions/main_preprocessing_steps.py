import numpy as np
    
# Plotting
import plotly.graph_objects as go

# Data saving
import pickle

# Importing the statistics module
from statistics import mode, mean, median, multimode
import scipy.stats


from subfunctions.check_axes_assignmentPLOT import *
from subfunctions.cut_initial_trials import *
from subfunctions.full_sig_2_cell import *
from subfunctions.detect_bad_trials_rot import *
from subfunctions.detect_bad_trials_trans import *
from subfunctions.make_a_properlist import *
from subfunctions.process_index_for_trials import *
from subfunctions.process_index_for_UD_trials_timedetect import *
from subfunctions.process_index_for_FBLR_trials_timedetect import *

# Creating folders for image or data saving
import os


def main_preprocessing_steps(varr, A, a, b, c, s, NUMmark, yr, plotORnot_ALL, filemarker):


    # ------------------------------
    # Cut the data by looking at when the data crosses zero
    # How to regulate to cut each trial
    # 1) choose a threshold for marg_of_zero and see if there are zone points at each zero intersection
    # 2) make the marg_of_zero as small as possible to acheive zone points at each zero intersection
    # 3) then change num_pt_grp such that it excludes zone points NOT at a zero intersection
    # (to prevent cutting a trial in two - make num_pt_grp bigger than the number of points at an
    # undesired zone intersection)
    # ------------------------------
    if varr['which_exp'] == 'rot':
        marg_of_zero = 0.6  #0.07
        num_pt_grp =  1 #5
        plotORnot_allpts = 0
        plotORnot = 0  # 1 = show figures, 0 = do not show figures
    elif varr['which_exp'] == 'trans':
        marg_of_zero = 1
        num_pt_grp = 1
        plotORnot_allpts = 0
        plotORnot = 0  # 1 = show figures, 0 = do not show figures
        
    new3_ind_st, new3_ind_end = cut_initial_trials(varr, A, marg_of_zero, num_pt_grp, plotORnot_allpts, plotORnot, which_data2use='time')
    # ------------------------------
    
    ind_st_ORG = new3_ind_st
    ind_end_ORG = new3_ind_end
    
    print('new3_ind_st : ' + str(new3_ind_st))
    print('new3_ind_end : ' + str(new3_ind_end))
    
    # ------------------------------
    # Put the joystick, cabin actual, and cabin command into a compact format: 
    # a 3D list (depth=trials, row=dp per trial, col=axis in order LR/RO, FB/PI, UD/YAW)
    # ------------------------------
    outJOY, outSIG, outSIGCOM, outNOISE = full_sig_2_cell(A, a, b, c, new3_ind_st, new3_ind_end, varr)
    # ------------------------------
    
    
    # ------------------------------
    # Reverse the sign of the 3rd axis of the joystick to test if original orientation was correct
    if yr == 1:
        # There was confusion if yaw joystick direction was reversed
        for tr in range(len(outJOY)):
            for axxx in range(2,3):
                outJOY[tr][:,axxx] = -outJOY[tr][:,axxx]
    # ------------------------------
    
    
    # ------------------------------
    # We reuse the total length of detected trials often : this value should stay fixed for all of preprocessing, just until the very end when we remove bad trials at one-time
    Len_tr = len(new3_ind_st)
    # ------------------------------
    
    
    # ------------------------------
    # Axis detection
    # ------------------------------
    # Determine Experimental stimulation axes from data : Data-driven axis detection
    # This parameter should have been given in the experimental matrix, but
    # due to the real-time nature of the experiment there could have been delays
    # saving the data to file; the experimental matrix parameters may not be exactly
    # aligned with the robot data.
    # ------------------------------
    axis_out = np.zeros((Len_tr))
    
    if varr['which_exp'] == 'rot':
        val2 = np.zeros((Len_tr))
        for tr in range(Len_tr):
            #print('tr : ' + str(tr))
            
            # Same method for FB, LR
            # a) sum the first 1/4 of each cabin movement signal for each axis : justification is that the first part of the signal is likely pure stimulus without participant influence

            # Calculate the sum of the two signals at the beginning and determine which is larger
            starter = 0
            ender = int(np.round(len(outSIG[tr])/4))
            temp2 = []
            for ax in range(3):        
                # get the max or sum per axis column
                temp2 = temp2 + [abs(sum( outSIG[tr][starter:ender, ax] ))]

            axis_out[tr] = np.argmax(temp2)
            val2[tr] = np.max(temp2)
            
    elif varr['which_exp'] == 'trans':
        for tr in range(Len_tr):
            tr_len = len(outSIG[tr])
            
            # Difference of first one third of data - want to detect reinitialization
            onethird_NEWend = int(tr_len/3)
            
            LR_tr = outSIG[tr][0:onethird_NEWend, 0]
            FB_tr = outSIG[tr][0:onethird_NEWend, 1]
            UD_tr = outSIG[tr][0:onethird_NEWend, 2]
            
            # First check if curve is UD no stim trial (LR and UD constant around 100)
            LRmode = mode(np.round(abs(LR_tr)))
            UDmode = mode(np.round(abs(UD_tr)))
            
            if LRmode >= 100 and UDmode >= 100:
                # Trial is UD no stim trial
                axis_out[tr] = 2
            else:
                # Baseline shift all curves to zero, abs
                LR_tr_zero = abs(LR_tr - LR_tr[0])
                FB_tr_zero = abs(FB_tr - FB_tr[0])
                UD_tr_zero = abs(UD_tr - UD_tr[0])

                LRsum = sum(LR_tr_zero)
                FBsum = sum(FB_tr_zero)
                UDsum = sum(UD_tr_zero)

                whichax = [LRsum, FBsum, UDsum]
                axis_out[tr] = np.argmax(whichax)
    # ------------------------------
    
    
    axis_out = [int(x) for x in axis_out]
    print('axis_out : ' + str(axis_out))
    
    
    # ------------------------------
    # Speed stimulus detection and detrended time
    # ------------------------------
    # Way 1: Experimental collected data axis detection
    trialnum_org = np.zeros((Len_tr))
    axis_org = np.zeros((Len_tr))
    speed_stim_org = np.zeros((Len_tr))
    # ................
    tar_ang_speed_val_co = np.zeros((Len_tr))
    
    
    time_org = []
    
    for tr in range(Len_tr):
        st = int(new3_ind_st[tr])
        stp = int(new3_ind_end[tr])
        
        trialnum_org[tr] = mode(A[st:stp, 1-1])   # trial count
        axis_org[tr] = mode(A[st:stp, 14-1])  # axis that was stimulated: axis that was stimulated: axis of interest (1:roll, 2:pitch, 3:yaw), (1:left/right, 2:forward/back, 3:up/down)
        
        # ------------------
        # Time initally calculated for plotting
        ttime = A[new3_ind_st[tr]:new3_ind_end[tr], 2-1]  # time in seconds
        dp_jump = 1     # detect jumps in the time greater than 1 second
        tt2 = vertshift_segments_of_data_wrt_prevsegment(ttime, dp_jump) # Time vertically shifted and baseline shifted to zero
        time_org =  time_org + [tt2]
        # ------------------
        
        if varr['which_exp'] == 'rot':
        	speed_stim_org[tr] = mode(A[st:stp, 19-1])   # gradual stimulation of choosen axis of stimulation ([1.2500; 0.5000; 0; -0.5000; -1.2500]
        	# ................
        	# Alternative way to obtain the speed_stim : this was more accurate because it was less affected by the delay
        	# The axis were confirmed to be correct : remember the joystick roll and pitch were reversed but target angular speed order was correct
        	val = axis_out[tr]  # 0=RO, 1=PI, 2=YA
        	if val == 0:
        		tar_ang_speed_val_co[tr] = mode(A[st:stp, 10-1])
        	elif val == 1:
        		tar_ang_speed_val_co[tr] = mode(A[st:stp, 11-1])
        	elif val == 2:
        		tar_ang_speed_val_co[tr] = mode(A[st:stp, 12-1])
        	# ................
        elif varr['which_exp'] == 'trans':
        	speed_stim_org[tr] = mode(A[st:stp, 10-1]) + mode(A[st:stp, 11-1]) + mode(A[st:stp, 12-1])
        	# target translational speed 1 + target translational speed 2 + target translational speed 3
        	# speed_stim_org is the sum of the target angular speed will give the speed_stim column
        	# ................
        	# Correct order :
        	val = axis_out[tr]  # 0=LR, 1=FB, 2=UD
        	if val == 0: 
        		tar_ang_speed_val_co[tr] = mode(A[st:stp, 10-1])
        	elif val == 1:
        		tar_ang_speed_val_co[tr] = mode(A[st:stp, 11-1])
        	elif val == 2:
        		tar_ang_speed_val_co[tr] = mode(A[st:stp, 12-1])
        	# ................
    # ------------------------------
    
    
    
    # ------------------------------
    # Check if axes are assigned correctly to trials: PLOTTING
    # ------------------------------
    if plotORnot_ALL == 1:
        filename = 'images_original_%s' % (varr['which_exp'])
        max_sig_val = check_axes_assignmentPLOT(s, outJOY, outSIG, axis_out, varr, filename, time_org)
    # ------------------------------
    
    
    # ------------------------------
    # Detect bad trials
    # ------------------------------
    g_rej_vec = np.zeros((Len_tr))  # global reject vector
    
    good_tr = list(range(Len_tr))
    
    if varr['which_exp'] == 'rot':
        cut_trial_ver_short, cut_trial_hor_short = detect_bad_trials_rot(Len_tr, good_tr, outSIG, axis_out, A, new3_ind_st, new3_ind_end)
        
        print('cut_trial_ver_short : ' + str(cut_trial_ver_short))
        print('cut_trial_hor_short : ' + str(cut_trial_hor_short))
        
        g_rej_vec[cut_trial_ver_short] = 30
        g_rej_vec[cut_trial_hor_short] = 40
        
    elif varr['which_exp'] == 'trans':
        robotjump_cutlist, robotstall_cutlist, LRFB_nonzero_start, UD_initialization, cut_trial_ver_short, cut_trial_hor_short, new3_ind_st, new3_ind_end, outJOY, outSIG, outSIGCOM = detect_bad_trials_trans(axis_out, Len_tr, good_tr, outJOY, outSIG, outSIGCOM, A, a, b, c, new3_ind_st, new3_ind_end, varr)
        
        print('robotjump_cutlist : ' + str(robotjump_cutlist))
        print('robotstall_cutlist : ' + str(robotstall_cutlist))
        print('LRFB_nonzero_start : ' + str(LRFB_nonzero_start))
        print('UD_initialization : ' + str(UD_initialization))
        print('cut_trial_ver_short : ' + str(cut_trial_ver_short))
        print('cut_trial_hor_short : ' + str(cut_trial_hor_short))
        
        g_rej_vec[robotjump_cutlist] = 10
        g_rej_vec[robotstall_cutlist] = 15
        g_rej_vec[LRFB_nonzero_start] = 20
        g_rej_vec[UD_initialization] = 25
        g_rej_vec[cut_trial_ver_short] = 30
        g_rej_vec[cut_trial_hor_short] = 40
    # ------------------------------


    # ------------------------------
    # Remove the bad trials from trials
    # ------------------------------
    if varr['which_exp'] == 'rot':
        cutall = cut_trial_ver_short, cut_trial_hor_short
    elif varr['which_exp'] == 'trans':
        cutall = robotjump_cutlist, robotstall_cutlist, LRFB_nonzero_start, UD_initialization, cut_trial_ver_short, cut_trial_hor_short

    tr_2_cut = make_a_properlist(cutall)
    tr_2_cut.sort()
    cut_tr = np.unique(tr_2_cut)
    cut_tr = [int(x) for x in cut_tr]
    # -------------------------------------
    
    print('cut_tr BEFORE the final stoppoint : ' + str(cut_tr))
    
    
    # -------------------------------------
    # Get a total list of good trials : good_tr are the remaining trials to process
    # -------------------------------------
    longvec = np.array(range(Len_tr))
    shortvec = cut_tr
    idx_from_long_not_in_short = np.setdiff1d(longvec, shortvec)
    good_tr = idx_from_long_not_in_short
    good_tr = [int(x) for x in good_tr]
    # -------------------------------------
    
    # print('good_tr : ' + str(good_tr))
    
    # -------------------------------------
    # Keep vectors the same original number of trials, 
    # put NUMmark for trials that are bad trials (horizontally and vertically short)
    # We update the start-stop index (new3_ind_st, new3_ind_end) to avoid selecting junk data.
    # -------------------------------------
    for tr in cut_tr:
        new3_ind_st[tr] = NUMmark  # To prevent distraction of bad trials when plotting, but we leave the trial
        new3_ind_end[tr] = NUMmark # To prevent distraction of bad trials when plotting, but we leave the trial
    # ------------------------------
	
    # print('new3_ind_st : ' + str(new3_ind_st))
    # print('new3_ind_end : ' + str(new3_ind_end))
    
    # -----------------------------
    # Update trial cut data matrices - so that AFTER this point the bad trials are NOT PLOTTED
    # to prevent distraction (we plot an empty plot to denote that the trial is bad).
    # We change the start-stop index (new3_ind_st, new3_ind_end) so we initially cut "good-looking" data,
    # basically cutting away when the robot was reinitializing or when the robot stalled/jumped and the 
    # data is still there.
    outJOY, outSIG, outSIGCOM, outNOISE = full_sig_2_cell(A, a, b, c, new3_ind_st, new3_ind_end, varr)
    # -----------------------------
    
    # AFTER, this point when we talk about "bad trials",it is actually for a real trial. For those,
    # we keep track of the trial number and select the trials that are on the "good trial" list.
    
    # -------------------------------------
    # Cut the good trial data more precisely depending on (RO/LR, PI/FB, YA/UD)
    # -------------------------------------
    starttrial_index = np.zeros((Len_tr, 1))
    stoptrial_index = np.zeros((Len_tr, 1))
    
    starttrial_index = [int(x) for x in starttrial_index] # to make axis in terms of python indexing
    starttrial_index = make_a_properlist(starttrial_index)
    
    stoptrial_index = [int(x) for x in stoptrial_index] # to make axis in terms of python indexing
    stoptrial_index = make_a_properlist(stoptrial_index)
    
    marg = 5  # margin around signal baseline for zonemax area
    
    
    # -------------------------------------
    
    outer_fn = 'main_processing_verificationOF_each_trial'
    # create a directory for saving images
    if not os.path.exists("%s%s%s" % (varr['main_path1'], filemarker, outer_fn)):
        os.mkdir("%s%s%s" % (varr['main_path1'], filemarker, outer_fn))
    
    # -------------------------------------
    
    filename = 'images_initialdetection_%s_s%d' % (varr['which_exp'], s)
    # create a directory for saving images
    if not os.path.exists("%s%s%s%s%s" % (varr['main_path1'], filemarker, outer_fn, filemarker, filename)):
        os.mkdir("%s%s%s%s%s" % (varr['main_path1'], filemarker, outer_fn, filemarker, filename))
        
    # -------------------------------------
    
    
    if varr['which_exp'] == 'rot':
        plotORnot = 0  # 1 = show figures, 0 = do not show figures
        plotORnot_derv = 0  # 1 = show figures, 0 = do not show figures
        
        starttrial_index, stoptrial_index = process_index_for_trials(s, good_tr, outSIG, marg, varr, starttrial_index, stoptrial_index, plotORnot, plotORnot_derv, axis_out, filename)
    elif varr['which_exp'] == 'trans':
        for tr in good_tr:
            # print('tr : ' + str(tr))
            
            if axis_out[tr] == 2:  # UD
                # print('Running UD processing')
                # Do special cutting process for UD trials only
                plotORnot = 0  # 1 = show figures, 0 = do not show figures
                plotORnot_derv = 0  # 1 = show figures, 0 = do not show figures
                starttrial_index, stoptrial_index = process_index_for_UD_trials_timedetect(s, tr, outSIG, starttrial_index, stoptrial_index, plotORnot, plotORnot_derv, marg, filename)
                
            elif axis_out[tr] < 2:  # FB or LR
                # print('Running FB or LR processing')
                # Do special cutting process for LR/FB trials only
                plotORnot = 0  # 1 = show figures, 0 = do not show figures
                plotORnot_derv = 0 # 1 = show figures, 0 = do not show figures
                starttrial_index, stoptrial_index = process_index_for_FBLR_trials_timedetect(s, tr, outSIG, marg, varr, starttrial_index, stoptrial_index, plotORnot, plotORnot_derv, axis_out, filename)
    # ------------------------------
    
    
    
    # ------------------------------  
    # Update global_index (new3_ind_st, new3_ind_end) with respect to local_index (starttrial_index, stoptrial_index)!!
    
    # Get the updated new4_ind_st and new4_ind_end
    new3_ind_st = np.array(new3_ind_st) + np.array(starttrial_index)
    new3_ind_st = make_a_properlist(new3_ind_st)
    # print('new3_ind_st : ' + str(new3_ind_st))

    new3_ind_end = np.array(new3_ind_st) + (np.array(stoptrial_index) - np.array(starttrial_index) )
    new3_ind_end = make_a_properlist(new3_ind_end)
    # print('new3_ind_end : ' + str(new3_ind_end))
    # ------------------------------ 
    
    
    # ------------------------------
    # Re-evaluation step of good trials considering the choosen starttrial_index and stoptrial_index
    # The final trial could be too short or have an unexpected jump depending on the final cut, remove these trials.
    # ------------------------------
    # Detect bad trials
    # ------------------------------
    if varr['which_exp'] == 'rot':
        final_cut_trial_ver_short, final_cut_trial_hor_short = detect_bad_trials_rot(Len_tr, good_tr, outSIG, axis_out, A, new3_ind_st, new3_ind_end)
        
        print('final_cut_trial_ver_short : ' + str(final_cut_trial_ver_short))
        print('final_cut_trial_hor_short : ' + str(final_cut_trial_hor_short))
        
        g_rej_vec[final_cut_trial_ver_short] = 30
        g_rej_vec[final_cut_trial_hor_short] = 40
        
    elif varr['which_exp'] == 'trans':
        final_robotjump_cutlist, final_robotstall_cutlist, final_LRFB_nonzero_start, final_UD_initialization, final_cut_trial_ver_short, final_cut_trial_hor_short, new3_ind_st, new3_ind_end, outJOY, outSIG, outSIGCOM = detect_bad_trials_trans(axis_out, Len_tr, good_tr, outJOY, outSIG, outSIGCOM, A, a, b, c, new3_ind_st, new3_ind_end, varr)
        
        print('final_robotjump_cutlist : ' + str(final_robotjump_cutlist))
        print('final_robotstall_cutlist : ' + str(final_robotstall_cutlist))
        print('final_LRFB_nonzero_start : ' + str(final_LRFB_nonzero_start))
        print('final_UD_initialization : ' + str(final_UD_initialization))
        print('final_cut_trial_ver_short : ' + str(final_cut_trial_ver_short))
        print('final_cut_trial_hor_short : ' + str(final_cut_trial_hor_short))
        
        g_rej_vec[final_robotjump_cutlist] = 10
        g_rej_vec[final_robotstall_cutlist] = 15
        g_rej_vec[final_LRFB_nonzero_start] = 20
        g_rej_vec[final_UD_initialization] = 25
        g_rej_vec[final_cut_trial_ver_short] = 30
        g_rej_vec[final_cut_trial_hor_short] = 40
    # ------------------------------


    # ------------------------------
    # Remove the bad trials from trials
    # ------------------------------
    if varr['which_exp'] == 'rot':
        cutall = final_cut_trial_ver_short, final_cut_trial_hor_short
    elif varr['which_exp'] == 'trans':
        cutall = final_robotjump_cutlist, final_robotstall_cutlist, final_LRFB_nonzero_start, final_UD_initialization, final_cut_trial_ver_short, final_cut_trial_hor_short

    tr_2_cut = make_a_properlist(cutall)
    tr_2_cut.sort()
    cut_tr_final = np.unique(tr_2_cut)
    cut_tr_final = [int(x) for x in cut_tr_final]
    # -------------------------------------
    
    print('cut_tr_final AFTER the final stoppoint : ' + str(cut_tr_final))
    
    # We note the bad trials after finding the final start-stop point in g_rej_vec, 
    # and empty out the index value in new3_ind_st and new3_ind_end (this is to prevent errors in the
    # future and to NOT plot bad trials - to check if all cut trials were removed (a bit redundant but a check)).
    
    # -------------------------------------
    # Get a total list of good trials : good_tr are the remaining trials to process
    # -------------------------------------
    longvec = np.array(range(Len_tr))
    shortvec = make_a_properlist([cut_tr, cut_tr_final])
    
    
    
    idx_from_long_not_in_short = np.setdiff1d(longvec, shortvec)
    good_tr = idx_from_long_not_in_short
    good_tr = [int(x) for x in good_tr]
    # -------------------------------------
    
    print('good_tr : ' + str(good_tr))
    
    # -------------------------------------
    # Keep vectors the same original number of trials, 
    # put NUMmark for trials that are bad trials (horizontally and vertically short)
    # -------------------------------------
    for tr in cut_tr:
        new3_ind_st[tr] = NUMmark  # To prevent distraction of bad trials when plotting, but we leave the trial
        new3_ind_end[tr] = NUMmark # To prevent distraction of bad trials when plotting, but we leave the trial
    # ------------------------------
    
    # In the next step, the trials are removed using g_rej_vec.
    
    # ------------------------------
    
    
    # ------------------------------ 
    # Need to redo outSIG, outJOY, and outSIGCOM.
    # And, redo trialnum, axis_org, time_org, speed_stim_org with the FINAL start-stop points
    outJOY, outSIG, outSIGCOM, outNOISE = full_sig_2_cell(A, a, b, c, new3_ind_st, new3_ind_end, varr)
    
    time_org = []
    for tr in range(Len_tr):
        st = new3_ind_st[tr]
        stp = new3_ind_end[tr]
        
        if st == stp:
            tt2 = 0
        else:
            trialnum_org[tr] = mode(A[st:stp, 1-1])   # trial count
            axis_org[tr] = mode(A[st:stp, 14-1])  # axis that was stimulated: axis that was stimulated
            
            ttime = A[st:stp, 2-1]  # time in seconds
            dp_jump = 1     # detect jumps in the time greater than 1 second
            tt2 = vertshift_segments_of_data_wrt_prevsegment(ttime, dp_jump)  # Time vertically shifted and baseline shifted to zero
            
            if varr['which_exp'] == 'rot':
            	speed_stim_org[tr] = mode(A[st:stp, 19-1])   # gradual stimulation of choosen axis of stimulation ([1.2500; 0.5000; 0; -0.5000; -1.2500]
            	# ................
            	# Alternative way to obtain the speed_stim : this was more accurate because it was less affected by the delay
            	# The axis were confirmed to be correct : remember the joystick roll and pitch were reversed but target angular speed order was correct
            	val = axis_out[tr]  # 0=RO, 1=PI, 2=YA
            	if val == 0:
            		tar_ang_speed_val_co[tr] = mode(A[st:stp, 10-1])
            	elif val == 1:
            		tar_ang_speed_val_co[tr] = mode(A[st:stp, 11-1])
            	elif val == 2:
            		tar_ang_speed_val_co[tr] = mode(A[st:stp, 12-1])
            	# ................
            elif varr['which_exp'] == 'trans':
            	speed_stim_org[tr] = mode(A[st:stp, 10-1]) + mode(A[st:stp, 11-1]) + mode(A[st:stp, 12-1])
            	# target translational speed 1 + target translational speed 2 + target translational speed 3
            	# speed_stim_org is the sum of the target angular speed will give the speed_stim column
            	# ................
            	# Correct order :
            	val = axis_out[tr]  # 0=LR, 1=FB, 2=UD
            	if val == 0: 
            		tar_ang_speed_val_co[tr] = mode(A[st:stp, 10-1])
            	elif val == 1:
            		tar_ang_speed_val_co[tr] = mode(A[st:stp, 11-1])
            	elif val == 2:
            		tar_ang_speed_val_co[tr] = mode(A[st:stp, 12-1])
				# ................
        time_org =  time_org + [tt2]
    # ------------------------------
    
    
    # ------------------------------
    # Confirmation check of axis : to see how similar data-driven axis_out is to the experimental matrix axis_org
    # Pearson correlation coefficient : corr (x,y) = cov (x,y) / (std (x) * std (y))
    corr_axis_out = np.corrcoef(axis_org, axis_out) # outputs a correlation matrix
    corr_axis_out = corr_axis_out[0,1]
    print('corr_axis_out : ' + str(corr_axis_out))
    # ------------------------------
    
    
    # ------------------------------
    # Speed stim sign measurements : There are multiple ways to calculate the speed stimulus direction
    # ------------------------------
    speed_stim_org_sign = make_a_properlist(np.sign(speed_stim_org))  # 1st way to calculate sign of speed stim (Experimental matrix)
    speed_stim_tas_sign = make_a_properlist(np.sign(tar_ang_speed_val_co))   # 2nd way to calculate sign of speed stim (Experimental matrix)
    # ------------------------------
        
        
    # ------------------------------
    FRT = np.zeros((Len_tr, 1))
    for tr in range(Len_tr):
        frt_tr = A[new3_ind_st[tr]:new3_ind_end[tr], 15-1]  # Taact or fRT - time at which the subject pushes/controls the joystick (first reacts)
        # print('frt_tr : ' + str(frt_tr))
        
        if frt_tr.any():  # if frt_tr is not empty
            FRT[tr] = np.max(frt_tr)
        
        # --------------------
        # fig = go.Figure()
        # config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        
        # xxORG = list(range(len(frt_tr)))
        # fig.add_trace(go.Scatter(x=xxORG, y=frt_tr, name='FRT', line = dict(color='red', width=2, dash='dash'), showlegend=True))
        
        # fig.update_layout(title='s : %d, tr : %d, first time respone : %f' % (s, tr, FRT), xaxis_title='data points', yaxis_title='FRT')
        # fig.show(config=config)
        # --------------------
    # ------------------------------
    
    return starttrial_index, stoptrial_index, speed_stim_org_sign, speed_stim_org, tar_ang_speed_val_co, speed_stim_tas_sign, axis_out, axis_org, new3_ind_st, new3_ind_end, g_rej_vec, outJOY, outSIG, outSIGCOM, outNOISE, corr_axis_out, trialnum_org, time_org, FRT, good_tr
