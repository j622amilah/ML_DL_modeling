import numpy as np

# Plotting
import plotly.graph_objects as go

# Data saving
import pickle

# Importing the statistics module
from statistics import mode, mean, median, multimode
import scipy.stats

# Personal python functions
from subfunctions.findall import *
from subfunctions.standarization_check_if_joy_moved import *
from subfunctions.detect_sig_change_wrt_baseline import *
from subfunctions.standarization_fill_in_matrix import *
from subfunctions.standarization_plotting import *



def standarization_notebadtrials(starttrial_index, stoptrial_index, axis_out, outSIG, outJOY, marg_joy, varr, s, strictness, dirC, good_tr):

    NOjoyMOVE = 70  #  an arbitraily number to denote the joystick was not moved in final X matrix
    
    # ------------------------------
    # Start Joystick analysis
    # ------------------------------
    # b) Only look at non-defective trials and initalize
    # num_of_trials should be equal to len(idx_alltr) I think

    # print('axis_out :' + str(axis_out))
    # print('starttrial_index :' + str(starttrial_index))
    # print('stoptrial_index :' + str(stoptrial_index))

    # nimp_val, idx_alltr = findall(stoptrial_index, '!=', 0)

    # print('idx_alltr :' + str(idx_alltr))
    idx_alltr = good_tr
    
    # c) Start loops over non-defective trials: search for when/what direction joystick and cabin moved
    num_of_tr = len(idx_alltr)
    # print('Number of trials to standardize :' + str(num_of_tr))

    # Initialize normalized_outSIG
    # Note : normalized_outSIG and outSIG need to be the same organization (list - array - list - list)
    normalized_outSIG = []
    for tr in range(len(outSIG)):
        if not outSIG[tr].any():  # if outSIG[tr] is empty = False
            # You can not put an empty list because python will not append an empty list in a list 
            normalized_outSIG = normalized_outSIG + [np.zeros((1,3))]
        else:
            normalized_outSIG = normalized_outSIG + [np.zeros((len(outSIG[tr]),3))]
            
    # print('length of outSIG : ' + str(len(outSIG)))
    # print('length of normalized_outSIG : ' + str(len(normalized_outSIG)))
    tr_c = 0

    # These are just for plotting and confirmation that it is working
    binary_marker = np.zeros((num_of_tr,1))
    direction_marker = np.zeros((num_of_tr,1))
    dir_meaning_marker = np.zeros((num_of_tr,1))  # direction marker meaning wrt binary(joy-cab follow)


    cut_trial_standard_move = []
    cut_trial_standard_dir = []

    for tr in idx_alltr:
        # print('tr :' + str(tr))
        
        # Step 1 : check if joystick moved
        joy_ax_dir, joy_ax_val, joy_ax_index = standarization_check_if_joy_moved(tr, outJOY, marg_joy)
        
        # Step 2 : Based on joystick movement: 1) continue analysis with cabin, 2) stop analysis --> Next trial
        # ------------------------------
        # Based on if joystick was moved : 1) continue analysis with cabin, 2) stop analysis --> Next trial
        # ------------------------------
        # print('joy_ax_index :' + str(joy_ax_index))
        
        if sum(joy_ax_index) == 0:
            # Joystick is never moved :  2) stop analysis --> Next trial

            # We keep the trial : there is no way to test if cabin follows joystick, and it is irrelavant 
            # the trial will be counted as a non-response to stimulus, trusting that the cabin would have 
            # moved with joystick movement 
            binary_marker[tr_c] = NOjoyMOVE
            direction_marker[tr_c] = NOjoyMOVE
            dir_meaning_marker[tr_c] = NOjoyMOVE
            
            ind_joy_ax_moved = NOjoyMOVE
            cab_index_viaSS = NOjoyMOVE
        else:
            # 1) continue analysis with cabin
            
            # ------------------------------
            # Indexing : Normalize outSIG for specific trials
            # e) Normalize physical cabin motion from [-1, 1] to make a fair comparison with joystick measure
            # print('length of outSIG[tr] : ' + str(len(outSIG[tr])))

            # Takes the max of the entire matrix
            maxcab = np.max(abs(outSIG[tr]))
            # print('maxcab :' + str(maxcab))

            if maxcab < 1:
                # should be zero
                normalized_outSIG[tr] = outSIG[tr]
            else:
                normalized_outSIG[tr] = outSIG[tr]/maxcab
            # ------------------------------
            

            # f) NOTE: we only look at joystick ---> cabin movement for trials and axes that the joystick were moved
            nimp_val, ind_joy_ax_moved = findall(joy_ax_index, '!=', 0)

            # print('ind_joy_ax_moved :' + str(ind_joy_ax_moved))

            cab_index = np.zeros((3,1))
            cab_dir = np.zeros((3,1))
            cab_val = np.zeros((3,1))
            cab_index_viaSS = np.zeros((3,1))
            # ------------------------------

            # g) Only calculate cabin movement for when the joystick was moved
            # if the joystick axis was NOT MOVED, the cab_values remain zeros from the initialization above
            # We look to see if the cabin moved after the joystick moved
            for ax in ind_joy_ax_moved:
                
                # print('int(joy_ax_index[ax]) :' + str(int(joy_ax_index[ax])))
                
                yall = normalized_outSIG[tr][int(joy_ax_index[ax])::, ax]     # Search from when the joystick was 1st moved to end - prevents false detection of cabin movement before the joystick was moved
                baseline = yall[0]

                # We look for change from the  baseline start, the first cabin movement point
                dpOFsig_in_zoneCAB, indexOFsig_in_zoneCAB, dp_sign_not_in_zoneCAB, indexOFsig_not_in_zoneCAB = detect_sig_change_wrt_baseline(yall, baseline, marg_joy)
                
                plotORnot = 0
                if plotORnot == 1:
                    # ------------------------------
                    fig = go.Figure()
                    config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
                    t = np.multiply(range(0, len(yall)), 1)

                    fig.add_trace(go.Scatter(x=t, y=yall, name='sig', line = dict(color='black', width=2, dash='solid'), showlegend=True))
                    plotb = baseline*np.ones((len(yall)))
                    fig.add_trace(go.Scatter(x=t, y=plotb))

                    outval = np.array(indexOFsig_in_zoneCAB)
                    fig.add_trace(go.Scatter(x=outval, y=dpOFsig_in_zoneCAB, name='pts inside of zone', mode='markers', marker=dict(color='green', size=10, line=dict(width=0)), showlegend=True))

                    outval1 = np.array(indexOFsig_not_in_zoneCAB)
                    fig.add_trace(go.Scatter(x=outval1, y=dp_sign_not_in_zoneCAB, name='pts outside of zone', mode='markers', marker=dict(color='magenta', size=10, line=dict(width=0)), showlegend=True))

                    fig.update_layout(title='Cabin motion', xaxis_title='data points', yaxis_title='Cabin position')

                    fig.show(config=config)
                    # ------------------------------

                # print('dpOFsig_in_zoneCAB :' + str(dpOFsig_in_zoneCAB))
                # print('indexOFsig_in_zoneCAB :' + str(indexOFsig_in_zoneCAB))
                # print('dp_sign_not_in_zoneCAB :' + str(dp_sign_not_in_zoneCAB))
                # print('indexOFsig_not_in_zoneCAB :' + str(indexOFsig_not_in_zoneCAB))


                # Outputs: cab_index, cab_val, cab_dir
                if not indexOFsig_not_in_zoneCAB.any():  # if indexOFsig_not_in_zoneCAB is empty = False
                    # cabin is not moved outside of the zone
                    cab_dir[ax] = 0
                    cab_val[ax] = 0
                    cab_index[ax] = 0
                    cab_index_viaSS[ax] = 0
                else:
                    diff = dp_sign_not_in_zoneCAB[0] - dpOFsig_in_zoneCAB[0]     # diff should always around marg_joy_cabin
                    cab_dir[ax] = np.sign(diff)
                    cab_val[ax] = dp_sign_not_in_zoneCAB[0]

                    # cabin index with respect to first joystick movement
                    cab_index[ax] = indexOFsig_not_in_zoneCAB[0]

                    # cabin index with respect to entire trial
                    cab_index_viaSS[ax] = (cab_index[ax] - 1) + joy_ax_index[ax]

            # print('cab_dir :' + str(cab_dir))
            # print('cab_val :' + str(cab_val))
            # print('cab_index :' + str(cab_index))
            # print('cab_index_viaSS :' + str(cab_index_viaSS))
            
            
            # -------------------------------------------------
            # Classify the order of movements
            # -------------------------------------------------
            bin_m, sca_ind, d_m = standarization_fill_in_matrix(ind_joy_ax_moved, cab_index_viaSS, joy_ax_index, dirC, cab_dir, joy_ax_dir)
            # -------------------------------------------------
            
            
            
            # -------------------------------------------------
            # Reducing the evaluation of each axes (3x1 vector) into a 
            # decision of whether to remove the trial or not
            # -------------------------------------------------
            # a] Movement : Did the cabin move after joystick movement? NO=note trial number
            if strictness == 0:
                # Lenient - Cases : 1) all 3 axes moved, 2) 2 axes moved, 3) 1 axis moved
                if len(ind_joy_ax_moved) == 3:
                    # did at least 2 out of 3 axes have joy-cabin follow motion
                    binary_marker[tr_c] = mode(bin_m)
                    if mode(bin_m) == 0: # 2 trials are 0, meaning they do not follow, so remove the trial
                        cut_trial_standard_move = cut_trial_standard_move + [tr]
                else:
                    # if 2 axes : did at least 1 out of 2 axes have joy-cabin follow motion
                    # if 1 axis : did the 1 axis have joy-cabin follow motion
                    for i in ind_joy_ax_moved:
                        if bin_m[i] == 1:
                            binary_marker[tr_c] = 1
                    
                        if bin_m[i] == 0:
                            cut_trial_standard_move = cut_trial_standard_move + [tr]
                            break
            elif strictness == 1:
                # Strict : if at least one axis of joy-cabin follow motion is wrong
                for i in ind_joy_ax_moved:
                    if bin_m[i] == 0:
                        binary_marker[tr_c] = 0
                        cut_trial_standard_move = cut_trial_standard_move + [tr]
                        break
            
            # ............................
            
            # b] Direction : Did the cabin move after joystick movement in the correct
            # direction? NO=note trial number
            if strictness == 0:
                # Lenient - Cases : 1) all 3 axes moved, 2) 2 axes moved, 3) 1 axis moved
                if len(ind_joy_ax_moved) == 3:
                    # did at least 2 out of 3 axes have joy-cabin follow motion
                    direction_marker[tr_c] = mode(d_m)
                    if mode(bin_m) == 0: # 2 trials are 0, meaning they do not follow, so remove the trial
                        cut_trial_standard_dir = cut_trial_standard_dir + [tr]
                else:
                    # if 2 axes : did at least 1 out of 2 axes have joy-cabin follow motion
                    # if 1 axis : did the 1 axis have joy-cabin follow motion
                    for i in ind_joy_ax_moved:
                        if d_m[i] == 1:
                            direction_marker[tr_c] = 1
                    
                        if bin_m[i] == 0:
                            cut_trial_standard_dir = cut_trial_standard_dir + [tr]
                            break
            elif strictness == 1:
                # Strict : if at least one axis of joy-cabin follow motion is wrong
                for i in ind_joy_ax_moved:
                    if bin_m[i] == 0:
                        direction_marker[tr_c] = 0
                        cut_trial_standard_dir = cut_trial_standard_dir + [tr]
                        break
            # ............................
            
            
            # -------------------------------------------------
        
        
        # -------------------------------------------------
        # Plotting
        # -------------------------------------------------
        # Making sense of the measures:
        dircor = 'NA' # not relavant if joy does not move, or if cabin does not follow joy
        if binary_marker[tr_c] == 0:
            descript = 'Wrong_joycab' # 'joy_move_cabinNOT_move'
        elif binary_marker[tr_c] == 1:
            descript = 'Correct_joycab' # 'joy_move_cabin_move'
            if direction_marker[tr_c] == 1:
                dircor = 'correct_dir'  
            else:
                dircor = 'NOTcorrect_dir'
        elif binary_marker[tr_c] == 70:
            descript = 'joyNOT_move'
        
        
        # A more meaningful direction_marker : discards irrelavant trials, 
        # we are only interested in trials where cabin follows joy
        if dircor == 'NA':
            dir_meaning_marker[tr_c] = 80   # cabin does not follow joy or joy does not move, irrelavant
        elif dircor == 'correct_dir':
            dir_meaning_marker[tr_c] = 1    # cabin follows joy, and direction correct
        elif dircor == 'NOTcorrect_dir':
            dir_meaning_marker[tr_c] = 0    # cabin follows joy, and direction not correct
        
        
        plotORnot = 0
        if plotORnot == 1:
            # ------------------------------
            # Make a folder for saving images
            if not os.path.exists("images_standard"):
                os.mkdir("images_standard")
            # ------------------------------
            standarization_plotting(s, tr, normalized_outSIG, outJOY, axis_val, varr, ind_joy_ax_moved, joy_ax_index, cab_index_viaSS, descript, dircor)
            
            
        tr_c = tr_c + 1
    
    
    return cut_trial_standard_move, cut_trial_standard_dir