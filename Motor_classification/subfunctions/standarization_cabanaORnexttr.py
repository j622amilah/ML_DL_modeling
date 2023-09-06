import numpy as np

# Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Importing the statistics module
from statistics import mode, mean, median, multimode
import scipy.stats

from subfunctions.findall import *
from subfunctions.detect_sig_change_wrt_baseline import *
from subfunctions.standarization_plotting import *
from subfunctions.standarization_fill_in_matrix import *

import os

def standarization_cabanaORnexttr(joy_ax_index, joy_ax_dir, binary_marker, scalar_marker, direction_marker, dir_meaning_marker, tr_c, NOjoyMOVE, outSIG, outJOY, tr, normalized_outSIG, marg_joy, dirC, axis_out, varr, s, strictness_move, strictness_dir):
	
	# ------------------------------
    # Based on if joystick was moved : 1) continue analysis with cabin, 2) stop analysis --> Next trial
    # ------------------------------
    if sum(joy_ax_index) == 0:
        # Joystick is never moved :  2) stop analysis --> Next trial

        # ------------------------------
        # Save results
        # ------------------------------
        binary_marker[tr_c] = NOjoyMOVE
        scalar_marker[tr_c] = NOjoyMOVE
        direction_marker[tr_c] = NOjoyMOVE
        
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
        # Reducing the evaluation of each axes (3x1 vector) into a scalar value 
        # -------------------------------------------------
        # a] binary_marker : Did the cabin move after joystick movement? 1=yes,cab follow ok, 0=no
        if strictness_move == 0:
            # Lenient - Cases : 1) all 3 axes moved, 2) 2 axes moved, 3) 1 axis moved
            binary_marker[tr_c] = 0   # Initialization
            if len(ind_joy_ax_moved) == 3:
                # did at least 2 out of 3 axes have joy-cabin follow motion
                binary_marker[tr_c] = mode(bin_m)  
            else:
                # if 2 axes : did at least 1 out of 2 axes have joy-cabin follow motion
                # if 1 axis : did the 1 axis have joy-cabin follow motion
                for i in ind_joy_ax_moved:
                    if bin_m[i] == 1:
                        binary_marker[tr_c] = 1
                        
        elif strictness_move == 1:
            # Strict : if at least one axis of joy-cabin follow motion is wrong, put binary_marker = 0
            binary_marker[tr_c] = 1   # Initialization
            for i in ind_joy_ax_moved:
                if bin_m[i] == 0:
                    binary_marker[tr_c] = 0
        
        # ............................
        
        # c] direction_marker : Was the direction of joy movement with respect to cabin movement ok, 
        # according to the defined conventions (case 0 and 1)?: 1=correct, 0=not correct
        if strictness_dir == 0:
            # Lenient - Cases : 1) all 3 axes moved, 2) 2 axes moved, 3) 1 axis moved
            direction_marker[tr_c] = 0   # Initialization
            if len(ind_joy_ax_moved) == 3:
                # did at least 2 out of 3 axes have correct joy-cabin direction motion
                direction_marker[tr_c] = mode(d_m)  
            else:
                # if 2 axes : did at least 1 out of 2 axes have correct joy-cabin direction motion
                # if 1 axis : did the 1 axis have correct joy-cabin direction motion
                for i in ind_joy_ax_moved:
                    if d_m[i] == 1:
                        direction_marker[tr_c] = 1
        elif strictness_dir == 1:
            # Strict : if at least one axis of joy-cabin direction is wrong, put direction_marker = 0
            direction_marker[tr_c] = 1   # Initialization
            for i in ind_joy_ax_moved:
                if d_m[i] == 0:
                    direction_marker[tr_c] = 0
        
        # ............................
        
        # b] scalar_marker : What is the time difference between joystick and cabin movement?: t_delay 
        # is transformed into mean time delay of joystick-cabin follow 
        temp = []
        for i in ind_joy_ax_moved:
            temp = temp + [sca_ind[i]]
        scalar_marker[tr_c] = mean(temp)
        
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
        if not os.path.exists("%s\\images_standard" % (varr['main_path1'])):
            os.mkdir("%s\\images_standard" % (varr['main_path1']))
        # ------------------------------
        standarization_plotting(s, tr, normalized_outSIG, outJOY, axis_out, varr, ind_joy_ax_moved, joy_ax_index, cab_index_viaSS, descript, dircor)
    
            
    return binary_marker, scalar_marker, direction_marker, dir_meaning_marker