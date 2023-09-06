import numpy as np

# Plotting
import plotly.graph_objects as go

# Personal python functions
from subfunctions.standarization_check_if_joy_moved import *
from subfunctions.make_a_properlist import *




def datadriven_FRT_vs_expmatFRT(num_of_tr, outJOY, time, FRT_em):
    
    # ------------------------------
    # Calculate FRT from the joystick data directly, and return the index
    # Find the AXIS the participant selected initially, then the DIRECTION from the joystick data
    # ------------------------------
    FRT_dd = []     # data-driven FRT
    FRT_ax = []
    FRT_dir = []
    diff_tot = []

    for tr in range(num_of_tr):
        # print('tr : ' + str(tr))
    
        # Center all joystick data at zero to remove biases
        for ax in range(3):
            baseline_shift = outJOY[tr][0,ax]
            outJOY[tr][:,ax] = outJOY[tr][:,ax] - baseline_shift
        
        # Find first joystick movement point
        marg_joy = 0.1
        joy_ax_dir, joy_ax_val, joy_ax_index = standarization_check_if_joy_moved(tr, outJOY, marg_joy)
        # print('joy_ax_dir : ' + str(joy_ax_dir))
        # print('joy_ax_val : ' + str(joy_ax_val))
        # print('joy_ax_index : ' + str(joy_ax_index))
        
        # print('np.sum(joy_ax_index) : ' + str(np.sum(joy_ax_index)))
        
        if int(np.sum(joy_ax_index)) == 0:
            # Joystick was not moved
            FRT_dd_tr = time[tr][len(time[tr])-1]     # Time is the last data point
            
            # -------------------------
            # fig = go.Figure()
            # config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
            # xxORG = list(range(len(time[tr])))
            # fig.add_trace(go.Scatter(x=xxORG, y=time[tr], name='time[tr]', line = dict(color='red', width=2, dash='dash'), showlegend=True))
            # fig.update_layout(title='time', xaxis_title='data points', yaxis_title='time')
            # fig.show(config=config)
            # -------------------------
            
            FRT_ax_pertr = 0
            FRT_dir_pertr = 0
            
            # print('len(time[tr]) : ' + str(len(time[tr])))
        else:
            # Joystick was moved
            # First remove the axes that people did not move - the zeros
            remaing_ax = []
            for ax in range(len(joy_ax_index)):
                if joy_ax_index[ax] != 0:
                    remaing_ax = remaing_ax + [ax]
                    
            ind_short = np.argmin(joy_ax_index[remaing_ax])
            
            FRT_ax_pertr = remaing_ax[ind_short]  # First axis choosen by participant
            # print('FRT_ax_pertr : ' + str(FRT_ax_pertr))
            
            # -------------------------
            FRT_ind = joy_ax_index[FRT_ax_pertr]  # the data point in the temporal series, in which the first axis was moved
            FRT_ind = int(FRT_ind[0])
            # print('FRT_ind : ' + str(FRT_ind))
            
            FRT_ax_val = joy_ax_val[FRT_ax_pertr]  # the joystick value of the first axis that was moved
            FRT_ax_val = FRT_ax_val[0]
            # print('FRT_ax_val : ' + str(FRT_ax_val))
            
            FRT_dir_pertr = joy_ax_dir[FRT_ax_pertr]  # First direction choosen by participant
            FRT_dir_pertr = int(FRT_dir_pertr[0])
            # print('FRT_dir_pertr : ' + str(FRT_dir_pertr))
            # -------------------------
            
            FRT_dd_tr = time[tr][FRT_ind]     # Time in with first axis was moved
            
        
        # print('FRT_dd_tr : ' + str(FRT_dd_tr))
        # print('FRT_em[tr][0] : ' + str(FRT_em[tr][0]))
        diff = FRT_em[tr][0] - FRT_dd_tr
        # print('diff : ' + str(diff))
        
        diff_tot = diff_tot + [diff]
        
        FRT_dd = FRT_dd + [FRT_dd_tr]
        FRT_ax = FRT_ax + [FRT_ax_pertr]
        FRT_dir = FRT_dir + [FRT_dir_pertr]
    # ------------------------------
    
    
    # print('diff_tot : ' + str(diff_tot))
    diff_tot = make_a_properlist(diff_tot)
    # -------------------------
    # fig = go.Figure()
    # config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
    # xxORG = list(range(len(diff_tot)))
    # fig.add_trace(go.Scatter(x=xxORG, y=diff_tot, name='diff_tot', line = dict(color='red', width=2, dash='dash'), showlegend=True))
    # fig.update_layout(title='Difference between FRT from experimental matrix and data-driven', xaxis_title='data points', yaxis_title='diff_tot')
    # fig.show(config=config)
    # -------------------------
    
    
    # print('FRT_dd : ' + str(FRT_dd))
    # -------------------------
    # fig = go.Figure()
    # config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
    # xxORG = list(range(len(FRT_dd)))
    # fig.add_trace(go.Scatter(x=xxORG, y=FRT_dd, name='FRT_dd', line = dict(color='red', width=2, dash='dash'), showlegend=True))
    # fig.update_layout(title='FRT data-driven', xaxis_title='data points', yaxis_title='FRT_dd')
    # fig.show(config=config)
    # -------------------------

    return FRT_dd, FRT_ax, FRT_dir