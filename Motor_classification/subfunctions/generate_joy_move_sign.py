import numpy as np

# Plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from subfunctions.find_signchange_w_window import *

# Creating folders for image or data saving
import os


 
def generate_joy_move_sign(tr, outJOY_filter, axis_out):

    num_of_dp = len(outJOY_filter[tr])
    
    # --------------------
    # Way 1 (old: less rigorous) : took the sign of the joystick position at each time step, 
    # assuming the participant would release the joystick and it would return to zero after 
    # each adjustment.  The joystick was velocity control so they need to release the joystick 
    # to make adjustments

    # Problem : (flawed logic) : the joystick movement on the stimuli axes will NOT always be 
    # greater than the other joystick axes.  So, you have to search each the slope of each axis. 

    # Instantaneous joystick movement : the sign of the joystick axis that moved the most
    # inst_joy_sign = []
    # for dp in range(num_of_dp):
        # ind_of_maxjoy_move = np.argmax([outJOY_filter[tr][dp,0], outJOY_filter[tr][dp,1], outJOY_filter[tr][dp,2]])
        # inst_joy_sign = inst_joy_sign + [np.sign(outJOY_filter[tr][dp,ind_of_maxjoy_move])]
    # --------------------


    # --------------------
    # Way 2  (more rigorous) : Take windowed steps across each trial, evaluate each window 
    # using detectchange_wrt_baseline to find the direction of movement per window.
    plotORnot = 0
    win_joy_sign, win_joy_ax = find_signchange_w_window(tr, outJOY_filter, axis_out, plotORnot)
    # --------------------

    plotORnot = 0
    if plotORnot == 1:
        # --------------------
        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})

        xxORG = list(range(len(win_joy_sign)))
        # fig.add_trace(go.Scatter(x=xxORG, y=inst_joy_sign, name='inst_joy_sign', line = dict(color='red', width=2, dash='solid'), showlegend=True))
        xxORG = list(range(len(win_joy_sign)))
        fig.add_trace(go.Scatter(x=xxORG, y=win_joy_sign, name='win_joy_sign', line = dict(color='blue', width=2, dash='solid'), showlegend=True))
        
        xxORG = list(range(len(outJOY_filter[tr])))
        if axis_out[tr] == 0:
            color_fillin0 = 'magenta'
            color_fillin1 = 'black'
            color_fillin2 = 'black'
        elif axis_out[tr] == 1:
            color_fillin0 = 'black'
            color_fillin1 = 'magenta'
            color_fillin2 = 'black'
        elif axis_out[tr] == 2:
            color_fillin0 = 'black'
            color_fillin1 = 'black'
            color_fillin2 = 'magenta'
        fig.add_trace(go.Scatter(x=xxORG, y=outJOY_filter[tr][:,0], name='outJOY_filter[tr][:,0]', line = dict(color=color_fillin0, width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=xxORG, y=outJOY_filter[tr][:,1], name='outJOY_filter[tr][:,1]', line = dict(color=color_fillin1, width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=xxORG, y=outJOY_filter[tr][:,2], name='outJOY_filter[tr][:,2]', line = dict(color=color_fillin2, width=2, dash='dash'), showlegend=True))
        
        fig.update_layout(title='tr : %d' % (tr), xaxis_title='data points', yaxis_title='joystick sign')
        fig.show(config=config)
        # --------------------

    return win_joy_sign, win_joy_ax