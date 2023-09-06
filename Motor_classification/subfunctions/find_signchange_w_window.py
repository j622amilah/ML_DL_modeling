import numpy as np

# Plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Data saving
import pickle

from subfunctions.standarization_check_if_joy_moved import *
from subfunctions.window_3axesDATA import *
from subfunctions.make_a_properlist import*



def find_signchange_w_window(tr, outJOY_filter, axis_out, plotORnot):

    win = 6
    sig_win_out, bin_tot, st_ind, end_ind = window_3axesDATA(tr, outJOY_filter, win)
    
    print('len(sig_win_out) : ' + str(len(sig_win_out)))
    
    tot_ax0 = []
    tot_ax1 = []
    tot_ax2 = []
    joyax0 = []
    joyax1 = []
    joyax2 = []
    for bins in range(bin_tot):
        
        marg_joy = 0.01
        # Put cut piece into function
        joy_ax_dir, joy_ax_val, joy_ax_index = standarization_check_if_joy_moved(bins, sig_win_out, marg_joy)
        # print('joy_ax_dir : ' + str(joy_ax_dir))
        # print('joy_ax_val : ' + str(joy_ax_val))
        # print('joy_ax_index : ' + str(joy_ax_index))
        
        # A temporal trace of joystick direction across trial
        tot_ax0 = tot_ax0 + [joy_ax_dir[0]*np.ones((win+1,1))]  # only saving RO/LR
        tot_ax1 = tot_ax1 + [joy_ax_dir[1]*np.ones((win+1,1))]  # only saving PI/FB
        tot_ax2 = tot_ax2 + [joy_ax_dir[2]*np.ones((win+1,1))]  # only saving YA/UD
        
        # A temporal trace of joystick axis used across trial : 
        # 0=joystick axis not used, 1=joystick axis used
        joyax0 = joyax0 + [np.abs(tot_ax0)]
        joyax1 = joyax1 + [np.abs(tot_ax1)]
        joyax2 = joyax2 + [np.abs(tot_ax2)]
        
        

    if axis_out[tr] == 0:
        win_joy_sign = make_a_properlist(tot_ax0)
        joyax0 = make_a_properlist(joyax0)
        win_joy_ax = joyax0
    elif axis_out[tr] == 1:
        win_joy_sign = make_a_properlist(tot_ax1)
        win_joy_ax = make_a_properlist(joyax1)
    elif axis_out[tr] == 2:
        win_joy_sign = make_a_properlist(tot_ax2)
        win_joy_ax = make_a_properlist(joyax2)
    
    
    
    if plotORnot == 1:
        # --------------------
        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})

        xxORG = list(range(len(tot_ax0)))
        
        fig = make_subplots(rows=3, cols=1)
        fig.add_trace(go.Scatter(x=xxORG, y=tot_ax0, name='tot_ax0', line = dict(color='red', width=2, dash='dash'), showlegend=True), row=1, col=1)

        fig.add_trace(go.Scatter(x=xxORG, y=tot_ax1, name='tot_ax1', line = dict(color='green', width=2, dash='dash'), showlegend=True), row=2, col=1)

        fig.add_trace(go.Scatter(x=xxORG, y=tot_ax2, name='tot_ax2', line = dict(color='blue', width=2, dash='dash'), showlegend=True), row=3, col=1)

        fig.update_layout(title='way2 : tr=%d' % (tr), xaxis_title='data points', yaxis_title='axis sign')
        fig.show(config=config)
        # --------------------
    
    return win_joy_sign, win_joy_ax