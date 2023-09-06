import numpy as np

# Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from subfunctions.detect_sig_change_wrt_baseline import *

import os


def standarization_check_if_joy_moved(tr, outJOY, marg_joy):

    # ------------------------------
    # Check if joystick MOVED
    # ------------------------------
    joy_ax = np.zeros((3,1))
    joy_ax_dir = np.zeros((3,1))
    joy_ax_val = np.zeros((3,1))
    joy_ax_index = np.zeros((3,1))

    # Find when joystick was moved and in what direction
    for ax in range(3):  # Search for joystick movement on the three axes used in the experiment
        joy_ax[ax] = ax

        sig = outJOY[tr][:, ax]
        baseline = sig[0]

        dpOFsig_in_zone, indexOFsig_in_zone, dp_sign_not_in_zone, indexOFsig_not_in_zone = detect_sig_change_wrt_baseline(sig, baseline, marg_joy)

        plotORnot = 0
        if plotORnot == 1:
            # ------------------------------
            fig = go.Figure()
            config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
            t = np.multiply(range(0, len(sig)), 1)

            fig.add_trace(go.Scatter(x=t, y=sig, name='sig', line = dict(color='black', width=2, dash='solid'), showlegend=True))
            plotb = baseline*np.ones((len(sig)))
            fig.add_trace(go.Scatter(x=t, y=plotb))

            outval = np.array(indexOFsig_in_zone)
            fig.add_trace(go.Scatter(x=outval, y=dpOFsig_in_zone, name='pts inside of zone', mode='markers', marker=dict(color='green', size=10, line=dict(width=0)), showlegend=True))

            outval1 = np.array(indexOFsig_not_in_zone)
            fig.add_trace(go.Scatter(x=outval1, y=dp_sign_not_in_zone, name='pts outside of zone', mode='markers', marker=dict(color='magenta', size=10, line=dict(width=0)), showlegend=True))

            fig.update_layout(title='Joystick motion', xaxis_title='data points', yaxis_title='Joystick position')

            fig.show(config=config)
            # ------------------------------

        # print('dpOFsig_in_zone :' + str(dpOFsig_in_zone))
        # print('indexOFsig_in_zone :' + str(indexOFsig_in_zone))
        # print('dp_sign_not_in_zone :' + str(dp_sign_not_in_zone))
        # print('indexOFsig_not_in_zone :' + str(indexOFsig_not_in_zone))
        
        
        # print('indexOFsig_not_in_zone.any() : ' + str(indexOFsig_not_in_zone.any()))

        if not indexOFsig_not_in_zone.any():  # if indexOFsig_not_in_zone is empty = False
            # joystick is not moved outside of the zone
            joy_ax_dir[ax] = 0
            joy_ax_val[ax] = 0
            joy_ax_index[ax] = 0
        else:
            # Take the difference between first datapoint in zone and first datapoint outside of zone
            diff = dp_sign_not_in_zone[0] - dpOFsig_in_zone[0]
            joy_ax_dir[ax] = np.sign(diff)   # joystick movement direction: 1=left, -1=right, 0=not moved

            joy_ax_val[ax] = abs( sig[indexOFsig_not_in_zone[0]] )# magnitude of first value outside of the zone

            joy_ax_index[ax] = indexOFsig_not_in_zone[0]     # index of the first value outside of the zone

    # print('joy_ax_dir :' + str(joy_ax_dir))
    # print('joy_ax_val :' + str(joy_ax_val))
    # print('joy_ax_index :' + str(joy_ax_index))

    return joy_ax_dir, joy_ax_val, joy_ax_index