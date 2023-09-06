import numpy as np

# Plotting
import plotly.graph_objects as go

from subfunctions.detect_jumps_in_data import *
from subfunctions.make_a_properlist import *


def freq_from_sig_timecounting(sig, t, ts, dp_jump, plotORnot):
    
    sig0 = make_a_properlist(sig)

    sig1 = [sig0[i]-sig0[0] for i in range(len(sig0))]
    N = len(sig1)

    # Remove floating point values by rounding down
    sig = np.round(sig1, 1)

    binary_sig = make_a_properlist(np.sign(sig))
    binary_sig = np.array(binary_sig)
    

    if plotORnot == 1:
        # --------------------

        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})

        fig.add_trace(go.Scatter(x=t, y=binary_sig, name='binary_sig', line = dict(color='red', width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=t, y=sig0, name='sig0', line = dict(color='green', width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=t, y=sig1, name='shifted', line = dict(color='blue', width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=t, y=sig, name='shifted-rounded', line = dict(color='purple', width=2, dash='dash'), showlegend=True))

        fig.update_layout(title='signals', xaxis_title='time', yaxis_title='signal')
        fig.show(config=config)

        # --------------------
        
    ind_jumpPT = detect_jumps_in_data(binary_sig, dp_jump)
    ind_jumpPT = [int(x+1) for x in ind_jumpPT]
    # print('ind_jumpPT : ' + str(ind_jumpPT))

    bin_chpt = np.array(binary_sig[ind_jumpPT])
    # print('bin_chpt : ' + str(bin_chpt))

    if not bin_chpt.any():
        # bin_chpt is empty
        fc = 1/(ts*N)
    else:
        # bin_chpt is NOT empty
        
        period_ind = ind_jumpPT[len(bin_chpt)-1]  #initialize, if 1st bin_chpt value is never found
        
        flag = 0
        for idx, val in enumerate(bin_chpt):  # search for first part not equal to zero
            # print('val : ', val)
            # print('flag : ', flag)
            if val == bin_chpt[0]:  # need to see the first number twice
                if flag == 0:  # first pass
                    flag = flag + 1
                elif flag == 1: # 2nd pass
                    period_ind = ind_jumpPT[idx]
                    flag = flag + 1 # this makes flag=2, so it can never fall into the 2nd pass

        # print('period_ind : ' + str(period_ind))
        per = t[period_ind] 
        # print('per : ' + str(per))
        fc = 1/per
        # print('fc : ' + str(fc))
    
    return fc