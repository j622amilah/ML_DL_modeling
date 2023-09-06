import numpy as np

# Importing the statistics module
from statistics import mode, mean, median, multimode
import scipy.stats


def detectUD_trials_cabindetect(outSIG, axis_out):

    # We know that LR and UD are almost the same during UD reinitialization, so to detect if a trial is 
    # UD from the data alone look for an almost zero difference between the two and where their actual 
    # values are NOT zero.

    tr_num = len(outSIG)
    
    # Way 1 : Detect which trials are UD
    UD_trials = []
    
    for i in range(tr_num):
        tr_len = len(outSIG[i])
        
        # Difference of first one third of data - want to detect initialization
        onethird_NEWend = int(tr_len/3)
        
        LR_tr = outSIG[i][0:onethird_NEWend, 0]
        FB_tr = outSIG[i][0:onethird_NEWend, 1]
        UD_tr = outSIG[i][0:onethird_NEWend, 2]
        
        # -----------------------------------
        # Plotting actual cabin per trial
        # -----------------------------------
        # fig = go.Figure()
        # config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        # dp = np.multiply(range(0, tr_len), 1)
        # 
        # fig.add_trace(go.Scatter(x=dp, y=LR_tr, name='LR_tr', line = dict(color='red', width=2, dash='dash'), showlegend=True))
        # fig.add_trace(go.Scatter(x=dp, y=FB_tr, name='FB_tr', line = dict(color='green', width=2, dash='dash'), showlegend=True))
        # fig.add_trace(go.Scatter(x=dp, y=UD_tr, name='UD_tr', line = dict(color='blue', width=2, dash='dash'), showlegend=True))
        # fig.update_layout(title='', xaxis_title='data points', yaxis_title='Cabin position')
        # fig.show(config=config)
        # -----------------------------------
        
        diff = sum(LR_tr - UD_tr)
        LRsum = abs(sum(LR_tr))
        FBsum = abs(sum(FB_tr))
        UDsum = abs(sum(UD_tr))

        # Ratio is better because it accounts for LR and UD not being zero
        LR_UDrat = LRsum/UDsum
        
        # Avoid entries that are empty, they will get cut in the next processing step
        LRmode = mode(np.round(abs(LR_tr)))
        FBmode = mode(np.round(abs(FB_tr)))
        UDmode = mode(np.round(abs(UD_tr)))
        # print('LRmode : ' + str(LRmode) + ', FBmode : ' + str(FBmode) + ', UDmode : ' + str(UDmode))
        
        # Way 2 : get baseline of reinitialization for UD, mode of signal from 1:onethird_NEWend, 
        # to find similar LR and UD baseline

        # UD_trials signature = LRmode and UDmode are either both 1 or (100 +/- 10)
        if ( LRmode == 1 or LRmode == 2 or LRmode >= 100) and ( UDmode == 1 or UDmode == 2 or UDmode >= 100):
            # Second check
            if (np.round(LR_UDrat) == 1):
                UD_trials = UD_trials + [i]

    # Updating the EXPERIMENTAL PARAMETER list: axis
    axis_out[UD_trials] = 3
    
    return UD_trials, axis_out