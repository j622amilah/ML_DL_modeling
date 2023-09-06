# Created by Jamilah Foucher, Février 01, 2021

# Search along the signal with respect to a baseline and margin around baseline.  
# Return points along the signal inside the baseline zone and outside the baseline zone.

# Purpose: To find when the signal changes and in what direction.  Possible to detect the frequency of the
# signal, if you set the baseline to zero.
# 
# Input VARIABLES:
# (1) sig is a vector in which you would like to search
#  
# (2) baseline is the y-axis point of interest on the signal (sig) (where you wish to make margin around)
# 
# (3) marg is the distance from the baseline

# Output VARIABLES:
# (1) dpOFsig_in_zone is a vector of sig data points in the baseline +/- marg zone
# 
# (2) indexOFsig_in_zone is a vector of index points in the baseline +/- marg zone
# 
# (3) dp_sign_not_in_zone is a vector of sig data points NOT in the baseline +/- marg zone
# 
# (4) indexOFsig_not_in_zone is a vector of index points NOT in the baseline +/- marg zone


# ------------------------------------------
# Example
# ------------------------------------------
# start_val = 0
# stop_val = 9.9
# ts = 0.1
# f = 1/ts  # sampling frequency
# N = int(f*stop_val)
# t = np.multiply(range(start_val, N), ts) 

# sig = np.random.rand(N)*t
# sig_max = sig[np.argmax(sig)]
# baseline = sig_max/2
# marg = 0.1*sig_max    # 10 percent of the signal height

# plotb = baseline*np.ones((len(sig)))

# #https://plotly.com/python/line-and-scatter/
# #fig = go.Figure(data=go.Scatter(x=t, y=y, mode='markers'))
# # OR
# fig = go.Figure()
# config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
# fig.add_trace(go.Scatter(x=t, y=plotb))
# fig.add_trace(go.Scatter(x=t, y=sig))

# dpOFsig_in_zone, indexOFsig_in_zone, dp_sign_not_in_zone, indexOFsig_not_in_zone = detect_sig_change_wrt_baseline(sig, baseline, marg)

# outval = np.array(indexOFsig_in_zone)*ts
# fig.add_trace(go.Scatter(x=outval, y=dpOFsig_in_zone, name='pts inside of zone', mode='markers', marker=dict(color='green', size=10, line=dict(width=0)), showlegend=True))

# outval1 = np.array(indexOFsig_not_in_zone)*ts
# fig.add_trace(go.Scatter(x=outval1, y=dp_sign_not_in_zone, name='pts outside of zone', mode='markers', marker=dict(color='magenta', size=10, line=dict(width=0)), showlegend=True))

# fig.show(config=config)
#  ------------------------------------------

import numpy as np


def detect_sig_change_wrt_baseline(sig, baseline, marg):
    
    # Initialize outputs
    dpOFsig_in_zone = []
    indexOFsig_in_zone = []

    # Note : if the entire signal is in the zone, these remain empty
    dp_sign_not_in_zone = []
    indexOFsig_not_in_zone = []
    
    for q in range(len(sig)):
        if sig[q] > (baseline-marg):
            if sig[q] < (baseline+marg):
                indexOFsig_in_zone = indexOFsig_in_zone + [q]
                dpOFsig_in_zone = dpOFsig_in_zone + [sig[q]]
       
    # If a portion of the signal is in the zone (above), assign the portion of the signal that is NOT in the zone          
    longvec = range(len(sig))
    shortvec = indexOFsig_in_zone
    
    # Returns the unique values in 1st array NOT in 2nd array
    idx_from_long_not_in_short = np.setdiff1d(longvec, shortvec)
    #print('idx_from_long_not_in_short : ' + str(idx_from_long_not_in_short))

    indexOFsig_not_in_zone = idx_from_long_not_in_short
    sig = np.array(sig)
    dp_sign_not_in_zone = sig[indexOFsig_not_in_zone]
            
    
    return dpOFsig_in_zone, indexOFsig_in_zone, dp_sign_not_in_zone, indexOFsig_not_in_zone