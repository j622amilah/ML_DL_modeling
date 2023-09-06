# Created by Jamilah Foucher, Septembre 1, 2021.

# Purpose : Find the natural frequency of a signal using the frequency response and phase response.
import numpy as np

from scipy import signal
from scipy.fft import fft, ifft

# Plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from subfunctions.make_a_properlist import *
from subfunctions.freqresp_functions import *



def freq_from_sig_freqresp(sig, t, ts, plotORnot):

    # You need the natural frequency of the output, then make the win the number of data points per period
    Xn_mag, Xn_phase, omeg, Xn_mag_half_db, Xn_phase_half, omg_half = get_freqresp_mag_phase(sig, t, ts)

    # --------------------

    # Find the frequency at which there is a significant magnitude and phase drop
    # This frequency is the natural frequency of the signal.
    # It could also be considered to be the 70dB drop frequency (or in other words all the frequencies above or below represent the frequency dynamics of the signal).
    max_ind = np.argmax(Xn_mag_half_db)
    # print('max_ind : ' + str(max_ind))

    cutoff_per = 0.3  # should be 0.3 for 30 percent decibel drop
    cut_mag = cutoff_per*Xn_mag_half_db[max_ind]
    # print('cut_mag : ' + str(cut_mag))

    # Search across the frequency magnitude to find the frequency cutoff point
    ind_out = np.NaN  # initialize 
    
    for i in range(max_ind, len(Xn_mag_half_db)):
        if Xn_mag_half_db[i] < cut_mag:
            ind_out = i
            break
    # print('ind_out : ' + str(ind_out))
    # print('magnitude at 0.3 dB drop : ' + str(Xn_mag_half_db[ind_out]))
    # print('frequency at 0.3 dB drop : ' + str(omg_half[ind_out]))
    
    # --------------------
    
    if np.isnan(ind_out):
        fc = np.NaN
    else:
        fc = omg_half[ind_out]  # Cut-off frequency; Needs to be less than fs/2
        
        if plotORnot == 1:
            # Plot a spectrum 
            fig = go.Figure()
            config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})

            fig = make_subplots(rows=2, cols=1)
            fig.append_trace(go.Scatter(x=omg_half, y=Xn_mag_half_db,), row=1, col=1)

            ind_out = ind_out*np.ones((2)) # must double the point : can not plot a singal point
            ind_out = [int(x) for x in ind_out] # convert to integer
            fig.append_trace(go.Scatter(x=omg_half[ind_out], y=Xn_mag_half_db[ind_out],), row=1, col=1)

            fig.append_trace(go.Scatter(x=omg_half, y=Xn_phase_half,), row=2, col=1)
            fig.update_layout(title='toy problem : amplitude and phase', xaxis_title='frequency', yaxis_title='mag (dB)')
            fig.show(config=config)

    # --------------------

    # You can choose this as fc, for filter design of a signal, to represent the dynamics of a signal.
    
    return fc