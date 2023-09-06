# Created by Jamilah Foucher, Aout 17, 2021.

# Purpose : Frequency response of a signal, plotting the phase and magnitude of the signal.  We use the frequency response to create of a filter/transfer function that describes the dynamics of the signal.

# ----------------
# Example
# ----------------
# Run the notebook Example_frequency_response.ipynb, for an example of running the three subfunctions: (1) get_freqresp_mag_phase, (2) select_fc (select a carrier frequency), (3) select_filter.


import numpy as np

from scipy import signal
from scipy.fft import fft, ifft

# Plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Personal python functions
from subfunctions.make_a_properlist import *


# ----------------------------------------------
def get_freqresp_mag_phase(sig, t, ts):

    Xn = fft(sig)
    Xn_mag = abs(Xn.real)
    Xn_phase = np.angle(Xn)

    # Get the frequency vector
    # freq = np.fft.fftfreq(t.size, d=ts)

    # OR

    # Manually change magnitude and phase: for the fourier transform only 
    # half of the signal is unique and the latter half is repeated.  Typically
    # people mirror the latter half on the negative axis.  The function fftfreq 
    # does this automatically
    omeg = []
    for k in range(len(Xn_mag)):
        omeg = omeg + [(2*np.pi*k)/len(Xn_mag)]

    N = len(Xn_mag)
    
    # Put the frequency response in the correct order : fft gives the result on a circle but mapped to a line it give the negative part first and then the positive part
    
    Xn_mag_man = Xn_mag[int(np.floor((N/2))):N], Xn_mag[0:int(np.floor(N/2))]
    # combine an array and an array == want a single nested array or single array
    Xn_mag_man = make_a_properlist(Xn_mag_man)   # is unfortunately slow if you have a lot of data
    
    Xn_phase_man = Xn_phase[int(np.floor((N/2))):N], Xn_phase[0:int(np.floor(N/2))]
    # combine an array and an array == want a single nested array or single array
    Xn_phase_man = make_a_properlist(Xn_phase_man)  # is unfortunately slow if you have a lot of data
    
    omg_half = np.array(omeg[0:int(np.floor(N/2))])
    # combine a scalar, array, scalar == want a single nested array or single array
    omeg_man = -omg_half[::-1], omg_half, (2*np.pi*N/2)/len(Xn_mag)  
    omeg_man = make_a_properlist(omeg_man)  # is unfortunately slow if you have a lot of data
    
    
    # Only look at right half side : called spectrum
    Xn_mag_half_db = 20*np.log10(Xn_mag[0:int(np.floor(N/2))])
    Xn_phase_half = Xn_phase[0:int(np.floor(N/2))]
    
    # print('length of full values : ', len(Xn_mag_man), len(Xn_phase_man), len(omg_half), len(omeg_man), len(Xn_mag_half_db), len(Xn_phase_half))

    # To output the mirrored spectrum, output: Xn_mag_man, Xn_phase_man, omeg_man
    # To output one-side of the mirrored spectrum, output: Xn_mag_half_db, Xn_phase_half, omg_half
    return Xn_mag_man, Xn_phase_man, omeg_man, Xn_mag_half_db, Xn_phase_half, omg_half
# ---------------------------------------------- 



# ----------------------------------------------
def select_fc(choose_fc, sig, t, ts, plotORnot):
    # --------------------
    # To design the filter : choose your desired frequency (fc)
    # 
    # 1) Natural frequency of signal or signals
    # 2) minimal frequency needed to reconstruct the signal
    # --------------------
    
    if choose_fc == 0:
        fc = freq_from_sig_freqresp(sig, t, ts, plotORnot)
        
    elif choose_fc == 1:
        # [2] cutoff is Nyquist frequency - this gives the best fit because it is the 
        # largest desired frequency needed to reconstruct the original sampled signal, but it is 
        # not the natural frequency.  If you want a filter that characterizes the signal dynamics
        # you want fc = around the natural frequency.
        fs = 1/ts
        fc = fs/2 - 0.0001

    # elif choose_fc == 2:
        # To do 
        # fc = freq_from_2sigs_freqresp
        
    return fc
# ----------------------------------------------



# ----------------------------------------------
def select_filter(n, fc, fs):

    # Select a filter
    w = fc / (fs / 2) # Normalize the frequency
    # print('w : ' + str(w))
    # print('w for Nyquist (w needs to be smaller than this value): ' + str((fs/2 - 0.0001) / (fs / 2)))
    
    # Want to pass the high frequencies, and remove the low frequencies, because
    # they describe the dominante frequencies of the signal
    
    # For dB-drop, evaluating human movement: you want the high frequencies that people move at because
    # those are the dominant movements.  The low frequency movements are usually not interesting.
    b, a = signal.butter(n, w, 'high')  # nth order, b=numerator, a=denominator
    
    # For signal smoothing or recreation, use Nyquist cutoff and remove noise.
    # b, a = signal.butter(n, w, 'low')
    
    # Keep b, a notataion to remember filter notation
    num = b
    den = a
    
    # print('num : ' + str(num))
    # print('den : ' + str(den))

    # Discrete time model:
    # dt = 1/fc
    # tf = (num, den, dt)
    
    # Continuous time model:
    tf = (num, den)
    
    print('tf : ' + str(tf))
    
    return tf