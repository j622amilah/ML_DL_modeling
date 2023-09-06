# Created by Jamilah Foucher, Aout 25, 2021.

# Purpose:  A simple version of n4sid, a projection mapping of the input signal to an output signal.  Given an input and corresponding output signal, a state-space model is created to predict the output given a similar input. 

# This type of model is often used in black-box modeling of human-in-the-loop scenarios, where the human is the state-space model that produces an output given some input.  This could be used to test 

# We break-down the problem into two steps : 1) Calculate the coefficients to 'transfer' each input value to the output value (this is the Hankel matrix, Markov parameters or transfer function matrix power series.  Simply, it is the multiplicative value that dialates the input point to the output point.), 2) Put the coefficients into a state-space/transfer function form (already proved canonical controllable form is used, a numerical method to transform power series to a transfer function needs more work).

# Input VARIABLES:
# (1) inputs_org
# (2) inputs_norm
# (3) outputs_org is the output vector
# (4) outputs_norm
# (5) time is the time vector for the collected inputs and outputs
# (6) n is the model order
# (7) ts
# (8) way_coeff
# (9) normy0Rnot
# (10) meth
# (11) metric_type
# (12) plot_ALL_predictions

# Output VARIABLES:
# (1) best_disORcon
# (2) tf_best
# (3) pred_output_best
# (4) metric_best
# 


# ----------------
# Example
# ----------------
# fs = 10  # sampling frequency
# ts = 1/fs  # sampling time

# start_val = 0  # secs
# stop_val = 10  # secs

# N = int(fs*stop_val)  # number of sample points
# time = np.multiply(range(start_val, N), ts)

# wave_num1 = 4  # the smaller the value the greater the num_of_peaks
# period = (2*math.pi)/wave_num1   # the period or wave number
# natural_freq_in = 1/period
# angle = period*time
# amp = 0.5
# inputs = amp * np.sin(angle)
# inputs = make_a_properlist(inputs)

# wave_num2 = 10  # the smaller the value the greater the num_of_peaks
# period = (2*math.pi)/wave_num2   # the period or wave number
# natural_freq_out = 1/period
# angle = period*time
# amp = 0.4
# outputs = amp * np.sin(angle)
# outputs = make_a_properlist(outputs)
# print('length of outputs : ' + str(len(outputs)))

# --------------------

# inputs0 = np.array(inputs)
# outputs0 = np.array(outputs)

# normy0Rnot = 0 # 0=do nothing to signals, 1=individual signal normalization, 2=combined signal normalization

# metric_type = 0   # 0=rsquared, 1=abs_error
# #rsquared returns a more stable result (reflecting the mean) - one might think that the 
# #gain needs to be increased on the transfer function but it , abs_error returns a result
# #reflecting the dynamics of the output 

# plot_final_prediction = 1
# plot_ALL_predictions = 0

# #version 4: Numerator of transfer function regulated with gain
# pred_output, tf_pred, metric_best, sysmeth_best, discon = n4sid_main_ver4(inputs0, outputs0, time, ts, normy0Rnot, metric_type, plot_ALL_predictions, plot_final_prediction)

# version 3: Coefficients regulated with gain
# #pred_output, tf_pred, r_squared_best, sysmeth_best, discon = n4sid_main2(inputs0, outputs0, time, ts, normy0Rnot)

# print('metric_best : ' + str(metric_best))
# print('sysmeth_best : ' + str(sysmeth_best))
# --------------------


import numpy as np

# Solve for eigenvalues of A
from numpy import linalg as LA

# importing "cmath" for complex number operations
import cmath

# Plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from scipy import signal

from scipy.signal import lsim
from scipy.signal import ss2tf
from scipy.signal import tf2ss

# Least squares 
from sklearn.linear_model import LinearRegression

# Markov parameters
import control

# Import math Library
import math

# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')

from make_a_properlist import *

from choose_best_regression_metric import *

from n4sid_prediction.getABC_canonical_control_form_gaintf import *
from n4sid_prediction.getABC_canonical_control_form_gaincoef import *
from n4sid_prediction.hankel_construction import *




def n4sid(inputs_org, inputs_norm, outputs_org, outputs_norm, time, n, ts, way_coeff, normy0Rnot, meth, metric_type, plot_ALL_predictions, plot_coef, where2regulate_gain, gainORnot):

    N = len(inputs_org)
    
    # print('N : ' + str(N))
    # print('inputs_norm : ' + str(inputs_norm))
    # print('outputs_norm : ' + str(outputs_norm))
    
    # --------------------
    # [Step 1] Transfer function matrix : calculate the coefficients to 'transfer' each input value to the output value (this is the Hankel matrix.  There are different ways that you can map the input to the output signal, below 2 different ways:
    # --------------------
    
    # --------------------
    # 0) Hankel matrix : transfer function coefficients (Not Correct)
    T_1strow = hankel_construction(inputs_norm, outputs_norm)
    # print('T_1strow[0:10] : ' + str(T_1strow[0:10]))
    # print('T_1strow : ' + str(T_1strow))
    # --------------------

    # --------------------
    # 1) Markov parameters : Compute the Hankel matrix parameters using a tool to confirm result 
    # H is the first row of the Hankel matrix, and they are the Markov parameters
    m = N  # the number of markov parameters
    H = control.markov(outputs_norm, inputs_norm, m)
    H = make_a_properlist(H)
    # print('H[0:10] : ' + str(H[0:10]))
    # print('H : ' + str(H))
    # --------------------
    
    # --------------------
    # 2) output resembled the Markov parameters for sinusoidal inputs and outputs
    # --------------------
    # Sometimes T_1strow and H can explode at the end
    rr = int(N/2)
    avgheight_1sthalf = np.mean( [max(T_1strow[0:rr]), max(H[0:rr])] )
    
    if abs(avgheight_1sthalf) < 0.1:
        # case when T_1strow and H = zero
        # Make the height 1/10th of the original height, very small but not zero
        avgheight = max(abs(outputs_org))/10
    else:
        # Wanted to say if the first half is similar to all, then use all, if not use 1sthalf
        avgheight = avgheight_1sthalf
    
    # Goal : we want the shape of the output, with the height of the other coefficient methods
    outs = avgheight*outputs_norm
    # --------------------
    
    # --------------------
    # Comparing the different coefficient methods:
    # --------------------
    if plot_coef == 1:
        xxORG = list(range(len(H)))
        
        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        
        fig.add_trace(go.Scatter(x=xxORG, y=T_1strow, name='T_1strow', line = dict(color='red', width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=xxORG, y=H, name='H', line = dict(color='magenta', width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=xxORG, y=outs, name='outs', line = dict(color='cyan', width=2, dash='dash'), showlegend=True))

        fig.add_trace(go.Scatter(x=xxORG, y=np.array(inputs_org), name='inputs_org', line = dict(color='blue', width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=xxORG, y=np.array(outputs_org), name='outputs_org', line = dict(color='green', width=2, dash='dash'), showlegend=True))

        fig.update_layout(title='coefficients compared with input-outputs', xaxis_title='data points', yaxis_title='amplitude')
        fig.show(config=config)
    # ------------------------
    
    
    # ------------------------
    # [Step 2] : Put the coefficients into a state-space/transfer function form (already proved canonical controllable form is used, a numerical method to transform power series to a transfer function needs more work).
    # ------------------------
    
    # Put a coeff matrix that corresponds with way_coeff
    if way_coeff == 0:
        mapping_coefs = T_1strow
    elif way_coeff == 1:
        mapping_coefs = H
    elif way_coeff == 2:
        mapping_coefs = outs
        
    # --------------------
    
    
    # --------------------
    # TESTED : does not seem to make a difference
    # I would think that this changes the gain on the transfer function.
    which_signals = 0
    if which_signals == 0:  # 0=original height signals, 1=normalized signals
        # We test the models using the ORIGINAL height signals
        inputs2test = inputs_org
        outputs2test = outputs_org
    elif which_signals == 1:
        inputs2test = inputs_norm
        outputs2test = outputs_norm
    # --------------------
    
    
    # --------------------
    # Weighting of tf (tf direct, projection coefficents)
    if where2regulate_gain == 0:
        tf_dis_best, pred_output_dis_best, tf_con_best, pred_output_con_best, metric_dis_best, metric_con_best = getABC_canonical_control_form_gaintf(way_coeff, mapping_coefs, inputs2test, outputs2test, time, n, ts, metric_type, plot_ALL_predictions, gainORnot)
    elif where2regulate_gain == 1:
        tf_dis_best, pred_output_dis_best, tf_con_best, pred_output_con_best, metric_dis_best, metric_con_best = getABC_canonical_control_form_gaincoef(way_coeff, mapping_coefs, inputs2test, outputs2test, time, n, ts, metric_type, plot_ALL_predictions, gainORnot)
    # --------------------
    
    
    # --------------------
    
    # Determine if discrete or continuous is best
    best_disORcon, metric_out = choose_best_regression_metric(metric_dis_best, metric_con_best, metric_type)
    
    if best_disORcon == 0:
        # Discrete is better
        tf_best = tf_dis_best
        pred_output_best = pred_output_dis_best
        metric_best = metric_dis_best
        # print('Discrete is best modeling type')
    elif best_disORcon == 1:
        # Continuous is better
        tf_best = tf_con_best
        pred_output_best = pred_output_con_best
        metric_best = metric_con_best
        # print('Continuous is best modeling type')
    
    # --------------------
    
    plot_final_coef_predition = 0
    
    if plot_final_coef_predition == 1:
        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})

        fig.add_trace(go.Scatter(x=time, y=inputs_org, name='inputs_org', line = dict(color='green', width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=time, y=outputs_org, name='outputs_org', line = dict(color='blue', width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=time, y=pred_output_best, name='predicted output', line = dict(color='red', width=2, dash='dash'), showlegend=True))
        
        fig.update_layout(title='%s : metric=%5.5f' % (meth, metric_best), xaxis_title='time', yaxis_title='signal')
        fig.show(config=config)
    
    # --------------------
    
    return best_disORcon, tf_best, pred_output_best, metric_best