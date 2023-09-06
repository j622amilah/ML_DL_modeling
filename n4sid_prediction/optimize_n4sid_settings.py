# Created by Jamilah Foucher, Aout 31, 2021.

# Can predict high frequency input to low frequency output better than it can predict
# low frequency input to high frequency output.

# Most human behavior is high frequency input to low frequency output, so this is OK for modeling 
# human behavior input-output data.

import numpy as np

from scipy import signal

# Data saving
import pickle

# Plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')

from make_a_properlist import *

from n4sid_prediction.n4sid_main import *
from rsquared_abserror import *
from equalized_signal_len import *



def optimize_n4sid_settings(inputs, outputs, time, ts):
    
    # metric_type_list : the projection coefficient method
    # 0=hankel
    # 1=markov
    # 2=output
    metric_type_list = [0, 1, 2]
    
    # where2regulate_gain_list : Weighting of tf (tf direct, projection coefficents)
    # 0=gain on tf numerator
    # 1=gain on projection coefficents
    where2regulate_gain_list = [1] # [0, 1]
    
    # gainORnot_list : gain value on the tf or coefficients
    # 0=gain equals 1 (no tunning of the original value)
    # 1=gain is varied (tunning of the original value)
    gainORnot_list = [0, 1]

    normy0Rnot = 0 # 0=do nothing to signals, 1=individual signal normalization, 2=combined signal normalization
    plot_final_prediction = 0  # 0=do not plot final prediction for each coef method, 1=plot final prediction for each coef method
    plot_ALL_predictions = 0 # 
    plot_coef = 0  # 0=do not plot coefficients, 1=plot coefficents

    pred_output_rs = []
    tf_pred_rs = []
    metric_rs = 1000
    sysmeth_rs = []
    discon_rs = []

    pred_output_ae = []
    tf_pred_ae = []
    metric_ae = 1000
    sysmeth_ae = []
    discon_ae = []

    pred_output_de = []
    tf_pred_de = []
    metric_de = 1000
    sysmeth_de = []
    discon_de = []

    for mt in metric_type_list:
        for w2rg in where2regulate_gain_list:
            for gOn in gainORnot_list:
                p_o, tf_p, m_b, s_b, dc = n4sid_main(inputs, outputs, time, ts, normy0Rnot, mt, plot_ALL_predictions, plot_final_prediction, plot_coef, w2rg, gOn)
                
                r_squared, adj_r_squared, abs_error, d_error = rsquared_abserror(outputs, p_o, time, ts, 1)
                
                methcho1, metric_out = choose_best_regression_metric(metric_rs, r_squared, 0)
                if methcho1 == 1:
                    pred_output_rs = p_o
                    tf_pred_rs = tf_p
                    metric_rs = r_squared
                    sysmeth_rs = s_b
                    discon_rs = dc
                    which_setting_rs = [mt, w2rg, gOn]
                
                methcho2, metric_out = choose_best_regression_metric(metric_ae, abs_error, 1)
                if methcho2 == 1:
                    pred_output_ae = p_o
                    tf_pred_ae = tf_p
                    metric_ae = abs_error
                    sysmeth_ae = s_b
                    discon_ae = dc
                    which_setting_ae = [mt, w2rg, gOn]
                
                methcho3, metric_out = choose_best_regression_metric(metric_de, d_error, 1)
                if methcho3 == 1:
                    pred_output_de = p_o
                    tf_pred_de = tf_p
                    metric_de = d_error
                    sysmeth_de = s_b
                    discon_de = dc
                    which_setting_de = [mt, w2rg, gOn]

    # --------------------

    # Plot the best of the 12 models for metric Rsquared
    # plot_best_coefmeth(time, inputs, outputs, pred_output_rs, sysmeth_rs, metric_rs, 1)
    # print('which_setting_rs : ' + str(which_setting_rs))

    # Plot the best of the 12 models for metric absolute error
    # plot_best_coefmeth(time, inputs, outputs, pred_output_ae, sysmeth_ae, metric_ae, 1)
    # print('which_setting_ae : ' + str(which_setting_ae))

    # Plot the best of the 12 models for metric distributed error
    # plot_best_coefmeth(time, inputs, outputs, pred_output_de, sysmeth_de, metric_de, 1)
    # print('which_setting_de : ' + str(which_setting_de))

    # --------------------

    # Great idea!  To compare the three, calculate r-squared for all three
    r_squared_rs, adj_r_squared, abs_error_rs, d_error = rsquared_abserror(outputs, pred_output_rs, time, ts, 1)
    r_squared_ae, adj_r_squared, abs_error_ae, d_error = rsquared_abserror(outputs, pred_output_ae, time, ts, 1)
    r_squared_de, adj_r_squared, abs_error_de, d_error = rsquared_abserror(outputs, pred_output_de, time, ts, 1)

    # print('Rsquared rs : ' + str(metric_rs))
    # print('Rsquared ae : ' + str(r_squared_ae))
    # print('Rsquared de : ' + str(r_squared_de))

    # print('Absolute error rs : ' + str(abs_error_rs))
    # print('Absolute error ae : ' + str(metric_ae))
    # print('Absolute error de : ' + str(abs_error_de))

    # --------------------
    
    if np.sum([r_squared_rs, r_squared_ae, r_squared_de]) == 0:
        # No model was found
        pred_output = 200
        tf_pred = 200
        r_squared = 200
        sysmeth = 200
        discon = 200
        which_setting = 200
    else:
        best_ind = np.argmax([r_squared_rs, r_squared_ae, r_squared_de])

        if best_ind == 0:
            pred_output = pred_output_rs
            tf_pred = tf_pred_rs
            r_squared = r_squared_rs
            sysmeth = sysmeth_rs
            discon = discon_rs
            which_setting = which_setting_rs
        elif best_ind == 1:
            pred_output = pred_output_ae
            tf_pred = tf_pred_ae
            r_squared = r_squared_ae
            sysmeth = sysmeth_ae
            discon = discon_ae
            which_setting = which_setting_ae
        elif best_ind == 2:
            pred_output = pred_output_de
            tf_pred = tf_pred_de
            r_squared = r_squared_de
            sysmeth = sysmeth_de
            discon = discon_de
            which_setting = which_setting_de

        plot_best_coefmeth(time, inputs, outputs, pred_output, sysmeth, r_squared, 1)

    return pred_output, tf_pred, r_squared, sysmeth, discon, which_setting