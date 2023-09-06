# Created by Jamilah Foucher, Aout 25, 2021.

# Purpose:  Organized usage of n4sid.py, in order to test:
#   1) projection coefficent methods (hankel, markov, output resized, etc), 
#   2) weighting of tf (tf direct, projection coefficents), 
#   3) prediction evaluation metric (r-squared, absolute error, distributed error).

# Using the prediction evaluation metric, we test both discrete and continuous time usages instead of resampling the signals for one of the two constructs; oversampling past Nyquist frequency for continuous construction and exact sampling to Nyquist frequency for discrete construction.  

# Thus, we decide if: (1) discrete or continuous construction is best, and (2) which of the projection coefficent methods are best.  The model with the best metic with respect to the output is returnned.

import numpy as np

from scipy import signal

# Import math Library
import math

# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')

from equalized_signal_len import *

from n4sid_prediction.n4sid import *
from n4sid_prediction.plot_best_coefmeth import *




def n4sid_main(inputs_org, outputs_org, t, ts, normy0Rnot, metric_type, plot_ALL_predictions, plot_final_prediction, plot_coef, where2regulate_gain, gainORnot):
    
    if 'avoid_infinite_loopcount' in locals():
        avoid_infinite_loopcount = avoid_infinite_loopcount + 1
    else:
        avoid_infinite_loopcount = 0  # Initialize
    
    # Initialize outputs
    pred_output = []
    tf_pred = []
    metric_val_best = []
    coef_meth_best = []
    discon = []
    
    n = 2
    zero_thresh = 0.1 # height of output signal movement

    # --------------------
    # Preprocess the input and output such that the n4sid mapping is correct
    # --------------------

    # --------------------
    # Step 1: normalize input and output signals for the coefficient prediction
    if normy0Rnot == 0:
        # Do nothing to the input and output
        inputs_norm = inputs_org
        outputs_norm = outputs_org
    elif normy0Rnot == 1:
        # Individual input and output normalization
        inputs_norm = inputs_org/max(abs(inputs_org))
        outputs_norm = outputs_org/max(abs(outputs_org))
    elif normy0Rnot == 2:
        # Combined input-output normalization
        maxval = max([max(inputs_org), max(outputs_org)])
        minval = min([min(inputs_org), min(outputs_org)])
        outputs_norm = (outputs_org - minval)/(maxval - minval)
        inputs_norm = (inputs_org - minval)/(maxval - minval)
    # --------------------



    if abs(np.max(outputs_norm)) > zero_thresh and len(outputs_norm) > 10:
        
        # ----------------
        
        # fig = go.Figure()
        # config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        
        # fig.add_trace(go.Scatter(x=t, y=inputs_norm, name='inputs_norm', line = dict(color='red', width=2, dash='dash'), showlegend=True))
        # fig.add_trace(go.Scatter(x=t, y=outputs_norm, name='outputs_norm', line = dict(color='blue', width=2, dash='dash'), showlegend=True))

        # fig.update_layout(title='coefficients compared with input-outputs', xaxis_title='data points', yaxis_title='amplitude')
        # fig.show(config=config)
        
        
        # ----------------
        way_coeff = 0  # 0=hankel, 1=markov, 2=output
        meth = 'hankel'
        best_disORcon_h, tf_best_h, pred_output_hankel, metric_h = n4sid(inputs_org, inputs_norm, outputs_org, outputs_norm, t, n, ts, way_coeff, normy0Rnot, meth, metric_type, plot_ALL_predictions, plot_coef, where2regulate_gain, gainORnot)
        # ----------------
        
        # ----------------
        way_coeff = 1  # 0=hankel, 1=markov, 2=output
        meth = 'markov'
        plot_coef = 0
        best_disORcon_m, tf_best_m, pred_output_markov, metric_m = n4sid(inputs_org, inputs_norm, outputs_org, outputs_norm, t, n, ts, way_coeff, normy0Rnot, meth, metric_type, plot_ALL_predictions, plot_coef, where2regulate_gain, gainORnot)
        # ----------------
        
        # ----------------
        way_coeff = 2  # 0=hankel, 1=markov, 2=output
        meth = 'outputs'
        plot_coef = 0
        best_disORcon_o, tf_best_o, pred_output_outputs, metric_o = n4sid(inputs_org, inputs_norm, outputs_org, outputs_norm, t, n, ts, way_coeff, normy0Rnot, meth, metric_type, plot_ALL_predictions, plot_coef, where2regulate_gain, gainORnot)
        # ----------------
        
        pred_output_hankel = np.array(pred_output_hankel)
        pred_output_markov = np.array(pred_output_markov)
        pred_output_outputs = np.array(pred_output_outputs)
        
        # print('pred_output_hankel : ' + str(pred_output_hankel))
        # print('pred_output_markov : ' + str(pred_output_markov))
        # print('pred_output_outputs : ' + str(pred_output_outputs))
        
        # --------------
        
        # At this point, we have a model (either continuous or discrete) for each of the three coefficient methods. 
        
        # --------------
        
        # Compare the coefficient methods that returned a model : use metrics and compare which one is best!
        # metric_h, metric_m, metric_o will equal 1000 if no match was found
        if metric_h == 1000:
            if metric_m == 1000:
                if metric_o == 1000:
                    pred_output = np.zeros((len(outputs_org)))
                    tf_pred = 200
                    metric_val_best = 200
                    coef_meth_best = 200
                    discon = 200
                    
                    if avoid_infinite_loopcount == 1:
                        print('NO coefficient methods could NOT find a stable mapping from the input and output.  Re-try with the signals normalized.')
                        # This could become an infinite loop - need to put a counter
                        normy0Rnot = 1  # 0=do nothing to signals, 1=individual signal normalization, 2=combined signal normalization
                        pred_output, tf_pred, metric_val_best, coef_meth_best, discon = n4sid_main(inputs_org, outputs_org, t, ts, normy0Rnot, metric_type)
                    else:
                        print('NO coefficient methods could NOT find a stable mapping from the input and output.  EXIT.')
                else:
                    # use outputs only
                    metric_val_best = metric_o
                    coef_meth_best = 2
                    
                    # Can not print the input,output,pred_output with time, if they are not all the same length
                    t_org_temp, inputs_org_temp, outputs_org_temp, pred_output_temp1 = equalized_signal_len(t, inputs_org, outputs_org, pred_output_outputs)
                    
                    plot_best_coefmeth(t_org_temp, inputs_org_temp, outputs_org_temp, pred_output_temp1, coef_meth_best, metric_val_best, plot_final_prediction)
                    
                    pred_output = pred_output_temp1
                    
                    tf_choices = [tf_best_h, tf_best_m, tf_best_o]
                    tf_pred = tf_choices[coef_meth_best]
                    discon_choices = [best_disORcon_h, best_disORcon_m, best_disORcon_o]
                    discon = discon_choices[coef_meth_best]
            else:
                if metric_o == 1000:
                    # use markov only
                    metric_val_best = metric_m
                    coef_meth_best = 1
                    
                    # Can not print the input,output,pred_output with time, if they are not all the same length
                    t_org_temp, inputs_org_temp, outputs_org_temp, pred_output_temp1 = equalized_signal_len(t, inputs_org, outputs_org, pred_output_markov)
                    
                    pred_output = pred_output_temp1
                    
                    plot_best_coefmeth(t_org_temp, inputs_org_temp, outputs_org_temp, pred_output, coef_meth_best, metric_val_best, plot_final_prediction)
                    
                    tf_choices = [tf_best_h, tf_best_m, tf_best_o]
                    tf_pred = tf_choices[coef_meth_best]
                    discon_choices = [best_disORcon_h, best_disORcon_m, best_disORcon_o]
                    discon = discon_choices[coef_meth_best]
                else:
                    # compare markov and outputs
                    coef_meth_out, metric_val_best = choose_best_regression_metric(metric_m, metric_o, metric_type)
                    
                    # Can not print the input,output,pred_output with time, if they are not all the same length
                    t_org_temp, inputs_org_temp, outputs_org_temp, pred_output_temp1, pred_output_temp2 = equalized_signal_len(t, inputs_org, outputs_org, pred_output_markov, pred_output_outputs)
                    
                    if coef_meth_out == 0:
                        coef_meth_best = 1  # markov
                        pred_output = pred_output_temp1
                    elif coef_meth_out == 1:
                        coef_meth_best = 2  # outputs
                        pred_output = pred_output_temp2
                    
                    plot_best_coefmeth(t_org_temp, inputs_org_temp, outputs_org_temp, pred_output, coef_meth_best, metric_val_best, plot_final_prediction)
                    
                    tf_choices = [tf_best_h, tf_best_m, tf_best_o]
                    tf_pred = tf_choices[coef_meth_best]
                    discon_choices = [best_disORcon_h, best_disORcon_m, best_disORcon_o]
                    discon = discon_choices[coef_meth_best]
        else:
            if metric_m == 1000:
                if metric_o == 1000:
                    # use hankel only
                    metric_val_best = metric_h
                    coef_meth_best = 0
                    
                    # Can not print the input,output,pred_output with time, if they are not all the same length
                    t_org_temp, inputs_org_temp, outputs_org_temp, pred_output_temp1 = equalized_signal_len(t, inputs_org, outputs_org, pred_output_hankel)
                    
                    plot_best_coefmeth(t_org_temp, inputs_org_temp, outputs_org_temp, pred_output_temp1, coef_meth_best, metric_val_best, plot_final_prediction)
                    
                    pred_output = pred_output_temp1
                    
                    tf_choices = [tf_best_h, tf_best_m, tf_best_o]
                    tf_pred = tf_choices[coef_meth_best]
                    discon_choices = [best_disORcon_h, best_disORcon_m, best_disORcon_o]
                    discon = discon_choices[coef_meth_best]
                else:
                    # compare hankel and outputs
                    coef_meth_out, metric_val_best = choose_best_regression_metric(metric_h, metric_o, metric_type)
                    
                    # Can not print the input,output,pred_output with time, if they are not all the same length
                    t_org_temp, inputs_org_temp, outputs_org_temp, pred_output_temp1, pred_output_temp2 = equalized_signal_len(t, inputs_org, outputs_org, pred_output_hankel, pred_output_outputs)
                    
                    if coef_meth_out == 0:
                        coef_meth_best = 0  # hankel
                        pred_output = pred_output_temp1
                    elif coef_meth_out == 1:
                        coef_meth_best = 2  # outputs
                        pred_output = pred_output_temp2
                    
                    plot_best_coefmeth(t_org_temp, inputs_org_temp, outputs_org_temp, pred_output, coef_meth_best, metric_val_best, plot_final_prediction)
                    
                    tf_choices = [tf_best_h, tf_best_m, tf_best_o]
                    tf_pred = tf_choices[coef_meth_best]
                    discon_choices = [best_disORcon_h, best_disORcon_m, best_disORcon_o]
                    discon = discon_choices[coef_meth_best]
            else:
                if metric_o == 1000:
                    # use hankel and markov
                    coef_meth_out, metric_val_best = choose_best_regression_metric(metric_h, metric_m, metric_type)
                    
                    # Can not print the input,output,pred_output with time, if they are not all the same length
                    t_org_temp, inputs_org_temp, outputs_org_temp, pred_output_temp1, pred_output_temp2 = equalized_signal_len(t, inputs_org, outputs_org, pred_output_hankel, pred_output_markov)
                    
                    if coef_meth_out == 0:
                        coef_meth_best = 0  # hankel
                        pred_output = pred_output_temp1
                    elif coef_meth_out == 1:
                        coef_meth_best = 1  # markov
                        pred_output = pred_output_temp2
                    
                    plot_best_coefmeth(t_org_temp, inputs_org_temp, outputs_org_temp, pred_output, coef_meth_best, metric_val_best, plot_final_prediction)
                    
                    tf_choices = [tf_best_h, tf_best_m, tf_best_o]
                    tf_pred = tf_choices[coef_meth_best]
                    discon_choices = [best_disORcon_h, best_disORcon_m, best_disORcon_o]
                    discon = discon_choices[coef_meth_best]
                else:
                    # compare hankel, markov and outputs
                    methcho1, metric_1stcompare_out = choose_best_regression_metric(metric_h, metric_m, metric_type)
                    methcho2, metric_2ndcompare_out = choose_best_regression_metric(metric_1stcompare_out, metric_o, metric_type)
                    
                    # Can not print the input,output,pred_output with time, if they are not all the same length
                    t_org_temp, inputs_org_temp, outputs_org_temp, pred_output_temp1, pred_output_temp2, pred_output_temp3 = equalized_signal_len(t, inputs_org, outputs_org, pred_output_hankel, pred_output_markov, pred_output_outputs)
                    
                    if methcho2 == 1:
                        coef_meth_best = 2  # outputs
                        pred_output = pred_output_temp3
                        metric_val_best = metric_2ndcompare_out
                    elif methcho1 == 0:
                        coef_meth_best = 0  # hankel
                        pred_output = pred_output_temp1
                        metric_val_best = metric_1stcompare_out
                    elif methcho1 == 1:
                        coef_meth_best = 1  # markov
                        pred_output = pred_output_temp2
                        metric_val_best = metric_1stcompare_out
                        
                    plot_best_coefmeth(t_org_temp, inputs_org_temp, outputs_org_temp, pred_output, coef_meth_best, metric_val_best, plot_final_prediction)
                    
                    tf_choices = [tf_best_h, tf_best_m, tf_best_o]
                    tf_pred = tf_choices[coef_meth_best]
                    discon_choices = [best_disORcon_h, best_disORcon_m, best_disORcon_o]
                    discon = discon_choices[coef_meth_best]
    else:
        print('Output is less than zero_thresh=0.1 (probably zeros) OR there are less than 10 data points in the input and output...n4sid will not give a good prediction with these signals.')
        pred_output = np.zeros((len(outputs_org)))
        tf_pred = 300
        metric_val_best = 300
        coef_meth_best = 300
        discon = 300
    
    return pred_output, tf_pred, metric_val_best, coef_meth_best, discon