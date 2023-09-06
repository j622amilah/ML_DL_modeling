# Created by Jamilah Foucher, Mai 30, 2021 

# Purpose: To calculate the coefficient of determination, adjusted coefficient of determination, 

# R-squared is useful, because it tells you if you should just use the mean or y (model a straight line) to get a better overall fit than yhat.  It is good for regression of data, but not necessarily good for measuring the difference between two lines.
    
# But R-squared is less useful if you want to capture the dynamics of y, like in control theory.  Using absolute error allows you to know if you are capturing the dynamics of the curve y, better than the global trend of y.
	
# Input VARIABLES:
# (1) y
# (2) yhat
# (3) num_of_predictors is the number of features

# Output VARIABLES:
# (1) r_squared
# (2) adj_r_squared
# (3) abs_error
# (4) avg_d_error


import numpy as np

# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')

from equalized_signal_len import *
from freq_from_sig_freqresp import *
from freq_from_sig_timecounting import *
from make_a_properlist import *



def rsquared_abserror(y, yhat, t, ts, num_of_predictors):
    
    # Check if y and yhat are vectors of the same length
    if np.isscalar(y) == True or not np.array(y).any() or np.isnan(y).any() == True:
        return 200, 200, 200, 200   # a value to indicate that the input vectors were not correct
    else:
        if np.isscalar(yhat) == True or not np.array(yhat).any() or np.isnan(yhat).any() == True:
            return 200, 200, 200, 200   # a value to indicate that the input vectors were not correct
        else:
            y, yhat = equalized_signal_len(y, yhat)
            N = len(y)   # total sample size

            y = np.reshape(y, (N,1))
            yhat = np.reshape(yhat, (N,1))

            SS_Residual = np.sum((y-yhat)**2)
            SS_Total = np.sum((y-np.mean(y))**2)

            # Relative measusre in terms of changes with respect to the mean : It is measuring the similar variations of the input and prediction with respect to the mean of the input
            # It does NOT measure absolute error between the input and prediction  

            # R-squared error or coefficient of determination.

            r_squared = 1 - (SS_Residual/SS_Total)

            # --------------------

            # Adjusted R-squared error.

            adj_r_squared = 1 - ((1-r_squared)*(N-1)/(N-num_of_predictors-1))

            # --------------------

            # Absolute error between the normalized input and prediction signal.

            # Normalize the signals with respect to both signals
            val = max([max(abs(y)), max(abs(yhat))])
            y_norm = y/val
            yhat_norm = yhat/val

            # Subtract the signals, sum across points, and divide by number of points, get a range of [-1, 1]
            y_error = (abs(y_norm) - abs(yhat_norm))/N  # abs is needed so that negative error does not cancel out positive error

            abs_error = np.sum(y_error)

            # --------------------


            # Calculate natural frequency of y, to know the optimal bin size for minimizing error
            # fc = freq_from_sig_freqresp(y, t, ts, 0)
            # OR
            dp_jump = 1
            fc = freq_from_sig_timecounting(y, t, ts, dp_jump)
            # print('calculated (fc) : ' + str(fc))

            # Determine if there is a reasonable fc
            if fc == 1/(ts*N):
                # Can not perform distributed_error because there is no period, the signal is 0.
                # So, return absolute error.
                avg_d_error = abs_error
            else:
                time_per_period = 1/fc

                num_of_dp_per_period = int(np.floor(time_per_period/ts))  

                win = num_of_dp_per_period  # length of each bin window
                # print('win : ' + str(win))

                # --------------------

                # Distributed/binned error between the normalized input and prediction signal.

                num_of_dp = len(y_error)
                start_val = 0
                stop_val = num_of_dp

                out = np.floor(range(start_val, stop_val, win))
                shift = 0
                if shift == 0:
                    # end point of previous window is one less than the start point of the next window
                    bin_tot = int(np.floor((len(out)*win)/(1+win)))
                elif shift == 1:
                    # end point of previous window is the same start point of next window 
                    if len(out)*win > stop_val:
                        bin_tot = len(out)-1
                    else:
                        bin_tot = len(out)
                # print('bin_tot : ' + str(bin_tot))

                # --------------------

                distributed_error = np.zeros((bin_tot))
                for bins in range(bin_tot):
                    st = bins + (bins*(win-shift))
                    stopper = st+win
                    distributed_error[bins] =  np.sum(y_error[st:stopper])
                # print('distributed_error : ' + str(distributed_error))

                # --------------------

                # Minimize error across every j bins
                start = 0
                j = 2
                ind = make_a_properlist(range(start, bin_tot, j))
                # print('ind : ' + str(ind))    

                d_sampled = []
                for i in ind:
                    d_sampled.append(distributed_error[i])
                # print('d_sampled : ' + str(d_sampled)) 

                avg_d_error = np.mean(d_sampled)

                # --------------------

            return r_squared, adj_r_squared, abs_error, avg_d_error