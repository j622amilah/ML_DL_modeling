# Created by Jamilah Foucher, Aout 30, 2021 

# Purpose:  To compare two regression metrics to determine if one regression method is better than another method. 

# Input VARIABLES:
# (1) metric_1 is the 1st metric value
# (2) metric_2 is the 2nd metric value
# (3) metric_type is the type of regression metric : 0=r_squared, 1=absolute error

# R-squared is useful, because it tells you if you should just use the mean or y (model a straight line) to get a better overall fit than yhat.  It is good for regression of data, but not necessarily good for measuring the difference between two lines.
    
# But R-squared is less useful if you want to capture the dynamics of y, like in control theory.  Using absolute error allows you to know if you are capturing the dynamics of the curve y, better than the global trend of y.
# 
# Output VARIABLES:
# (1) best_meth is the index number of the metric that was found to be best
# (2) metric_best is the metric value in which was found to be best



import numpy as np


def choose_best_regression_metric(metric_1, metric_2, metric_type):

    if metric_type == 0:
        # --------------------
        # Compare two R-squared values and find the R-squared value that is closer to 1.
        # --------------------
        
        # Initialize outputs
        r_squared_best = []

        metric_1 = np.array(metric_1)
        metric_1 = metric_1.item()

        metric_2 = np.array(metric_2)
        metric_2 = metric_2.item()

        # --------------------
        # Choose which rsquared is better : rsquared value closest to 1.
        # Find the distance of the r_squared value to 1 for positive and negative values.
        desired_val = 1
        # --------------------
        
    elif metric_type == 1 or metric_type == 2:
        # --------------------
        # Compare two normalized absolute error values and find the smallest value.
        # --------------------
        # Choose which absolute error is better : absolute error value closest to 0.
        # Find the distance of the absolute error value to 0 for positive and negative values.
        desired_val = 0
        # --------------------
        
        
    if metric_1 <= 0:
        diff_1_2one = desired_val - metric_1  # 1-neg_num
    else:
        diff_1_2one = abs(desired_val - metric_1) # 1-pos_num
    # print('diff_1_2one: ' + str(diff_1_2one))
    
    if metric_2 <= 0:
        diff_2_2one = desired_val - metric_2  # 1-neg_num
    else:
        diff_2_2one = abs(desired_val - metric_2)
    # print('diff_2_2one: ' + str(diff_2_2one))    
    
    # Find the smallest distance
    best_meth = np.argmin([diff_1_2one, diff_2_2one])
    
    
    if best_meth == 0:
        metric_best = metric_1
    elif best_meth == 1:
        metric_best = metric_2
        

    return best_meth, metric_best