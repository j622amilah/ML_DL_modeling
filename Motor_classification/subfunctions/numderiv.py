import numpy as np


from subfunctions.make_a_properlist import *


def numderiv(x, t):
    # Created by Jamilah Foucher, Février 01, 2021

    # Purpose: Numerical derivative
    # 
    # Input VARIABLES:
    # (1) x is a vector in which you want the numerical derivative
    # 
    # (2) t is a time vector

    # Output VARIABLES:
    # (1) dx is a vector of the numerical derivative

    # -------------------------------

    if len(x) > 4:
        dx0 = np.ones(len(x))
        for i in range(len(x)-1):
	        dx0[i] = ( x[i+1] - x[i] ) / ( t[i+1] - t[i] )
        
        dx = dx0[0:len(dx0)-2], dx0[len(dx0)-2], dx0[len(dx0)-2] 
    else:
        # -------------------------------
        # Global linear regression using data spread, or slope, or derivative
        # -------------------------------

        # More precise for long x signal
        # x_mean = ( max(x) - min(x) ) / 2
        # dx = ( (x_mean - x[0] ) / ( t[-1] - t[0] )

        # OR

        # More precise for short x signal
        dx = ( x[-1] - x[0] ) / ( t[-1] - t[0] )
        #b_GS = x(1);

        #print('Global regression using data spread : [' + str(dx) + ', ' + str(b_GS) + ']') 
        #y_predict_GS = (dx*x + b_GS);  # linear estimation of x

        # See linear_regression.m for more precise methods of linear regression
        # -------------------------------
    
    dx = make_a_properlist(dx)
       
    return dx
