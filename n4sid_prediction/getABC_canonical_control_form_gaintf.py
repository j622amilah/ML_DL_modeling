# Created by Jamilah Foucher, Aout 25, 2021 

import numpy as np

# Plotting
import plotly.graph_objects as go

# Symbolic math Library
import sympy as sym
from sympy import Matrix, symbols


from scipy import signal

from scipy.signal import lsim
from scipy.signal import ss2tf
from scipy.signal import tf2ss

# Solve for eigenvalues of A
from numpy import linalg as LA

# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')

from make_a_properlist import *

# from pole_placement.pole_placement import *
from choose_best_regression_metric import *

# Calculate regression metrics
from sklearn.metrics import r2_score
from rsquared_abserror import *

from n4sid_prediction.plot_best_coefmeth import *


def getABC_canonical_control_form_gaintf(way_coeff, mapping_coefs, inputs_generic, outputs_generic, time, n, ts, metric_type, plot_ALL_predictions, gainORnot):
    
    inputs_generic = np.array(inputs_generic)
    outputs_generic = np.array(outputs_generic)
    
	# ------------ DO NOT RUN ------------
	# FULL FORM : but the algebraic expressions are too long to solve for

	# a1, a2, a3, a4, c1, c2, b1, b2 = symbols('a1 a2 a3 a4 c1 c2 b1 b2')

	# A = Matrix([[a1, a2],[a3, a4]])
	# print('A', str(A))
	# print('size of A : ', str(A.shape))

	# B = Matrix([b1, b2])# creates a 2x1 matrix, but also Matrix([[c1], [c2]]) creates a 2x1 matrix
	# print('B', str(B))
	# print('size of B : ', str(B.shape))

	# C = Matrix([c1, c2])# 2x1 matrix, but also Matrix([[c1], [c2]]) creates a 2x1 matrix
	# print('C.T', str(C.T)) # change to a 1x2
	# print('size of C.T : ', str(C.T.shape))

	# # The eight matrix equations that we need in algebraic form are :
	# print('C.T*B : ', str(C.T*B))

	# print('C.T*A*B : ', str(C.T*A*B))

	# print('C.T*A*A*B : ', str(C.T*A*A*B))

	# print('C.T*A*A*A*B : ', str(C.T*A*A*A*B))

	# print('C.T*A*A*A*A*B : ', str(C.T*A*A*A*A*B))

	# print('C.T*A*A*A*A*A*B : ', str(C.T*A*A*A*A*A*B))

	# print('C.T*A*A*A*A*A*A*B : ', str(C.T*A*A*A*A*A*A*B))

	# print('C.T*A*A*A*A*A*A*A*B : ', str(C.T*A*A*A*A*A*A*A*B))

	# beta0, beta1, beta2, beta3, beta4, beta5, beta6, beta7 = sym.symbols('beta0 beta1 beta2 beta3 beta4 beta5 beta6 beta7')

	# T00, T01, T02, T03, T04, T05, T06, T07 = sym.symbols('T00 T01 T02 T03 T04 T05 T06 T07')

	# out = sym.solve(( b1*c1 + b2*c2 - T00, b1*(a1*c1 + a3*c2) + b2*(a2*c1 + a4*c2) - T01, b1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + b2*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)) - T02, b1*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + b2*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) - T03, b1*(a1*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a3*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))) + b2*(a2*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a4*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))) - T04, b1*(a1*(a1*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a3*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))) + a3*(a2*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a4*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))))) + b2*(a2*(a1*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a3*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))) + a4*(a2*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a4*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))))) - T05, b1*(a1*(a1*(a1*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a3*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))) + a3*(a2*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a4*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))))) + a3*(a2*(a1*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a3*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))) + a4*(a2*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a4*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))))) + b2*(a2*(a1*(a1*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a3*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))) + a3*(a2*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a4*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))))) + a4*(a2*(a1*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a3*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))) + a4*(a2*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a4*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))))) - T06, b1*(a1*(a1*(a1*(a1*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a3*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))) + a3*(a2*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a4*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))))) + a3*(a2*(a1*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a3*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))) + a4*(a2*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a4*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))))) + a3*(a2*(a1*(a1*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a3*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))) + a3*(a2*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a4*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))))) + a4*(a2*(a1*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a3*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))) + a4*(a2*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a4*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))))))) + b2*(a2*(a1*(a1*(a1*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a3*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))) + a3*(a2*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a4*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))))) + a3*(a2*(a1*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a3*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))) + a4*(a2*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a4*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))))) + a4*(a2*(a1*(a1*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a3*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))) + a3*(a2*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a4*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))))) + a4*(a2*(a1*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a3*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2)))) + a4*(a2*(a1*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a3*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))) + a4*(a2*(a1*(a1*c1 + a3*c2) + a3*(a2*c1 + a4*c2)) + a4*(a2*(a1*c1 + a3*c2) + a4*(a2*c1 + a4*c2))))))) - T07 ), (a1, a2, a3, a4, c1, c2, b1, b2))
	# print('out: ' + str(out))

	# Full form is not solveable...
	# ------------ DO NOT RUN ------------
    
    
    # ------------------------
    
    N = len(inputs_generic)
    
    if n == 2:
        # Simplification of full form, by assuming the form is canonical controllable form :
        a1 = 0
        a2 = 1

        c1 = 1
        c2 = 0

        a3, a4, b1, b2 = symbols('a3 a4 b1 b2')

        A = Matrix([[a1, a2],[a3, a4]])
        # print('A', str(A))
        # print('size of A : ', str(A.shape))

        B = Matrix([b1, b2]) # 2x1 matrix
        # print('B', str(B))
        # print('size of B : ', str(B.shape))

        C = Matrix([c1, c2])# 2x1 matrix, but also Matrix([[c1], [c2]]) creates a 2x1 matrix
        # print('C.T', str(C.T)) # change to a 1x2
        # print('size of C.T : ', str(C.T.shape))

        # Only four matrix equations are needed in algebraic form :
        # print('C.T*B : ', str(C.T*B))

        # print('C.T*A*B : ', str(C.T*A*B))

        # print('C.T*A*A*B : ', str(C.T*A*A*B))

        # print('C.T*A*A*A*B : ', str(C.T*A*A*A*B))

        # C.T*B :  Matrix([[b1]])
        # C.T*A*B :  Matrix([[b2]])
        # C.T*A*A*B :  Matrix([[a3*b1 + a4*b2]])
        # C.T*A*A*A*B :  Matrix([[a3*a4*b1 + b2*(a3 + a4**2)]])

        # --------------------

        T00, T01, T02, T03 = sym.symbols('T00 T01 T02 T03')

        out = sym.solve(( b1 - T00, b2 - T01, a3*b1 + a4*b2 - T02, a3*a4*b1 + b2*(a3 + a4**2) - T03 ), (b1, b2, a3, a4))
        # print('out: ' + str(out))
        
        # out: [(T00, 
        # T01, 
        # -(T01*T03 - T02**2)/(T00*T02 - T01**2), 
        # (T00*T03 - T01*T02)/(T00*T02 - T01**2)
        # )]
        # --------------------
        
        # --------------------
        # Way 2 : start using coefficient values when they are not zero
        flag = 0  # flag is redundant : just to ensure that the break works
        for i in range(len(mapping_coefs)):
            if mapping_coefs[i] != 0 and flag == 0:
                ist = i
                flag = 1
                break
        # print('ist : ' + str(ist))
        
        if way_coeff == 0:
            # Get coefficients from Hankel matrix
            T00 = mapping_coefs[ist]
            T01 = mapping_coefs[ist+1]
            T02 = mapping_coefs[ist+2]
            T03 = mapping_coefs[ist+3]
        elif way_coeff == 1:
            # Get coefficients from Markov parameters
            T00 = mapping_coefs[ist+1]
            T01 = mapping_coefs[ist+2]
            T02 = mapping_coefs[ist+3]
            T03 = mapping_coefs[ist+4]
        elif way_coeff == 2:
            # output : Testing the output because the markov parameters and dialation strongly resemble the output.  Can we just use the output scaled in some situations?
            T00 = mapping_coefs[ist+1]
            T01 = mapping_coefs[ist+2]
            T02 = mapping_coefs[ist+3]
            T03 = mapping_coefs[ist+4]
        # --------------------
        
        # print('T00: ' + str(T00))
        # print('T01: ' + str(T01))
        # print('T02: ' + str(T02))
        # print('T03: ' + str(T03))
        
        # --------------------
        
        # Plug in numbers for the unknown coefficients
        b1 = T00
        b2 = T01
        a3 = -(T01*T03 - T02**2)/(T00*T02 - T01**2)
        a4 = (T00*T03 - T01*T02)/(T00*T02 - T01**2)
        
        # print('b1: ' + str(b1))
        # print('b2: ' + str(b2))
        # print('a3: ' + str(a3))
        # print('a4: ' + str(a4))

        # --------------------

        # Compose the A, B, C, D matricies
        
        # Exception : if A contains 'nan', use identity matrix to give a bad fit, but keep the algorithm running.
        # A good fit will eventually be found or not.
        A = np.array([[a1, a2],[a3, a4]])
        if np.isnan(A).any() == True:
            A = np.array([[a1, a2],[0.01, 0.05]])   # some values that the algorithm can compute
            
        B = np.reshape([b1, b2], (2,1))
        C = np.reshape([[c1],[c2]], (1,2))
        D = np.zeros((1,1))

        # print('A : ', str(A))
        # print('B : ', str(B))
        # print('C : ', str(C))
        # print('D : ', str(D))

        # print('size of A : ', str(A.shape))
        # print('size of B : ', str(B.shape))
        # print('size of C : ', str(C.shape))
        # print('size of D : ', str(D.shape))

        # --------------------

        eigenvalues_of_A, v = LA.eig(np.array(A))
        # print('eigenvalues_of_A : ' + str(eigenvalues_of_A))

        # --------------------

        (num, den) = ss2tf(A, B, C, D)

        num = np.array(make_a_properlist(num))
        num = np.round(num, 5)
        # print('num : ' + str(num))
        
        den = np.array(make_a_properlist(den))
        den = np.round(den, 5)
        # print('den : ' + str(den))

        # --------------------
        
        # Technically, all signals are discrete.  But, you can "get away" with using a continuous time
        # formulation if the system is over-sampled.  So both discrete and continuous time calculations are performed, just to understand if the signal is over-sampled or not.
        
        # --------------------
        
        if gainORnot == 0:
            test_loop = 1
            kout = [1]
        elif gainORnot == 1:
            # Generate a list of gains to multiply with the numerator : to try to get a better fit
            test_loop = 10
            endK_tune = 0.001
            
            if max([max(np.abs(inputs_generic)), max(np.abs(outputs_generic))]) > max(np.abs(mapping_coefs)):
                k = max([max(np.abs(inputs_generic)), max(np.abs(outputs_generic))])/max(np.abs(mapping_coefs))
            else:
                k = max(np.abs(mapping_coefs))/max([max(np.abs(inputs_generic)), max(np.abs(outputs_generic))])
            
            Kincrement = (k-endK_tune)/test_loop
            
            # Generate positive gain values from k to endK_tune
            kk = [k]  # initial value of the k vector
            for i in range(test_loop):
                kk = kk + [kk[i-1]-Kincrement]
            kk = np.ravel(kk)
            # print('kk : ' + str(kk))

            # Generate equivalent positive and negative gain values
            kout = np.concatenate((kk, -kk), axis=0)
            kout = np.array(kout)
        
        # --------------------
        
        # Discrete system :
        metric_dis_best = 1000
        tf_dis_best = []
        pred_output_dis_best = []
        for i in range(len(kout)):
            tf_dis = (kout[i]*num, den, ts)
            # print('tf_dis : ' + str(tf_dis))
            
            pred_t_dis, pred_output_dis  = signal.dlsim(tf_dis, inputs_generic, t=time)
            
            # Make sure signal in a list
            pred_output_dis = make_a_properlist(pred_output_dis)
            pred_t_dis = make_a_properlist(pred_t_dis)
            
            # Make sure it is an array so we can cut the data in the next step
            pred_output_dis = np.array(pred_output_dis)
            pred_output_dis = pred_output_dis.real
            
            # Ensure predicted signal and outputs are the same length : sometimes pred_output has one point less than outputs
            minlen = min([len(pred_output_dis), len(outputs_generic)])
            pred_output_dis = np.ravel(np.reshape(pred_output_dis[0:minlen], (1, minlen)))
            outputs_generic = np.ravel(np.reshape(outputs_generic[0:minlen], (1, minlen)))
            
            # r_squared_dis_sklearn = r2_score(outputs_generic, pred_output_dis)
            # OR
            r_squared_dis, adj_r_squared_dis, abs_error_dis, avg_d_error_dis = rsquared_abserror(outputs_generic, pred_output_dis, time, ts, 1)
            
            if metric_type == 0:
                metric_dis = r_squared_dis
                # print('r_squared_dis : ' + str(r_squared_dis))
                # print('r_squared_dis_sklearn : ' + str(r_squared_dis_sklearn))
            elif metric_type == 1:
                metric_dis = abs_error_dis
                # print('abs_error_dis : ' + str(abs_error_dis))
            elif metric_type == 2:
                metric_dis = avg_d_error_dis
                # print('avg_d_error_dis : ' + str(avg_d_error_dis))
            
            # Plotting
            plot_best_coefmeth(time, inputs_generic, outputs_generic, pred_output_dis, way_coeff, metric_dis, plot_ALL_predictions)
            
            # Update output parameters
            methcho1, metric_out = choose_best_regression_metric(metric_dis_best, metric_dis, metric_type)
            if methcho1 == 1:
                tf_dis_best = tf_dis
                pred_output_dis_best = pred_output_dis
                metric_dis_best = metric_dis
        
        # --------------------
        
        # Continuous system :
        # with pole_placement
        # tf_con = (num, den)
        # sys_type = 'cont'
        # den_pp = pole_placement(tf_con, sys_type)
        
        metric_con_best = 1000
        tf_con_best = []
        pred_output_con_best = []
        for i in range(len(kout)):
            # with pole_placement
            # tf_con = (kout[i]*num, den_pp.real)
            
            # without pole_placement
            tf_con = (kout[i]*num, den.real)
            
            # print('tf_con : ' + str(tf_con))
            
            pred_t_con, pred_output_con, xout_con = signal.lsim(tf_con, U=inputs_generic, T=time)
            
            # Make sure signal in a list
            pred_output_con = make_a_properlist(pred_output_con)
            pred_t_con = make_a_properlist(pred_t_con)
            
            # Make sure it is an array so we can cut the data in the next step
            pred_output_con = np.array(pred_output_con)
            pred_output_con = pred_output_con.real
            
            # Ensure predicted signal and outputs are the same length : sometimes pred_output has one point less than outputs
            minlen = min([len(pred_output_con), len(outputs_generic)])
            pred_output_con = np.ravel(np.reshape(pred_output_con[0:minlen], (1, minlen)))
            outputs_generic = np.ravel(np.reshape(outputs_generic[0:minlen], (1, minlen)))
            
            # r_squared_con_sklearn = r2_score(outputs_generic, pred_output_con)
            # OR
            r_squared_con, adj_r_squared_con, abs_error_con, avg_d_error_con = rsquared_abserror(outputs_generic, pred_output_con, time, ts, 1)
            
            if metric_type == 0:
                metric_con = r_squared_con
                # print('r_squared_con : ' + str(r_squared_con))
                # print('r_squared_con_sklearn : ' + str(r_squared_con_sklearn))
            elif metric_type == 1:
                metric_con = abs_error_con
                # print('abs_error_con : ' + str(abs_error_con))
            elif metric_type == 2:
                metric_con = avg_d_error_con
                # print('avg_d_error_con : ' + str(avg_d_error_con))
            
            # Plotting
            plot_best_coefmeth(time, inputs_generic, outputs_generic, pred_output_con, way_coeff, metric_con, plot_ALL_predictions)
            
            # Update output parameters
            methcho1, metric_out = choose_best_regression_metric(metric_con_best, metric_con, metric_type)
            # print('methcho1 : ' + str(methcho1))
            # print('metric_out : ' + str(metric_out))
            
            if methcho1 == 1:
                tf_con_best = tf_con
                pred_output_con_best = pred_output_con
                metric_con_best = metric_out
        # --------------------
    else:
        tf_dis_best = []
        pred_output_dis_best = []
        tf_con_best = []
        pred_output_con_best = []
        print('This only works for 2nd order systems at the moment...put n=2 and rerun')
        
        
    return tf_dis_best, pred_output_dis_best, tf_con_best, pred_output_con_best, metric_dis_best, metric_con_best