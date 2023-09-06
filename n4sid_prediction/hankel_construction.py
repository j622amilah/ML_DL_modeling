# Created by Jamilah Foucher, Aout 25, 2021.

# The Hankel matrix g(s) is the mapping of the input points onto the output points.

# y(s) = g(s)u(s)
# where g(s) = h1*s^-1 + h2*s^-2 + h3*s^-3 + ... + hN*s^-N

# The projection is not multiplying a previous input point by a scalar to get a future output point.

# The projection is dilating each input point by some scalar and summing them all up to get each output point.  

import numpy as np

# Least squares 
from sklearn.linear_model import LinearRegression

# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')

from make_a_properlist import *



def hankel_construction(inputs, outputs):
    
    N = len(inputs)
    
    inputs = np.array(inputs)

    # Logic for solving for Hankel matrix entries:
    # y = X*u is the least squares equation, the hankel matrix is X and is a matrix where the 
    # the upper-triangular part has unique entries.  For example, the first row contains all the unique entries, the second row contains all the unique entries except for the last value and the entries are shifted on space to the right.  This construction continues for the entire NxN matrix, until the Nth row of the matrix has the first unique entry.  
    
    # y = (X_basis*beta)*u
    # make u into the X_basis matrix by arranging u in each row of X_basis, shifting u to the right by one space for each increasing row.
    
    # y = X_basis*beta  which is the least squares formulation
    
    # Result : The beta values are NOT CORRECT because they do not the same as the Markov parameters or the dialation approach.  Perhaps the difference is caused by the fact that I only create the upper triangular section.
    
    # *** TO do later : create the lower triangular section. ***
    basis_mat = np.zeros((N,N))
    for r in range(N):
        # print('r : ' + str(r))
        for c in range(N):
            # print('c : ' + str(c))
            if c == 0:
                st = r
            else:
                st = st + 1
            if st < N:
                basis_mat[r,c] = inputs[st:st+1]    
                
    # print('basis_mat : ' + str(basis_mat))


    # Create linear regression object
    regr = LinearRegression()
    regr.fit(basis_mat, outputs)

    # This give the coefficients for the ratio of the : beta = B/A
    # beta is the first row of the Hankel matrix!
    beta = regr.coef_
    # print('beta : ' + str(beta))
    
    # To create Hankel matrix (T), arrange beta in the same manner as basis_mat
    # *** TO do later 
    # print('T : ' + str(T))
    
    # Test the rank of this matrix : in the construction to obtain the controllable canonical form of the A, B, C matricies, the observability (O) and controllability (C) matrix must be full rank.  And, thus the Hankel matrix (T) must be full rank because T=O*C.
    # num_of_rank = matrix_rank(T) 
    # print('num_of_rank : ' + str(num_of_rank))

    # first row of T is equivalent to the Markov parameters, which are the beta values!
    T_1strow = beta
    T_1strow = make_a_properlist(T_1strow)
    # print('T_1strow : ' + str(T_1strow))
    
    return T_1strow