import numpy as np
from scipy.interpolate import interp1d




def linear_intercurrentpt_makeshortSIGlong_interp1d(shortSIG, longSIG):

	x = np.linspace(shortSIG[0], len(shortSIG), num=len(shortSIG), endpoint=True)
	y = shortSIG
	# print('x : ', x)
	
	
	# -------------
	kind = 'linear'
	# kind : Specifies the kind of interpolation as a string or as an integer specifying the order of the spline interpolator to use. The string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the previous or next value of the point; ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers (e.g. 0.5, 1.5) in that ‘nearest-up’ rounds up and ‘nearest’ rounds down. Default is ‘linear’.
	
	if kind == 'linear':
		f = interp1d(x, y)
	elif kind == 'cubic':
		f = interp1d(x, y, kind='cubic')
	# -------------
	
	xnew = np.linspace(shortSIG[0], len(shortSIG), num=len(longSIG), endpoint=True)
	# print('xnew : ', xnew)

	siglong = f(xnew)

	return siglong
