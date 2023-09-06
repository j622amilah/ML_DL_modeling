import pywt
from pywt import wavedec
import matplotlib.pyplot as plt

import numpy as np
from scipy.interpolate import interp1d
from mlxtend.preprocessing import minmax_scaling

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pandas as pd

def scale_feature_data(feat, plotORnot):
    
    columns = ['0']
    dat = pd.DataFrame(data=feat, columns=columns)
    scaled_data0 = minmax_scaling(dat, columns=columns)
    scaled_data_mlx = list(scaled_data0.to_numpy().ravel())
    # OR 
    scaled_data_norma = []
    for q in range(len(feat)):
        scaled_data_norma.append( (feat[q] - np.min(feat))/(np.max(feat) - np.min(feat)) )  # normalization : same as mlxtend
    # OR 
    shift_up = [i - np.min(feat) for i in feat]
    scaled_data_posnorma = [q/np.max(shift_up) for q in shift_up]  # positive normalization : same as mlxtend
    # OR 
    scaled_data_standardization = [(q - np.mean(feat))/np.std(feat) for q in feat]  # standardization
    
    if plotORnot == 1:
        fig = make_subplots(rows=2, cols=1)
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        xxORG = list(range(len(feat)))
        fig.add_trace(go.Scatter(x=xxORG, y=feat, name='feat', line = dict(color='red', width=2, dash='solid'), showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=xxORG, y=scaled_data_mlx, name='scaled : mlxtend', line = dict(color='red', width=2, dash='solid'), showlegend=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=xxORG, y=scaled_data_norma, name='scaled : normalization', line = dict(color='cyan', width=2, dash='solid'), showlegend=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=xxORG, y=scaled_data_posnorma, name='scaled : positive normalization', line = dict(color='blue', width=2, dash='solid'), showlegend=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=xxORG, y=scaled_data_standardization, name='scaled : standardization', line = dict(color='orange', width=2, dash='solid'), showlegend=True), row=2, col=1)
        fig.update_layout(title='feature vs scaled featue', xaxis_title='data points', yaxis_title='amplitude')
        fig.show(config=config)

    return scaled_data_mlx



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


# level : the number of levels to decompose the time signal, le nombre des marquers par signale


def tsig_2_discrete_wavelet_transform(sig, waveletname, level, plotORnot):

	# On peut calculater dans deux façons: 0) dwt en boucle and then idwt, 1) wavedec et waverec
	# Mais le deux ne donnent pas le meme reponses, wavedec et waverec semble plus raisonable.
	
	facon_a_faire = 1
	
	
	if facon_a_faire == 0:
		coef0 = {}
		coef1 = {}
		for k in range(level):
			(coef0[k], coef1[k]) = pywt.dwt(sig, waveletname)
			sig = coef0[k]
			# first coefficient (coef0) is the last level (it has less data points than coef1), feed coef0
			# (the lowest decomposed signal) back into dwt to decompose the signal further
			
		if plotORnot == 1:
			fig, axx = plt.subplots(nrows=level, ncols=2, figsize=(5,5))
			axx[0,0].set_title("coef0")  # Approximation coefficients
			axx[0,1].set_title("coef1")  # Detail coefficients
			for k in range(level):
				axx[k,0].plot(coef0[k], 'r') # output of the low pass filter (averaging filter) of the DWT
				axx[k,1].plot(coef1[k], 'g') # output of the high pass filter (difference filter) of the DWT
			plt.tight_layout()
			plt.show()
		
		coeff = coef1
	
	elif facon_a_faire == 1:
	
		coeff = wavedec(sig, waveletname, level=level)

		if plotORnot == 1:
			fig, axx = plt.subplots(nrows=level, ncols=1, figsize=(5,5))
			axx[0].set_title("coef")  # Pas certain si c'est coef0 ou coef1
			for k in range(level):
				axx[k].plot(coeff[k], 'r') # output of the low pass filter (averaging filter) of the DWT
			plt.tight_layout()
			plt.show()
	
	
	
	coeff_interp = []
	for i in range(level):
		# Interpolate toutes les signales dans coeff avoir le meme taille que sig 
		siglong = linear_intercurrentpt_makeshortSIGlong_interp1d(coeff[i], sig)
		
		# -----------------------------------

		# Normalize img_flatten values: on ne veut pas que des valeurs sont pres de zero
		siglong1 = scale_feature_data(siglong, plotORnot=plotORnot)

		# -----------------------------------
		
		coeff_interp.append(siglong1)
	
	
	return coeff_interp
