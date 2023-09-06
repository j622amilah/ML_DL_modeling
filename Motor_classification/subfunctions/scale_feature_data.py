from mlxtend.preprocessing import minmax_scaling

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pandas as pd
import numpy as np

def scale_feature_data(feat, plotORnot):

	if int(np.mean(feat)) == 0:
		scaled_data_mlx = feat
	else:
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
