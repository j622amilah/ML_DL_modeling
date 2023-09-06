import numpy as np

# Filtering
from scipy import signal

import plotly.graph_objects as go


def filter_sig3axes(sig3axes, plotORnot):

    n = 4   # filter order
    fs = 250 # data sampling frequency (Hz)
    fc = 10  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(n, w, 'low')  # 3rd order

    sig3axes_filter = []
    for tr in range(len(sig3axes)):
        output = signal.filtfilt(b, a, np.transpose(sig3axes[tr]))
        sig3axes_filter = sig3axes_filter + [np.transpose(output)]
        
    # print('sig3axes_filter[tr] : ' + str(sig3axes_filter[0]))

    
    if plotORnot == 1:
        # --------------------
        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})

        xxORG = list(range(len(sig3axes[tr])))
        fig.add_trace(go.Scatter(x=xxORG, y=sig3axes[tr][:,0], name='sig3axes[tr][:,0]', line = dict(color='red', width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=xxORG, y=sig3axes[tr][:,1], name='sig3axes[tr][:,1]', line = dict(color='green', width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=xxORG, y=sig3axes[tr][:,2], name='sig3axes[tr][:,2]', line = dict(color='blue', width=2, dash='dash'), showlegend=True))

        xxORG = list(range(len(sig3axes_filter[tr])))
        fig.add_trace(go.Scatter(x=xxORG, y=sig3axes_filter[tr][:,0], name='sig3axes_filter[tr][:,0]', line = dict(color='red', width=2, dash='solid'), showlegend=True))
        fig.add_trace(go.Scatter(x=xxORG, y=sig3axes_filter[tr][:,1], name='sig3axes_filter[tr][:,1]', line = dict(color='green', width=2, dash='solid'), showlegend=True))
        fig.add_trace(go.Scatter(x=xxORG, y=sig3axes_filter[tr][:,2], name='sig3axes_filter[tr][:,2]', line = dict(color='blue', width=2, dash='solid'), showlegend=True))

        fig.update_layout(title='joystick signal', xaxis_title='data points', yaxis_title='joystick')
        fig.show(config=config)
        # --------------------

    return sig3axes_filter