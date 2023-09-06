import numpy as np

# Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Personal python functions
from subfunctions.make_a_properlist import *

import os

def check_axes_assignmentPLOT(s, outJOY, outSIG, axis_out, varr, filename, time_org):
    
    
    # create a directory for saving images
    if not os.path.exists("%s/%s_s%d" % (varr['main_path1'], filename, s)):
        os.mkdir("%s/%s_s%d" % (varr['main_path1'], filename, s))
        
    # if not os.path.exists("%s\\%s_s%d" % (varr['main_path1'], filename, s)):
    #     os.mkdir("%s\\%s_s%d" % (varr['main_path1'], filename, s))
    
    max_sig_val = []
    for tr in range(len(axis_out)):
        # print('tr : ' + str(tr))
    
        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        
        # ----------
        # Cabin
        # ----------
        if time_org[tr] == 0:  # when plotting after cutting, we put a 0 in the data to denote a bad trial 
            xxORG = list(range(len(outSIG[tr])))
            xaxis_name = 'data points'
        else: 
            xxORG = time_org[tr]
            xaxis_name = 'time (s)'
        fig.add_trace(go.Scatter(x=xxORG, y=outSIG[tr][:,0], name='cab %s' % (varr['anom'][0]), line = dict(color='red', width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=xxORG, y=outSIG[tr][:,1], name='cab %s' % (varr['anom'][1]), line = dict(color='green', width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=xxORG, y=outSIG[tr][:,2], name='cab %s' % (varr['anom'][2]), line = dict(color='blue', width=2, dash='dash'), showlegend=True))
        
        axis_out = [int(x) for x in axis_out]
        
        

        # ----------
        # Joystick
        # ----------
        fig.add_trace(go.Scatter(x=xxORG, y=outJOY[tr][:,0], name='joy %s' % (varr['anom'][0]), line = dict(color='red', width=2, dash='solid'), showlegend=True))
        fig.add_trace(go.Scatter(x=xxORG, y=outJOY[tr][:,1], name='joy %s' % (varr['anom'][1]), line = dict(color='green', width=2, dash='solid'), showlegend=True))
        fig.add_trace(go.Scatter(x=xxORG, y=outJOY[tr][:,2], name='joy %s' % (varr['anom'][2]), line = dict(color='blue', width=2, dash='solid'), showlegend=True))
        
        mag_sum_joy = np.sum(outJOY[tr])
        str_axisval = varr['anom'][axis_out[tr]]
        title_str = 'cab=dash, joy=solid, sub=%d, tr=%d, axis:%s, joymagsum=%f' % (s, tr, str(str_axisval), mag_sum_joy)
        fig.update_layout(title=title_str, xaxis_title=xaxis_name, yaxis_title='Movement')
        
        fig.show(config=config)
        
        fig.write_image("%s/%s_s%d/fig%d.png" % (varr['main_path1'], filename, s, tr))
        # fig.write_image("%s\\%s_s%d\\fig%d.png" % (varr['main_path1'], filename, s, tr))

    return
