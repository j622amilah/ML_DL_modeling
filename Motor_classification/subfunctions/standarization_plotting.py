import numpy as np

# Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def standarization_plotting(s, tr, normalized_outSIG, outJOY, axis_out, varr, ind_joy_ax_moved, joy_ax_index, cab_index_viaSS, descript, dircor):

    # -------------------------------------------------
    # Plotting
    # -------------------------------------------------
    fig = go.Figure()
    config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})

    xxORG = list(range(len(normalized_outSIG[tr])))

    # ----------
    # Cabin
    # ----------
    fig.add_trace(go.Scatter(x=xxORG, y=normalized_outSIG[tr][:,0], name='cab %s' % (varr['anom'][0]), line = dict(color='red', width=2, dash='dash'), showlegend=True))
    fig.add_trace(go.Scatter(x=xxORG, y=normalized_outSIG[tr][:,1], name='cab %s' % (varr['anom'][1]), line = dict(color='green', width=2, dash='dash'), showlegend=True))
    fig.add_trace(go.Scatter(x=xxORG, y=normalized_outSIG[tr][:,2], name='cab %s' % (varr['anom'][2]), line = dict(color='blue', width=2, dash='dash'), showlegend=True))

    # ----------
    # Joystick
    # ----------
    fig.add_trace(go.Scatter(x=xxORG, y=outJOY[tr][:,0], name=varr['anom'][0], line = dict(color='red', width=2, dash='solid'), showlegend=True))
    fig.add_trace(go.Scatter(x=xxORG, y=outJOY[tr][:,1], name=varr['anom'][1], line = dict(color='green', width=2, dash='solid'), showlegend=True))
    fig.add_trace(go.Scatter(x=xxORG, y=outJOY[tr][:,2], name=varr['anom'][2], line = dict(color='blue', width=2, dash='solid'), showlegend=True))

    # Only plot axes where there is joystick movement
    if descript != 'joyNOT_move':
        for ax in ind_joy_ax_moved:
            
            if ax == 0:
                color_out = 'red'
            elif ax == 1:
                color_out  = 'green'
            elif ax == 2:
                color_out = 'blue'
            
            if joy_ax_index[ax] != 0:
                # 1) Joystick movement detection point
                der_sp = joy_ax_index[ax]*np.ones((2)) # must double the point : can not plot a singal point
                der_sp = [int(x) for x in der_sp] # convert to integer
                # print('joy_detect: ' + str(der_sp))
                fig.add_trace(go.Scatter(x=der_sp, y=outJOY[tr][der_sp, ax], name='joy_detect', mode='markers', marker=dict(color=color_out, size=10, line=dict(color=color_out, width=0)), showlegend=True))
            
            if cab_index_viaSS[ax] != 0:
                # 2) Cabin movement detection point
                sp_i_m = cab_index_viaSS[ax]*np.ones((2)) # must double the point : can not plot a singal point
                sp_i_m = [int(x) for x in sp_i_m] # convert to integer
                # print('cabin_detect: ' + str(sp_i_m))
                fig.add_trace(go.Scatter(x=sp_i_m, y=normalized_outSIG[tr][sp_i_m, ax], name='cabin_detect', mode='markers', marker=dict(color=color_out, size=15, symbol=5, line=dict(color=color_out, width=0)), showlegend=True))
            
            # mode='lines+markers'
            # marker=dict(color=color_out, size=10, symbol=5, line=dict(color=color_out, width=0))
            
            # 0 gives circles
            # 1 gives squares
            # 3 gives '+' signs
            # 5 gives triangles, etc.
            # https://plotly.com/python/reference/#box-marker-symbol



    str_axisval = varr['anom'][axis_out[tr]]
    title_str = 'cab=dash, joy=solid, sub=%d, tr=%d, axis:%s, %s, %s' % (s, tr, str(str_axisval), descript, dircor)
    fig.update_layout(title=title_str, xaxis_title='data points', yaxis_title='Movement')


    fig.show(config=config)

    fig.write_image("images_standard/fig%d.png" % (tr))

    return