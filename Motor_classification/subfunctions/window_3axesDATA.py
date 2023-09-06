import numpy as np

# Plotting
import plotly.graph_objects as go

from subfunctions.make_a_properlist import *


def window_3axesDATA(tr, threeax_data, win):
    
    num_of_dp = len(threeax_data[tr][:,0])
    
    start_val = 0
    
    stop_val = num_of_dp
    # print('stop_val : ' + str(stop_val))
    
    ts = win
    # print('ts : ' + str(ts))
    
    out = np.floor(range(start_val, stop_val, ts))
    # print('out : ' + str(out))
    
    tot_ax0 = []
    tot_ax1 = []
    tot_ax2 = []
    
    shift = 0
    if shift == 0:
        # end point of previous window is one less than the start point of the next window
        bin_tot = int(np.floor((len(out)*win)/(1+win)))
    elif shift == 1:
        # end point of previous window is the same start point of next window 
        if len(out)*win > stop_val:
            bin_tot = len(out)-1
        else:
            bin_tot = len(out)
    # print('bin_tot : ' + str(bin_tot))
    
    
    
    # Break the signal into windows
    sig_win_out = []
    st_ind = []
    end_ind = []
    
    for bins in range(bin_tot):
        # print('bins : ' + str(bins))
        
        st = bins + (bins*(win-shift))
        ender = st+win
        # print('st : ' + str(st))
        # print('ender : ' + str(ender))
        
        st_ind = st_ind + [st]
        end_ind = end_ind + [ender]
        
        # Normal binning
        sig_win1 = threeax_data[tr][st:ender, :]
        # print('sig_win1 : ' + str(sig_win1))
        
        # put the three columns into a list ---> to form a "matrix"
        sig_win_out = sig_win_out + [sig_win1]
        # print('sig_win_out : ' + str(sig_win_out))
        # --------------------
        
        # To plot the columns
        sig_win2 = np.transpose(sig_win1)
        # print('sig_win2 : ' + str(sig_win2))
        
        plotORnot = 0
        if plotORnot == 1:
            # --------------------
            fig = go.Figure()
            config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})

            xxORG = list(range(len(sig_win2[0])))
            fig.add_trace(go.Scatter(x=xxORG, y=sig_win2[0], name='sig_win2[0]', line = dict(color='red', width=2, dash='dash'), showlegend=True))
            fig.add_trace(go.Scatter(x=xxORG, y=sig_win2[1], name='sig_win2[1]', line = dict(color='green', width=2, dash='dash'), showlegend=True))
            fig.add_trace(go.Scatter(x=xxORG, y=sig_win2[2], name='sig_win2[2]', line = dict(color='blue', width=2, dash='dash'), showlegend=True))

            fig.update_layout(title='bins : %d' % (bins), xaxis_title='data points', yaxis_title='axis sign')
            fig.show(config=config)
            # --------------------


    # print('length of sig_win_out : ' + str(len(sig_win_out)))

    plotORnot = 0
    if plotORnot == 1:
        # --------------------
        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        
        # --------------------
        # Original signal
        # --------------------
        xxORG = list(range(len(threeax_data[tr])))
        fig.add_trace(go.Scatter(x=xxORG, y=threeax_data[tr][:,0], name='ORG threeax_data[tr][:,0]', line = dict(color='red', width=2, dash='solid'), showlegend=True))
        fig.add_trace(go.Scatter(x=xxORG, y=threeax_data[tr][:,1], name='ORG threeax_data[tr][:,1]', line = dict(color='green', width=2, dash='solid'), showlegend=True))
        fig.add_trace(go.Scatter(x=xxORG, y=threeax_data[tr][:,2], name='ORG threeax_data[tr][:,2]', line = dict(color='blue', width=2, dash='solid'), showlegend=True))
        # --------------------
        
        # --------------------
        # Windowed signal
        # --------------------
        # print('length of org signal : ' + str(len(threeax_data[tr][:,0])))
        
        # print('bin_tot+len(sig_win2[0])' + str(bin_tot+len(sig_win2[0])))
        
        for bins in range(bin_tot):
            
            sig_win2 = np.transpose(sig_win_out[bins])
            
            # print('sig_win2 : ' + str(sig_win2))
            st = bins + (bins*win)
            ender = st+win
            xxxbit = list(range(st,ender))
            
            if bins%2 == 1:
                sym_num = 0
                couleur0 = 'yellow'
                couleur1 = 'black'
                couleur2 = 'cyan'
            else:
                sym_num = 5
                couleur0 = 'green'
                couleur1 = 'magenta'
                couleur2 = 'blue'
            
            fig.add_trace(go.Scatter(x=xxxbit, y=sig_win2[0], name='sig_win2[0]', mode='markers', marker=dict(color=couleur0, size=8, symbol=sym_num, line=dict(color='green', width=0)), showlegend=True))
            fig.add_trace(go.Scatter(x=xxxbit, y=sig_win2[1], name='sig_win2[1]', mode='markers', marker=dict(color=couleur1, size=8, symbol=sym_num, line=dict(color='magenta', width=0)), showlegend=True))
            fig.add_trace(go.Scatter(x=xxxbit, y=sig_win2[2], name='sig_win2[2]', mode='markers', marker=dict(color=couleur2, size=8, symbol=sym_num, line=dict(color='blue', width=0)), showlegend=True))

        fig.update_layout(title='threeax_data', xaxis_title='data points', yaxis_title='axis sign')
        fig.show(config=config)
        # --------------------
     
     
    st_ind = make_a_properlist(st_ind)
    end_ind = make_a_properlist(end_ind)
    
    return sig_win_out, bin_tot, st_ind, end_ind
