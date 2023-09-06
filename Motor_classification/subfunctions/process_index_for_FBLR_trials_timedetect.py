import numpy as np

from scipy.signal import find_peaks

# Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from subfunctions.numderiv import *
from subfunctions.make_a_properlist import *
from subfunctions.detect_sig_change_wrt_baseline import *


def process_index_for_FBLR_trials_timedetect(s, tr, outSIG, marg, varr, starttrial_index, stoptrial_index, plotORnot, plotORnot_derv, axis_val, filename):

    # An FB and LR trial consists of [trial mvt + reinitialization mvt + possible UD initialization mvt] events.
    # For each FB, LR, and rot (RO, PI, YA) start-stop index, this step tries to determine when the 
    # "trial mvt" event stops, so we can remove the reinitialization data portion
  
    # print('tr : ' + str(tr))
    # print('axis_val[tr] : ' + str(axis_val[tr]))

    # Need to add -1 because in python it starts from 0 instead of 1
    # axis_val remains unshifted, ie: LR=1, FB=2, UD=3
    axis_str = varr['anom'][axis_val[tr]]
    # print('axis_str : ' + axis_str)


    # Get the correct LR or FB signal
    axxx = int(axis_val[tr])
    sig = outSIG[tr][:, axxx]

    # ------------------------------------------
    # No need to search for start point : start point is first point
    starttrial_index[tr] = 0
    # ------------------------------------------    



    # ------------------------------------------
    # Determine the stopping point
    # ----------------------------------------
    
    # 1) Determine if there is a possible UD initialization followed by the FB or LR signal
    
    # Do baseline search from startpoint - take the last point of the exited baseline zone as the end point
    baseline = sig[0]
    dpOFsig_in_zone, indexOFsig_in_zone, dp_sign_not_in_zone, indexOFsig_not_in_zone = detect_sig_change_wrt_baseline(sig, baseline, marg)
    
    last_pt = indexOFsig_in_zone[-1]
    
    # Cut data again to get new sig
    sig = sig[0:last_pt]
    
    
    # 2) Perform normal stop point procedure 
    # --------------------------------------------
    # Selection of stoptrial_index_derv
    # --------------------------------------------
    Asig = abs(sig)
    Lsig = len(sig)
    
    # LR or FB cabin movement derivative
    local_slope_vec = numderiv(sig, list(range(len(sig))) )
    local_slope_vec = make_a_properlist(local_slope_vec)

    #print('local_slope_vec : ' + str(local_slope_vec))
    
    if plotORnot_derv == 1:
        # ------------------------------  
        # Is derivative correct?
        # ------------------------------
        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        fig = make_subplots(rows=2, cols=1, start_cell="bottom-left")
            
        dp = np.array(range(len(sig)))
        fig.add_trace(go.Scatter(x=dp, y=sig, name='sig', line = dict(color='magenta', width=2, dash='dash'), showlegend=True), row=1, col=1)
        dp1 = np.array(range(len(local_slope_vec)))
        fig.add_trace(go.Scatter(x=dp1, y=local_slope_vec, name='local_slope_vec', line = dict(color='blue', width=2, dash='dash'), showlegend=True), row=2, col=1)
        fig.show(config=config)
        # ------------------------------
    
    # ------------------------------
    # Finding the maximum and minimum points of the LR or FB cabin movement derivative
    # Need to think about distinguishing the troughs of the peaks
    peaks, properties = find_peaks(local_slope_vec, width=3)

    #print('peaks : ' + str(peaks))
    # ------------------------------

    if plotORnot_derv == 1:
        # ------------------------------  
        # Is peak detection correct?
        # ------------------------------
        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})

        fig = make_subplots(rows=2, cols=1, start_cell="bottom-left")

        # raw signal
        dp = np.array(range(len(sig)))
        fig.add_trace(go.Scatter(x=dp, y=sig, name='sig', line = dict(color='black', width=2, dash='dash'), showlegend=True), row=1, col=1)        

        # markers = signal derivative peaks plotted on signal
        fig.add_trace(go.Scatter(x=peaks, y=sig[peaks], name='peaks', mode='markers', marker=dict(color='magenta', size=10, line=dict(width=0)), showlegend=True), row=1, col=1)

        # derivative of signal
        dp = np.array(range(len(local_slope_vec)))
        fig.add_trace(go.Scatter(x=dp, y=local_slope_vec, name='local_slope_vec', line = dict(color='blue', width=2, dash='dash'), showlegend=True), row=2, col=1)

        # markers = signal derivative peaks plotted on signal derivative
        local_slope_vec = np.array(local_slope_vec)
        fig.add_trace(go.Scatter(x=peaks, y=local_slope_vec[peaks], name='peaks', mode='markers', marker=dict(color='magenta', size=10, line=dict(width=0)), showlegend=True), row=2, col=1)

        fig.show(config=config)
        # ------------------------------

    # ------------------------------
    # Detect the stop index of the trial ---> First selection process **** Take the last min/max point
    if not peaks.any():  # If peaks == []
        stoptrial_index_derv = len(sig)-1
    else:
        stoptrial_index_derv =  peaks[-1]
    # print('stoptrial_index_derv : ' + str(stoptrial_index_derv))
    # --------------------------------------------


    # --------------------------------------------
    # For FB and LR: selction of stoptrial_index_max and stoptrial_index_zoneofmax
    # --------------------------------------------
    # Predict Stop point with the max and a margin-zone
    ival = np.argmax(Asig)
    val = np.max(Asig)

    # Get 2nd selection of stop point index, do not need to add start point because the start point for
    # FB and LR is 1 (movement starts from 0 or initial baseline)
    stoptrial_index_max = ival
    # print('stoptrial_index_max : ' + str(stoptrial_index_max))

    # Using trunctated portion of signal called 'sig', from startpoint to end_of_signal, 
    # so we can start with ivalUD.
    ind3 = []
    val3 = []
    for q in range(stoptrial_index_max, len(sig)):
        if abs(sig[q]) > (val - marg):
            if abs(sig[q]) < (val + marg):
                ind3 = ind3 + [q]
                val3 = val3 + [abs(sig[q])]

    stoptrial_index_zoneofmax = ind3[-1]
    # print('stoptrial_index_zoneofmax : ' + str(stoptrial_index_zoneofmax))        
    
    # Determine which stop point prediction is most accurate
    #diff_bt_stoppts = stoptrial_index_derv - stoptrial_index_zoneofmax
    # --------------------------------------------


    # --------------------------------------------
    # Decision about stoptrial_index
    # --------------------------------------------
    # Way 1:
    # For now take stop point that the last signal derivative peak point
    # stoptrial_index[tr] = stoptrial_index_derv

    # Way 2:
    # Organize all the points in assending order
    pts = stoptrial_index_derv, stoptrial_index_max, stoptrial_index_zoneofmax
    pts = make_a_properlist(pts)
    pts.sort()
    #print('pts : ' + str(pts))
    
    # Most logic : choose the point before the last point (is the max point before a change in slope)
    stoptrial_index[tr] = pts[1] 
    # --------------------------------------------
    # print('stoptrial_index[tr] : ' + str(stoptrial_index[tr]))


    # --------------------------------------------
    # FIGURE: Final plot of choosen start-stop point
    if plotORnot == 1:
        marksize = 15
        
        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        
        dp = np.array(range(Lsig))
        fig.add_trace(go.Scatter(x=dp, y=sig, name='sig', line = dict(color='blue', width=2, dash='dash'), showlegend=True))
        
        fig.add_trace(go.Scatter(x=dp, y=Asig, name='Asig', line = dict(color='black', width=2, dash='dash'), showlegend=True))
        
        #int_st = int(starttrial_index[tr][0])
        #int_stop = int(stoptrial_index[tr][0])
        int_st = starttrial_index[tr]
        int_stop = stoptrial_index[tr]
        dp1 = np.array(range(int_st, int_stop))
        # print('starttrial_index[tr] : ' + str(int_st))
        # print('stoptrial_index[tr] : ' + str(int_stop))


        fig.add_trace(go.Scatter(x=dp1, y=outSIG[tr][int_st:int_stop, int(axxx)], name='final signal', line = dict(color='yellow', width=2, dash='dash'), showlegend=True))

        
        # 0) Start point
        st_pt = starttrial_index[tr]*np.ones((2)) # must double the point : can not plot a singal point
        st_pt = [int(x) for x in st_pt] # convert to integer
        # print('starttrial_index[tr][0] : ' + str(st_pt))
        fig.add_trace(go.Scatter(x=st_pt, y=sig[st_pt], name='startpt', mode='markers', marker=dict(color='black', size=10, line=dict(color='black', width=0)), showlegend=True))

        
        # 1) Derivative stop point
        der_sp = stoptrial_index_derv*np.ones((2)) # must double the point : can not plot a singal point
        der_sp = [int(x) for x in der_sp] # convert to integer
        # print('stoptrial_index_derv: ' + str(der_sp))
        # print('length of sig : ' + str(len(sig)))
        
        fig.add_trace(go.Scatter(x=der_sp, y=sig[der_sp], name='stopderv', mode='markers', marker=dict(color='blue', size=10, line=dict(color='blue', width=0)), showlegend=True))

        # 2) Max baseline shifted zero abs signal stop point
        sp_i_m = stoptrial_index_max*np.ones((2)) # must double the point : can not plot a singal point
        sp_i_m = [int(x) for x in sp_i_m] # convert to integer
        # print('stoptrial_index_max: ' + str(sp_i_m))
        fig.add_trace(go.Scatter(x=sp_i_m, y=sig[sp_i_m], name='stopmax', mode='markers', marker=dict(color='red', size=10, line=dict(color='red', width=0)), showlegend=True))
        
        
        # 3) Baseline zone exit around max stop point
        sp_i_zofm = stoptrial_index_zoneofmax*np.ones((2)) # must double the point : can not plot a singal point
        sp_i_zofm = [int(x) for x in sp_i_zofm] # convert to integer
        # print('stoptrial_index_zoneofmax: ' + str(sp_i_zofm))
        fig.add_trace(go.Scatter(x=sp_i_zofm, y=sig[sp_i_zofm], name='stopzonemax', mode='markers', marker=dict(color='magenta', size=10, line=dict(color='magenta', width=0)), showlegend=True))

        title_str = 'sub: %d, axis: %s, trial: %d, starttrial: %d, stoptrial: %d, derv: %d, max: %d, zonemax: %d, sign=%d, mag=%d' % (s, axis_str, tr, starttrial_index[tr], stoptrial_index[tr], stoptrial_index_derv, stoptrial_index_max, stoptrial_index_zoneofmax)
        
        fig.update_layout(title=title_str, xaxis_title='data points', yaxis_title='Cabin position')

        fig.show(config=config)
        
        # fig.write_image("%s/fig%d.png" % (filename, tr))
        fig.write_image("%s\\fig%d.png" % (filename, tr))
    # --------------------------------------------
    # The end of INSIDE the for statement

    del sig

    return starttrial_index, stoptrial_index
