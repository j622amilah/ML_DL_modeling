import numpy as np

from statistics import mode

from scipy.signal import find_peaks

# Data saving
import pickle

# Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from subfunctions.detect_jumps_in_index_vector import *
from subfunctions.numderiv import *
from subfunctions.make_a_properlist import *
from subfunctions.detect_sig_change_wrt_baseline import *


def process_index_for_UD_trials_timedetect(s, tr, outSIG, starttrial_index, stoptrial_index, plotORnot, plotORnot_derv, marg, filename):
    
    # A UD trial consist of [trial mvt + reinitialization mvt] events.
    # For each UD start-stop index, this step tries to determine when the "trial mvt" event start and stops.
    # We can remove the initialization and reinitialization data portion
    

    # Get the correct UD signal
    sig = outSIG[tr][:,2]

    # ------------------------------------------
    # No need to search for start point : start point is first point
    starttrial_index[tr] = 0
    # ------------------------------------------
    
    # ------------------------------------------
    # Determine the stopping point
    # ------------------------------------------    
    # For each UD trial, when the "reinitialization mvt" event (or end of trial) starts the slope of the 
    # robot movement significantly changes to return to the initial position.  
    # So, we search for the maximum or minimum detected peak, the last peak to be specific, in the cabin movement derivative.
    
    local_slope_vec = numderiv(sig, list(range(len(sig))) )
    local_slope_vec = make_a_properlist(local_slope_vec)

    # ------------------------------
    # Finding the maximum and minimum points of the UD cabin movement derivative
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html 
    # border = min or max range in the y-direction to find signal peaks

    # distance = x-direction interval between peaks

    # peak_prominences = The prominence of a peak measures how much a peak stands out from the surrounding 
    # baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line
    # properties["prominences"].max()   # to get max prominences of signal
    # properties["prominences"], properties["widths"]     # to get prominence and width of signal

    # M
    # width = it searchs for a peak in a certain specified 'window width' 

    # peaks, properties = find_peaks(x, height=(-border, border), distance=150, prominence=1, width=20)   # peaks are the peak indicies

    # Need to think about distinguishing the troughs of the peaks
    
    # Do peaks and troughs of derivative (replicate peakdet)
    # Works for one example, but may not be robust to all 
    # peaks, properties = find_peaks(local_slope_vec, prominence=0.06, width=0.5)  # width=3 and up
    # OR
    # Do peaks only of derivative
    peaks, properties = find_peaks(local_slope_vec, width=3)
    # print('peaks : ' + str(peaks))

    local_slope_vec = np.array(local_slope_vec)
    #print('local_slope_vec[peaks] : ' + str(local_slope_vec[peaks]))


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

    # ------------------------------------------
    # Detect the stop index of the trial ---> First selection process **** Take the last min/max point
    stoptrial_index_derv =  peaks[-1]
    #print('stoptrial_index_derv : ' + str(stoptrial_index_derv))
    # ------------------------------------------

    # You have to cut the signal from starttrial to stoptrial because the true max will not be maximum  
    sig = outSIG[tr][starttrial_index[tr]:int(stoptrial_index_derv), 2] - outSIG[tr][starttrial_index[tr], 2]
    Asig = abs(sig)
    Lsig = len(sig)

    # the 2nd stop point detection, is the positive maximum value of the portion of the UD cabin signal 
    # from the start point to the first detected stop point
    ivalUD = np.argmax(Asig)
    valUD = np.max(Asig)

    # Get 2nd selection of stop point index, wrt to the length of the total signal length
    stoptrial_index_max = starttrial_index[tr] + ivalUD


    # ------------------------------
    #fig = go.Figure()
    #config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
    #t = np.multiply(range(Lsig), 1) 
    # 3) Max baseline shifted zero abs signal stop point
    #sp_i_m = ivalUD*np.ones((2)) # must double the point : can not plot a singal point
    #sp_i_m = [int(x) for x in sp_i_m] # convert to integer
    #print('ivalUD: ' + str(sp_i_m))
    #fig.add_trace(go.Scatter(x=sp_i_m, y=Asig[sp_i_m], name='stopmax', mode='markers', marker=dict(color='red', size=10, line=dict(color='red', width=0)), showlegend=True))
    #fig.add_trace(go.Scatter(x=t, y=Asig, name='Asig', line = dict(color='green', width=2, dash='dash'), showlegend=True))
    #fig.add_trace(go.Scatter(x=t, y=sig, name='org sig', line = dict(color='black', width=2, dash='dash'), showlegend=True))
    #fig.show(config=config)
    # ------------------------------

    # Two or more start selection points detected : typical UD trial (robot goes out of the baseline zone and back in)
    # Get 3rd selection of stop point index: using the max location of the signal, we put a baseline zone around it to detect when the 
    # max value decreases.  This could give us a better stop point prediction (it is a point in between 
    # stoptrial_index_max and stoptrial_index_derv)

    # Using trunctated portion of signal called 'sig', from startpoint to end_of_signal, 
    # so we can start with ivalUD.
    # If the value is in the baseline zone put it in the vector ind3 and val3.
    # The last value of this vector is when it goes out of the zone.

    ind3 = []
    val3 = []
    # the search length is different than the length of Asig so we do not use detect_sig_change_wrt_baseline
    for q in range(ivalUD, Lsig):
        if Asig[q] > (valUD - marg):
            if Asig[q] < (valUD + marg):
                ind3 = ind3 + [q]
                val3 = val3 + [Asig[q]]

    # ------------------------------------------
    stoptrial_index_zoneofmax = starttrial_index[tr] + ind3[-1]
    #print('stoptrial_index_zoneofmax : ' + str(stoptrial_index_zoneofmax))
    # ------------------------------------------
    
    # print('starttrial_index : ' + str(starttrial_index))
    # print('stoptrial_index : ' + str(stoptrial_index))
    
    # --------------------------------------------
    
    
    # ------------------------------
    # Determine if UD trial is constant stimulation, increasing/decreasing
    # Search along starting point to determine if signal move away from starting point value.
    # ------------------------------
    baseline = outSIG[tr][starttrial_index[tr], 2]
    # print('baseline : ' + str(baseline))
    # ------------------------------
    
    # Search along the signal LR/UD from first point, to see when I hit the zone around the baseline; 
    # to find the index that corresponds to the baseline
    dpOFsig_in_zone, indexOFsig_in_zone, dp_sign_not_in_zone, indexOFsig_not_in_zone = detect_sig_change_wrt_baseline(outSIG[tr][:, 2], baseline, marg)
    
    plotORnot0 = 0
    if plotORnot0 == 1:
        # ------------------------------
        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        t = np.multiply(range(0, len(outSIG[tr][:, 2])), 1) 
        plotb = baseline*np.ones((len(outSIG[tr][:, 2])))

        fig.add_trace(go.Scatter(x=t, y=plotb, name='baseline', line = dict(color='black', width=2, dash='dash'), showlegend=True))
        
        fig.add_trace(go.Scatter(x=t, y=outSIG[tr][:,0], name='cab LR', line = dict(color='red', width=2, dash='solid'), showlegend=True))
        
        fig.add_trace(go.Scatter(x=t, y=outSIG[tr][:,2], name='cab UD', line = dict(color='green', width=2, dash='solid'), showlegend=True))
        outval = np.array(indexOFsig_in_zone)
        fig.add_trace(go.Scatter(x=outval, y=dpOFsig_in_zone, name='pts inside of zone', mode='markers', marker=dict(color='green', size=10, line=dict(width=0)), showlegend=True))
        outval1 = np.array(indexOFsig_not_in_zone)
        fig.add_trace(go.Scatter(x=outval1, y=dp_sign_not_in_zone, name='pts outside of zone', mode='markers', marker=dict(color='magenta', size=10, line=dict(width=0)), showlegend=True))
        fig.show(config=config)
        # ------------------------------
    
    
    # Search for non-consecutive data points, or jumps in the data
    # Jumps in the x index value tells when you enter, exit, enter, etc... the baseline zone
    index_st, value_st, index_end, value_end = detect_jumps_in_index_vector(indexOFsig_in_zone)
    
    index_st_niz, value_st_niz, index_end_niz, value_end_niz = detect_jumps_in_index_vector(indexOFsig_not_in_zone)
    # append the point in which the signal goes out of the baseline (this is a viable stop point, it is when the robot returns to start point UD)
    value_end = value_end, value_st_niz[-1]
    value_end = make_a_properlist(value_end)
    
    # print('value_st : ' + str(value_st))
    # print('value_end : ' + str(value_end))
    # ------------------------------

    if len(value_st) == 1:
        # Zero stim : robot stays in the baseline zone the entire time
        # It never goes out of the start baseline zone, so the stop point is the corresponding end value (the entire length of the signal)
        
        stoptrial_index[tr] = value_end[-1] # the last point before the robot descends
    else:
        # Way 1:
        # If stoptrial_index_derv is close to the end of the signal, choose stoptrial_index_zoneofmax
        # If stop point is close to the signal end we erroneously include data from reinitialization
        #if stoptrial_index_derv > len(outSIG[tr][:,2])-50:
        #    stoptrial_index[tr] = stoptrial_index_zoneofmax
        #else:
        #    # Risks to have a small initial portion of reinitialization
        #    stoptrial_index[tr] = stoptrial_index_derv
        
        # Way 2:
        # Organize all the points in assending order
        # print('------------- tr : ' + str(tr))
        # print('starttrial_index[tr] : ' + str(starttrial_index[tr]))
        # print('value_st : ' + str(value_st))
        # print('value_end : ' + str(value_end))
        # print('stoptrial_index_derv : ' + str(stoptrial_index_derv))
        # print('stoptrial_index_max : ' + str(stoptrial_index_max))
        # print('stoptrial_index_zoneofmax : ' + str(stoptrial_index_zoneofmax))
        
        pts = starttrial_index[tr], value_st, value_end, stoptrial_index_derv, stoptrial_index_max, stoptrial_index_zoneofmax
        pts = make_a_properlist(pts)
        # print('before sorting pts : ' + str(pts))
        pts.sort()
        # print('after sorting pts : ' + str(pts))
        # print('value_end[-1] : ' + str(value_end[-1]))
        
        # Most logic : choose the point before the larges value_end[-1] OR value_st[-1] value (the return to UD start) 
        keeper = []            
        for ii in range(len(pts)):
            if np.max([value_end[-1], value_st[-1]]) > pts[ii]:
                keeper = keeper + [pts[ii]]
        # print('keeper : ' + str(keeper))
        stoptrial_index[tr] = keeper[-1]
        # print('stoptrial_index[tr] : ' + str(stoptrial_index[tr]))
    # --------------------------------------------
    
    # print('starttrial_index : ' + str(starttrial_index))
    # print('stoptrial_index : ' + str(stoptrial_index))
    
    # --------------------------------------------	
    # To get the experimental stimulation direction (double checking experimental matrix) : 1, -1, 0
    # We look for the experimental stmulus direction at the start of the trial, because it is highly
    # likely not influenced by participant interaction - this should give the true experimental settings   
    
    # You can not look at the slope of the intial curve like in LR and FB, because UD starts at different 
    # times from the startpoint and it takes a longer time for the cabin to move up or down.  It is more
    # reliable to look for the directional difference from the startpoint to when 'out of zone' is detected. 
    
    # print('starttrial_index[tr] : ' + str(starttrial_index[tr]))
    # print('stoptrial_index[tr] : ' + str(stoptrial_index[tr]))
    
    starttrial_index = [int(x) for x in starttrial_index]
    stoptrial_index = [int(x) for x in stoptrial_index]
    
    vec = outSIG[tr][starttrial_index[tr]:stoptrial_index[tr], 2]
    baseline = vec[0]
    marg = 10
    dpOFsig_in_zone, indexOFsig_in_zone, dp_sign_not_in_zone, indexOFsig_not_in_zone = detect_sig_change_wrt_baseline(vec, baseline, marg)

    #print('dpOFsig_in_zone : ' + str(dpOFsig_in_zone))
    #print('indexOFsig_in_zone : ' + str(indexOFsig_in_zone))
    #print('dp_sign_not_in_zone : ' + str(dp_sign_not_in_zone))
    #print('indexOFsig_not_in_zone : ' + str(indexOFsig_not_in_zone))
    
    # ------------------------------
    #fig = go.Figure()
    #config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
    #t = np.multiply(range(0, len(vec)), 1) 
    #plotb = baseline*np.ones((len(vec)))        
    #fig.add_trace(go.Scatter(x=t, y=plotb))
    #fig.add_trace(go.Scatter(x=t, y=vec))
    #outval = np.array(indexOFsig_in_zone)
    #fig.add_trace(go.Scatter(x=outval, y=dpOFsig_in_zone, name='pts inside of zone', mode='markers', marker=dict(color='green', size=10, line=dict(width=0)), showlegend=True))
    #outval1 = np.array(indexOFsig_not_in_zone)
    #fig.add_trace(go.Scatter(x=outval1, y=dp_sign_not_in_zone, name='pts outside of zone', mode='markers', marker=dict(color='magenta', size=10, line=dict(width=0)), showlegend=True))
    #fig.show(config=config)
    # ------------------------------
    

    # --------------------------------------------
    if plotORnot == 1:
        marksize = 15
        L_Osig = len(outSIG[tr][:,2])
        Osig = abs(outSIG[tr][:,2])

        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})

        dp = np.array(range(L_Osig))
        fig.add_trace(go.Scatter(x=dp, y=Osig, name='sig', line = dict(color='blue', width=2, dash='dash'), showlegend=True))

        #dp1 = np.array(range(int(starttrial_index[tr]), int(stoptrial_index[tr])))
        #fig.add_trace(go.Scatter(x=dp1, y=outSIG[tr][int(starttrial_index[tr]):int(stoptrial_index[tr]), 2], name='final signal', line = dict(color='yellow', width=2, dash='dash'), showlegend=True))
        dp1 = np.array(range(starttrial_index[tr], stoptrial_index[tr]))
        fig.add_trace(go.Scatter(x=dp1, y=outSIG[tr][starttrial_index[tr]:stoptrial_index[tr], 2], name='final signal', line = dict(color='yellow', width=2, dash='dash'), showlegend=True))

        # 0) Start point
        st_pt = starttrial_index[tr]*np.ones((2)) # must double the point : can not plot a singal point
        st_pt = [int(x) for x in st_pt] # convert to integer
        #print('starttrial_index[tr] : ' + str(st_pt))
        fig.add_trace(go.Scatter(x=st_pt, y=Osig[st_pt], name='startpt', mode='markers', marker=dict(color='black', size=10, line=dict(color='black', width=0)), showlegend=True))
        
        # -------
        # Original :
        # 1) Stop point in Start point zone
        stop_pt_in_spz = value_end[-1]*np.ones((2)) # must double the point : can not plot a singal point
        stop_pt_in_spz = [int(x) for x in stop_pt_in_spz] # convert to integer
        #print('value_end[-1] : ' + str(stop_pt_in_spz))
        fig.add_trace(go.Scatter(x=stop_pt_in_spz, y=Osig[stop_pt_in_spz], name='stopzonestart', mode='markers', marker=dict(color='black', size=10, line=dict(color='black', width=0)), showlegend=True))
        # -------
        
        # -------
        # Correction
        # value_end = np.array(value_end)
        # if value_end.any():
            # # 1) Stop point in Start point zone
            # stop_pt_in_spz = value_end[-1]*np.ones((2)) # must double the point : can not plot a singal point
            # stop_pt_in_spz = [int(x) for x in stop_pt_in_spz] # convert to integer
            # #print('value_end[-1] : ' + str(stop_pt_in_spz))
            # fig.add_trace(go.Scatter(x=stop_pt_in_spz, y=Osig[stop_pt_in_spz], name='stopzonestart', mode='markers', marker=dict(color='black', size=10, line=dict(color='black', width=0)), showlegend=True))
        # -------

        # 2) Derivative stop point
        der_sp = stoptrial_index_derv*np.ones((2)) # must double the point : can not plot a singal point
        der_sp = [int(x) for x in der_sp] # convert to integer
        #print('stoptrial_index_derv: ' + str(der_sp))
        fig.add_trace(go.Scatter(x=der_sp, y=Osig[der_sp], name='stopderv', mode='markers', marker=dict(color='blue', size=10, line=dict(color='blue', width=0)), showlegend=True))

        # 3) Max baseline shifted zero abs signal stop point
        sp_i_m = stoptrial_index_max*np.ones((2)) # must double the point : can not plot a singal point
        sp_i_m = [int(x) for x in sp_i_m] # convert to integer
        #print('stoptrial_index_max: ' + str(sp_i_m))
        fig.add_trace(go.Scatter(x=sp_i_m, y=Osig[sp_i_m], name='stopmax', mode='markers', marker=dict(color='red', size=10, line=dict(color='red', width=0)), showlegend=True))

        # 4) Baseline zone exit around max stop point
        sp_i_zofm = stoptrial_index_zoneofmax*np.ones((2)) # must double the point : can not plot a singal point
        sp_i_zofm = [int(x) for x in sp_i_zofm] # convert to integer
        #print('stoptrial_index_zoneofmax: ' + str(sp_i_zofm))
        fig.add_trace(go.Scatter(x=sp_i_zofm, y=Osig[sp_i_zofm], name='stopzonemax', mode='markers', marker=dict(color='magenta', size=10, line=dict(color='magenta', width=0)), showlegend=True))

        # -------------------------
        # 0) Peaks of signal derivative
        for q in range(len(value_st)):
            vs = value_st[q]*np.ones((2)) # must double the point : can not plot a singal point
            vs = [int(x) for x in vs] # convert to integer
            fig.add_trace(go.Scatter(x=vs, y=outSIG[tr][vs, 2], name='st derv peaks', mode='markers', marker=dict(color='cyan', size=5, line=dict(color='cyan', width=0)), showlegend=True))
        
        # -------
        # Original :
        for q in range(len(value_end)):
            ve = value_end[q]*np.ones((2)) # must double the point : can not plot a singal point
            ve = [int(x) for x in ve] # convert to integer
            fig.add_trace(go.Scatter(x=ve, y=outSIG[tr][ve, 2], name='end derv peaks', mode='markers', marker=dict(color='orange', size=5, line=dict(color='orange', width=0)), showlegend=True))
            
        title_str = 'sub: %d, axis : UD, trial: %d, starttrial: %d, stoptrial: %d, stoppt derv: %d, stoppt max: %d, stoppt zone: %d, value end: %d, sign=%d, mag=%d' % (s, tr, starttrial_index[tr], stoptrial_index[tr], stoptrial_index_derv, stoptrial_index_max, stoptrial_index_zoneofmax, value_end[-1])
        # -------
        
        # -------
        # Correction
        # if value_end.any():
            # for q in range(len(value_end)):
                # ve = value_end[q]*np.ones((2)) # must double the point : can not plot a singal point
                # ve = [int(x) for x in ve] # convert to integer
                # fig.add_trace(go.Scatter(x=ve, y=outSIG[tr][ve, 2], name='end derv peaks', mode='markers', marker=dict(color='orange', size=5, line=dict(color='orange', width=0)), showlegend=True))

            # title_str = 'sub: %d, axis : UD, trial: %d, starttrial: %d, stoptrial: %d, stoppt derv: %d, stoppt max: %d, stoppt zone: %d, value end: %d, sign=%d, mag=%d' % (s, tr, starttrial_index[tr], stoptrial_index[tr], stoptrial_index_derv, stoptrial_index_max, stoptrial_index_zoneofmax, value_end[-1])
        # else: 
            # title_str = 'sub: %d, axis : UD, trial: %d, starttrial: %d, stoptrial: %d, stoppt derv: %d, stoppt max: %d, stoppt zone: %d, sign=%d, mag=%d' % (s, tr, starttrial_index[tr], stoptrial_index[tr], stoptrial_index_derv, stoptrial_index_max, stoptrial_index_zoneofmax)
        # -------

        fig.update_layout(title=title_str, xaxis_title='data points', yaxis_title='Cabin position')

        fig.show(config=config)
        
        # fig.write_image("%s/fig%d.png" % (filename, tr))
        fig.write_image("%s\\fig%d.png" % (filename, tr))
    # --------------------------------------------


    
    return starttrial_index, stoptrial_index
