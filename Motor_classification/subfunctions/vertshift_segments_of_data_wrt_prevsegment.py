import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from subfunctions.detect_jumps_in_data import *
from subfunctions.detect_nonconsecutive_values import *
from subfunctions.make_a_properlist import *



def vertshift_segments_of_data_wrt_prevsegment(*arg):

    # Created by Jamilah Foucher, March 10, 2021
    # 
    # Purpose: aligns data segments with respect to the last point in the previous data segments.  
    # The first data segment is aligned with zero. 
    # 
    # Input VARIABLES:
    # 1) tt is the signal (that you would like to vertically shift per jump segment)
    # 
    # (2) dp_jump is the data point jump in the signal y that you would like to detect 
    # 
    # Output VARIABLES:
    # (1) vshift_vec_out


    # vshift_vec_out = vertshift_segments_of_data_wrt_prevsegment(tt, dp_jump, varr.sumf_path)

    # Mandatory INPUTS
    tt = arg[0]
    dp_jump = arg[1]

    # ------------------------------

    # ------------------------------------
    # Detect jumps in time data
    #dp_jump = 1     # if this value is exceeded, then we say a jump is detected, detect jumps in the time greater than 1 second
    
    # desired_break_num is number of time you would like to search for jump detection.  
    # Sometimes you only want to detect if there is a jump in the data, and if so you move on to another process.  
    # If so, set desired_break_num=1.
    desired_break_num = len(tt)    # search the entire data length (give max value)
    
    ind_jumpPT = detect_jumps_in_data(tt, dp_jump, desired_break_num)
    #print('ind_jumpPT : ' + str(ind_jumpPT))
    # ------------------------------------
    
    # ------------------------------------
    # Need to get the elasped time - to determine if the data cut is sufficiently long enough for a trial
    ind_jumpPT = np.array(ind_jumpPT)
    if not ind_jumpPT.any(): # ind_jumpPT == []: # There are no abnormal y-axis jumps in data
        # Shift entire time vector to zero and finish
        tt2 = tt - tt[0]
        printpts = 0
    else:
        # ------------------
        tt1 = tt - tt[0]    # Shift time vector to zero
        printpts = 1
        
        ind_jumpPT = np.reshape(ind_jumpPT, (len(ind_jumpPT), 1))  # Convert to column
        tind_temp = 0, ind_jumpPT, len(tt1)-1

        # Ensure that tind_temp is a proper list - sometimes there are several arrays in a list
        tind_temp_list = make_a_properlist(tind_temp)
        # ------------------
        
        # ------------------
        # Do non-consecutive to get first touch point to zero
        non_consec_vec, non_consec_ind = detect_nonconsecutive_values(tind_temp_list)
        # ------------------
        
        # ------------------
        if non_consec_vec[len(non_consec_vec)-1] == len(tt1)-2:
            #last value cut off
            tind0_0 = non_consec_vec[0:len(non_consec_vec)-1], len(tt1)-1
            tind0 = make_a_properlist(tind0_0)
        else:
            tind0 = non_consec_vec
        
        tind = np.unique(tind0)     # Need unique monotonically increasing values for the index below
        
        # ------------------
        # Initialization
        if len(tind0) > len(tind):
            if tind0[0] == tind0[1]:
                
                ydist = abs( tt1[tind[0]+1] - tt1[tind[0]] )
                
                if tt1[tind[0]] > tt1[tind[0]+1]:
                    tt2 = np.concatenate( ( tt1[tind[0]], (ydist+0.1)+tt1[tind[0]+1:tind[1]] ), axis=0) # row vector
                else:
                    tt2 = np.concatenate( ( tt1[tind[0]], (ydist+0.1)-tt1[tind[0]+1:tind[1]] ), axis=0) # row vector
        else:
            ydist = 0
            tt2 = tt1[tind[0]:tind[1]]
        # ------------------
        
        
        # ------------------
        # Initialization : exception for jump at second point because it is a consecutive count from 1
        # It is present in tind_temp, but gets eliminated afterwards
        # %if (tind0(2,1) == 1) || (tind_temp(1,1)+1 == tind_temp(2,1)) %(tind0(2,1) == 1) means that (tind0(1,1) == 1) automatically
        if ind_jumpPT[0] == 1:
            
            # explicit exception due to ind_jumpPT(1,1)=2
            ydist = abs( tt1[ind_jumpPT[0]+1] - tt1[ind_jumpPT[0]] )
            
            if tt1[tind_temp_list[1]] > tt1[tind_temp_list[1]+1]:
                tt2 = np.concatenate( ( tt1[tind_temp_list[0]:tind_temp_list[1]], (ydist+0.1)+tt1[tind_temp_list[1]+1:tind_temp_list[2]] ), axis=0)
            else:
                tt2 = np.concatenate( ( tt1[tind_temp_list[0]:tind_temp_list[1]], (ydist+0.1)-tt1[tind_temp_list[1]+1:tind_temp_list[2]] ), axis=0)
        # ------------------
        
        # Get line from remaining jump points : there are no exceptions so we can loop until the end
        for j in range(len(tind)-2):
            ydist = ydist + ( tt1[int(tind[int(j+1)])] - tt1[int(tind[int(j+1)]+1)] )
            tt2 = np.concatenate((tt2, ydist+0.1+tt1[ int(tind[int(j+1)]+1):int(tind[int(j+2)]) ]), axis=0) 
        # ------------------

    # END of else indentation
        
        
    # ------------------
    # Generate missing 1 or 2 points at the end : 
    # shift time vector needs to be the same length as original time vector
    diffpad = tt2[len(tt2)-1] - tt2[len(tt2)-2]
    
    #print('length of tt : ' + str(len(tt)))
    #print('length of tt2 : ' + str(len(tt2)))
    
    missingpt = len(tt)-len(tt2)
    ptval = tt2[len(tt2)-1]
    last2pt = np.zeros((missingpt))
    for u in range(missingpt):
        ptval = ptval + diffpad
        last2pt[u] = ptval 
    
    tt3 = tt2, last2pt
    tt2 = make_a_properlist(tt3)  # Save reconstructed monotonically increasing time vector
    # ------------------------------------
     
    vshift_vec_out = tt2
    #print('length of vshift_vec_out : ' + str(len(vshift_vec_out)))
    
    # ------------------
    plotORnot = 0  # 1 = show figures, 0 = do not show figures

    if plotORnot == 1:
        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        dp = np.array(range(len(tt)))

        fig.add_trace(go.Scatter(x=dp, y=tt, name='time', line = dict(color='black', width=2, dash='dash'), showlegend=True))
    
        if printpts == 1:
            fig.add_trace(go.Scatter(x=tind, y=tt[tind], name='jumps_in_time', mode='markers', marker=dict(color='red', size=5, line=dict(color='red', width=0)), showlegend=True))

        fig.add_trace(go.Scatter(x=dp, y=vshift_vec_out, name='time', line = dict(color='orange', width=2, dash='dash'), showlegend=True))
        fig.show(config=config)
    # ------------------
    
                                        
    return vshift_vec_out