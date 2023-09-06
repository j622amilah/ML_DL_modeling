import numpy as np
    
# Plotting
import plotly.graph_objects as go

from subfunctions.findall import *
from subfunctions.detect_sig_change_wrt_baseline import *
from subfunctions.detect_jumps_in_index_vector import *
from subfunctions.make_a_properlist import *






def cut_initial_trials(varr, A, marg_of_zero, num_pt_grp, plotORnot_allpts, plotORnot, which_data2use):

    # ------------------------------
    # (1) Cut the data by looking at when the data crosses zero
    # ------------------------------
    # a) Detect points around 0/initial starting point
    cpos1 = make_a_properlist(A[:,6-1])  # RO/LR
    cpos2 = make_a_properlist(A[:,7-1])  # PI/FB
    cpos3 = make_a_properlist(A[:,8-1])  # YA/UD
    
    # print('Cut the data by looking at when the data crosses zero\n')
    if which_data2use == 'cabin':
        sum_sig = []
        for i in range(len(cpos1)):
            tot = cpos1[i] + cpos2[i] + cpos3[i]
            sum_sig = sum_sig + [tot]
         
    elif which_data2use == 'time':
        # Use time to get cut index
        sum_sig = make_a_properlist(A[:,2-1])  # time

    #print('length of sum_sig : ' + str(len(sum_sig)))

    # Want to detect points around zero with a +/- margin of 0.001
    baseline = 0
    dpOFsig_in_zone, indexOFsig_in_zone, dp_sign_not_in_zone, indexOFsig_not_in_zone = detect_sig_change_wrt_baseline(sum_sig, baseline, marg_of_zero)

    #print('dpOFsig_in_zone : ' + str(dpOFsig_in_zone))
    #print('indexOFsig_in_zone : ' + str(indexOFsig_in_zone))
    #print('dp_sign_not_in_zone : ' + str(dp_sign_not_in_zone))
    #print('indexOFsig_not_in_zone : ' + str(indexOFsig_not_in_zone))

    # ------------------------------
    if plotORnot_allpts == 1:
        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        t = np.multiply(range(0, len(sum_sig)), 1)
        
        fig.add_trace(go.Scatter(x=t, y=sum_sig, name='sum_sig', line = dict(color='black', width=2, dash='dash'), showlegend=True))
        
        fig.add_trace(go.Scatter(x=t, y=cpos1, name='cpos1', line = dict(color='red', width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=t, y=cpos2, name='cpos2', line = dict(color='green', width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=t, y=cpos3, name='cpos3', line = dict(color='blue', width=2, dash='dash'), showlegend=True))
        
        plotb = baseline*np.ones((len(sum_sig)))
        fig.add_trace(go.Scatter(x=t, y=plotb))
        
        outval = np.array(indexOFsig_in_zone)
        fig.add_trace(go.Scatter(x=outval, y=dpOFsig_in_zone, name='pts inside of zone', mode='markers', marker=dict(color='green', size=10, line=dict(width=0)), showlegend=True))
        
        # outval1 = np.array(indexOFsig_not_in_zone)
        # fig.add_trace(go.Scatter(x=outval1, y=dp_sign_not_in_zone, name='pts outside of zone', mode='markers', marker=dict(color='magenta', size=10, line=dict(width=0)), showlegend=True))
        
        fig.show(config=config)
    # ------------------------------


    # b) Look for consecutive points in the zone : start-stop area groups in between trials
    index_st, value_st, index_end, value_end = detect_jumps_in_index_vector(indexOFsig_in_zone)
    
    # Extra check : add last point to value_end
    if value_end[len(value_end)-1] != len(sum_sig):
        value_end = value_end + [len(sum_sig)]

    # value_st and value_end SHOULD be the same length
    # print('value_st : ' + str(value_st))
    # print('value_end : ' + str(value_end))
    # print('length of value_st : ' + str(len(value_st)))
    # print('length of value_end : ' + str(len(value_end)))

    # c) Decide how many points in a group represent a true positive (non-relavant for trans, needed for rotation) 
    lenvec = np.zeros((len(value_st)))
    for j in range(len(value_st)):
        lenvec[j:j+1] = len(np.multiply(range(value_st[j], value_end[j]), 1))
    # print('lenvec : ' + str(lenvec))
        
    # The number of data points grouped together that is considered as a reliable trial-cut
    newvec, ind_of_index = findall(lenvec, '>=', num_pt_grp)

    # print('ind_of_index: ' + str(ind_of_index))

    # d) Get mid-point of each group of points
    # The indexed vector must be an array and the index can be a list
    value_st = np.array(value_st)
    value_end = np.array(value_end)
    
    new2_ind_st = value_st[ind_of_index]
    new2_ind_end = value_end[ind_of_index]
    
    
    # Keep the groups together
    
    
    # Combine the two points into one point : two patterns: 1) small distance between point=find middle point, large distance=count both points
    diff = new2_ind_end - new2_ind_st
    new3_ind_all = []

    for i in range(len(diff)):
        if diff[i] < 50: 
            new3_ind_all = new3_ind_all + [new2_ind_st[i] + np.floor(diff[i]/2)]
        else:
            # This accounts for zero-stim trial
            new3_ind_all = new3_ind_all + [new2_ind_st[i]]
            new3_ind_all = new3_ind_all + [new2_ind_end[i]]

    # e) Make start-stop index from mid-points
    new3_ind_all = [int(x) for x in new3_ind_all]
    # print('new3_ind_all : ' + str(new3_ind_all))

    
    # Take care of exceptions : first and last point
    if new3_ind_all[0] == 0:
        # pass all values to new3_ind_st, except the last point at the end
        new3_ind_st = new3_ind_all[0:len(new3_ind_all)-1]
    else:
        # check if first point of new3_ind_all is close to 0 or if it is missing (only happens for UD start)
        if new3_ind_all[0] > 150:  # stimulation was programmed to last for 15 secs
            # the first point is MISSING - a UD trial
            # add 0 onto new2_ind_st
            new3_ind_st = 0, new3_ind_all[0:len(new3_ind_all)-1]
            
        else:
            #the first point is there but it is not at zero
            # replace first point with zero
            new3_ind_st = 0, new3_ind_all[1:len(new3_ind_all)-1]
       
    new3_ind_st = make_a_properlist(new3_ind_st)
    new3_ind_st = np.array(new3_ind_st)  # turn into an array, need an array to index it

    L = len(new3_ind_st)
    new3_ind_end = np.zeros((L))
    for ii in range(1,L):
        new3_ind_end[int(ii-1):int(ii)] = new3_ind_st[int(ii):int(ii+1)]-1
    new3_ind_end[int(L-1):int(L)] = len(sum_sig)-1

    # Check length of both - they need to both be the same length
    new3_ind_st = [int(x) for x in new3_ind_st] 
    new3_ind_end = [int(x) for x in new3_ind_end]
    
    # Keep indexes as arrays because we will index them
    # print('new3_ind_st : ' + str(new3_ind_st))
    # print('new3_ind_end : ' + str(new3_ind_end))
    # print('length of new3_ind_st : ' + str(len(new3_ind_st)))
    # print('length of new3_ind_end : ' + str(len(new3_ind_end)))
    # ------------------------------
        

    # ------------------------------
    # Plot ALL cabin actual data with START-STOP points found from looking
    # at where the summed cabin movement crosses zero
    # ------------------------------
    if plotORnot == 1:
        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        t = np.multiply(range(0, len(sum_sig)), 1)

        fig.add_trace(go.Scatter(x=t, y=sum_sig, name='sum_sig', line = dict(color='black', width=2, dash='dash'), showlegend=True))

        fig.add_trace(go.Scatter(x=t, y=cpos1, name='cpos1', line = dict(color='red', width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=t, y=cpos2, name='cpos2', line = dict(color='green', width=2, dash='dash'), showlegend=True))
        fig.add_trace(go.Scatter(x=t, y=cpos3, name='cpos3', line = dict(color='blue', width=2, dash='dash'), showlegend=True))

        plotb = baseline*np.ones((len(sum_sig)))
        fig.add_trace(go.Scatter(x=t, y=plotb))

        sum_sig = np.array(sum_sig)

        outval = np.array(new3_ind_st)
        #print('length of sum_sig : ' + str(len(sum_sig)))
        
        fig.add_trace(go.Scatter(x=outval, y=sum_sig[new3_ind_st], name='st pts', mode='markers', marker=dict(color='green', size=10, line=dict(width=0)), showlegend=True))

        outval1 = np.array(new3_ind_end)
        fig.add_trace(go.Scatter(x=outval1, y=sum_sig[new3_ind_end], name='end pts', mode='markers', marker=dict(color='magenta', size=10, line=dict(width=0)), showlegend=True))


        fig.update_layout(title='Actual cabin motion', xaxis_title='data points', yaxis_title='Cabin position')

        #fig.show()
        # OR
        fig.show(config=config)
    # ------------------------------


    return new3_ind_st, new3_ind_end
