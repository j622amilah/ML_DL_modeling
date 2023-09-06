import numpy as np


def full_sig_2_cell(A, a, b, c, ind_st, ind_end, varr):
    
    # ------------------------------
    # Put the joystick, cabin actual, and cabin command into a compact format: 
    # a 3D list (depth=trials, row=dp per trial, col=axis in order LR/RO, FB/PI, UD/YAW)
    # ------------------------------
    outJOY = []
    outSIG = []
    outSIGCOM = []
    outNOISE = []
    
    for tr in range(len(ind_st)):    # number of trials
        
        lendat = len(A[:,5:6])        
        st = int(ind_st[tr])
        stp = int(ind_end[tr])
        
        # NOTE : it may seem like a lot of transformations between row and column vectors,
        # but we use the temp dictionary to store joy, cpos, and ccom because it is easier to
        # remember the entries than the A matrix (remove possibility to make a mistake)
        # and in matlab I could delete A (so less data in RAM)
        
        # ------------------------------
        # Joystick
        # ------------------------------
        temp_list1a = A[:,a:a+1]
        temp_list1a1 = np.reshape(temp_list1a, (1,lendat))  # Transform to a row vector, is a list in a list
        temp_list1a2_row = temp_list1a1[0][st:stp]  # Get the list, then start-stop index row vector
        lendat_cut = len(temp_list1a2_row)
        temp_list1a2_col = np.reshape(temp_list1a2_row, (lendat_cut, 1)) # start-stop indexed column vector
        
        temp_list1b = A[:,b:b+1]
        temp_list1b1 = np.reshape(temp_list1b, (1,lendat))
        temp_list1b2_row = temp_list1b1[0][st:stp] 
        temp_list1b2_col = np.reshape(temp_list1b2_row, (lendat_cut, 1))
        
        temp_list1c = A[:,c:c+1]
        temp_list1c1 = np.reshape(temp_list1c, (1,lendat))
        temp_list1c2_row = temp_list1c1[0][st:stp]
        temp_list1c2_col = np.reshape(temp_list1c2_row, (lendat_cut, 1))
        
        # ------------------------------
        # Cabin actual
        # ------------------------------
        temp_list2a = A[:,5:6]  # RO/LR 
        temp_list2a1 = np.reshape(temp_list2a, (1,lendat))
        temp_list2a2_row = temp_list2a1[0][st:stp]
        temp_list2a2_col = np.reshape(temp_list2a2_row, (lendat_cut, 1))
        
        #temp_list2b = temp['cpos'+varr['anom'][1]]
        temp_list2b = A[:,6:7]  # PI/FB
        temp_list2b1 = np.reshape(temp_list2b, (1,lendat))
        temp_list2b2_row = temp_list2b1[0][st:stp]
        temp_list2b2_col = np.reshape(temp_list2b2_row, (lendat_cut, 1))
        
        #temp_list2c = temp['cpos'+varr['anom'][2]]
        temp_list2c = A[:,7:8]  # YA/UD
        temp_list2c1 = np.reshape(temp_list2c, (1,lendat))
        temp_list2c2_row = temp_list2c1[0][st:stp]
        temp_list2c2_col = np.reshape(temp_list2c2_row, (lendat_cut, 1))
        
        # ------------------------------
        # Cabin command
        # ------------------------------
        temp_list3a = A[:,2:3]  # RO/LR
        temp_list3a1 = np.reshape(temp_list3a, (1,lendat))
        temp_list3a2_row = temp_list3a1[0][st:stp]
        temp_list3a2_col = np.reshape(temp_list3a2_row, (lendat_cut, 1))
        
        temp_list3b = A[:,3:4]  # PI/FB   
        temp_list3b1 = np.reshape(temp_list3b, (1,lendat))
        temp_list3b2_row = temp_list3b1[0][st:stp]
        temp_list3b2_col = np.reshape(temp_list3b2_row, (lendat_cut, 1))
        
        temp_list3c = A[:,4:5]  # YA/UD
        temp_list3c1 = np.reshape(temp_list3c, (1,lendat))
        temp_list3c2_row = temp_list3c1[0][st:stp]
        temp_list3c2_col = np.reshape(temp_list3c2_row, (lendat_cut, 1))
        
        
        # ------------------------------
        # Noise
        # ------------------------------
        temp_list4a = A[:,20-1]  # RO/LR
        temp_list4a1 = np.reshape(temp_list4a, (1,lendat))
        temp_list4a2_row = temp_list4a1[0][st:stp]
        temp_list4a2_col = np.reshape(temp_list4a2_row, (lendat_cut, 1))
        
        temp_list4b = A[:,21-1]  # PI/FB   
        temp_list4b1 = np.reshape(temp_list4b, (1,lendat))
        temp_list4b2_row = temp_list4b1[0][st:stp]
        temp_list4b2_col = np.reshape(temp_list4b2_row, (lendat_cut, 1))
        
        temp_list4c = A[:,22-1]  # YA/UD
        temp_list4c1 = np.reshape(temp_list4c, (1,lendat))
        temp_list4c2_row = temp_list4c1[0][st:stp]
        temp_list4c2_col = np.reshape(temp_list4c2_row, (lendat_cut, 1))
        
        
        # Assemble cut trial matricies
        temp_list1_trmat = np.concatenate((np.concatenate((temp_list1a2_col, temp_list1b2_col), axis=1), temp_list1c2_col), axis=1)
        temp_list2_trmat = np.concatenate((np.concatenate((temp_list2a2_col, temp_list2b2_col), axis=1), temp_list2c2_col), axis=1)
        temp_list3_trmat = np.concatenate((np.concatenate((temp_list3a2_col, temp_list3b2_col), axis=1), temp_list3c2_col), axis=1)
        temp_list4_trmat = np.concatenate((np.concatenate((temp_list4a2_col, temp_list4b2_col), axis=1), temp_list4c2_col), axis=1)
        
        # print('size of an axis trial block in outSIGCOM : ' + str(temp_list3_trmat.shape))
        
        # Stack trial matrices of [0:tr_dplen by 3]
        outJOY = outJOY + [temp_list1_trmat]
        outSIG = outSIG + [temp_list2_trmat]
        outSIGCOM = outSIGCOM + [temp_list3_trmat]
        outNOISE = outNOISE + [temp_list4_trmat]
    
    #outSIGCOM = np.array(outSIGCOM)
    # print('Size of outSIGCOM : ' + str(outSIGCOM.shape))
    
    return outJOY, outSIG, outSIGCOM, outNOISE