import numpy as np

# Plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Data saving
import pickle

# Importing the statistics module
from statistics import mode, mean, median, multimode
import scipy.stats

# Personal python functions
from subfunctions.check_axes_assignmentPLOT import *
from subfunctions.cut_initial_trials import *

from subfunctions.findall import *

from subfunctions.main_preprocessing_steps import *
from subfunctions.make_a_properlist import *

from subfunctions.size import *
from subfunctions.standarization_cabanaORnexttr import *
from subfunctions.standarization_check_if_joy_moved import *

# Creating folders for image or data saving
import os



def just_stat_joy_cab_directional_ctrl(varr):


    # ------------------------------

    # Statistical justification of joystick and cabin directional control for both rotation and translation
    NUMmark = 0   #  an arbitrarily large number to denote a non-valid entry
    NOjoyMOVE = 70  #  an arbitrarily number to denote the joystick was not moved in final X matrix

    fs = 10 #  Original experimental sampling was at 250Hz, but data is downsampled to 10Hz
    ts = 1/fs

    mj_val = [0.1, 0.05, 0.01]

    # ------------------------------------------------------------

    # 1) Load exp : put the experiment that you wish to run
    for exp in range(2):  # 0=rotation, 1=translation

        if exp == 0:
            # Rotational data - 18 participants
            varr['which_exp'] = 'rot'
            varr['subjects'] = 'GV-123','AW-123','CDV-456','LB-000','PJ-123','PN-509','DL-777','SS-531','MD-565','CB-161','PI-112','FD-112','JMF-123','LB-195','LM-123','MBC-777','PB-123','SA-643'
            varr['sub_nom_rot'] = 'GV','AW','CDV','LB','PJ','PN','DL','SS','MD','CB','PI','FD','JMF','LB','LM','MBC','PB','SA'
            varr['anom'] = 'RO', 'PI', 'YA'
            varr['joyanom'] = '1', '2', '3'  # Numbers for standarization, because we switch axes.  For pre-processing we set an axis like in anom.
            varr['vals'] = 0.5, 1.25, 0
            # varr['data_path'] = '%s\\DATA_10Hz_rot' % (varr['main_path'])  # Windows
            varr['data_path'] = '%s/DATA_10Hz_rot' % (varr['main_path'])
        elif exp == 1:
            # Translational data - 14 participants
            varr['which_exp'] = 'trans'
            varr['subjects'] = '20170411-TB-463','20172206-NB-777','20170413-GP-007','20172806-SM-308','20170404-AV-280','20170308-MK-160','20172008-AS-007','20170824-GL-380','p9_16102017-RW-115','p10_12102017-SG-123','p11_17102017-LG-123','p12_10102017-HS-000','p13_13102017-GB-666','p14_20171014-SL-132'
            varr['sub_nom_rot'] = 'TB','NB','GP','SM','AV','MK','AS','GL','RW','SG','LG','HS','GB','SL'        
            varr['anom'] = 'LR', 'FB', 'UD'
            varr['joyanom'] = '1', '2', '3'
            varr['vals'] = 3.75, 15, 0
            # varr['data_path'] = '%s\\DATA_10Hz_trans' % (varr['main_path'])  # Windows
            varr['data_path'] = '%s/DATA_10Hz_trans' % (varr['main_path'])


        # 2) Load subjects
        subs = range(len(varr['subjects']))
        subs = make_a_properlist(subs)

        for s in subs:
            print('s : ' + str(s))
            
            # ------------------------------
            # (1) Load data
            # A = np.loadtxt("%s\\%s.txt" % (varr['data_path'], varr['subjects'][s]))  # Windows
            A = np.loadtxt("%s/%s.txt" % (varr['data_path'], varr['subjects'][s]))

            # print('Size of A matrix : ' + str(size(A)))     # result(row00=9445, col00=22)
            # ------------------------------
            
            
            # 3) Orientation of axes - 6 combinations of joystick axe
            for orax in range(6):
                print('orax : ' + str(orax))
                
                if orax == 0:     # [1, 2, 3]  Translation orientation
                    a = 16-1  # labeled (PI/LR) - joystick movement
                    b = 17-1   # labeled (RO/FB) - joystick movement
                    c = 18-1   # labeled (YA/UD) - joystick movement
                elif orax == 1:   # [1, 3, 2]
                    a = 16-1
                    b = 18-1
                    c = 17-1
                elif orax == 2:   # [2, 1, 3]  Rotation orientation
                    a = 17-1
                    b = 16-1
                    c = 18-1
                elif orax == 3:   # [2, 3, 1]
                    a = 17-1
                    b = 18-1
                    c = 16-1
                elif orax == 4:   # [3, 1, 2]
                    a = 18-1
                    b = 16-1
                    c = 17-1
                elif orax == 5:   # [3, 2, 1]
                    a = 18-1
                    b = 17-1
                    c = 16-1
                    
                
                for yr in range(2):
                    
                    # ------------------------------
                    # There was confusion if yaw joystick direction was reversed
                    yr = 0  # 0=keep the sign of yaw/UD joystick, 1=reverse the sign of yaw/UD joystick
                    # ------------------------------
                    
                    
                    # ------------------------------
                    # a) Pre-process the data using the selected orax, dirC, and mj_val 
                    starttrial_index, stoptrial_index, speed_stim_sign, speed_stim_mag, speed_stim_org, axis_out, axis_org, new3_ind_st, new3_ind_end, g_rej_vec, outJOY, outSIG, outSIGCOM, outNOISE, corr_axis_out, corr_speed_stim_out, trialnum_org, time_org, FRT, good_tr = main_preprocessing_steps(varr, A, a, b, c, s, NUMmark, yr)
                    # ------------------------------

                    # ------------------------------
                    # Check if axes are assigned correctly to trials: PLOTTING final plot
                    # ------------------------------
                    plotORnot = 1  # 1 = show figures, 0 = do not show figures
                    if plotORnot == 1:
                        filename = 'images_cutfinal_%s' % (varr['which_exp'])
                        check_axes_assignmentPLOT(s, outJOY, outSIG, axis_out, varr, filename, time_org)
                    # ------------------------------
                    
                        
                    # 4) Load directional control orientation
                    for dirC in range(2):  # (2) joystick-cabin directional control orientation [ (joy + cab = u) vs (-joy + cab = u) ]
                        print('dirC : ' + str(dirC))
                        
                        
                        # 5) Load deadzone margin
                        for mj in range(len(mj_val)):  # (3) deadzone margin for the joystick: when joystick depassed this value the cabin moved   
                            print('mj : ' + str(mj))
                            
                            
                            for strict in range(4):
                                if strict == 0: 
                                    strness = [0, 0]   # [movement, direction] : 0=Lenient, 1=Strict
                                elif strict == 1:
                                    strness = [0, 1]   # [movement, direction] : 0=Lenient, 1=Strict
                                elif strict == 3:
                                    strness = [1, 0]   # [movement, direction] : 0=Strict, 1=Lenient
                                elif strict == 4:
                                    strness = [1, 1]   # [movement, direction] : 0=Strict, 1=Strict
                                print('strict : ' + str(strict))
                                
                            
                                # Deadzone = 0.1  - Rotation and Translation orientation
                                marg_joy = mj_val[mj]
                            
                            
                                # ------------------------------
                                # Start Joystick analysis
                                # ------------------------------
                                # b) Only look at non-defective trials and initalize
                                # num_of_trials should be equal to len(idx_alltr) I think

                                # print('axis_out :' + str(axis_out))
                                # print('starttrial_index :' + str(starttrial_index))
                                # print('stoptrial_index :' + str(stoptrial_index))

                                nimp_val, idx_alltr = findall(stoptrial_index, '!=', 0)

                                # print('idx_alltr :' + str(idx_alltr))

                                # c) Start loops over non-defective trials: search for when/what direction joystick and cabin moved
                                num_of_tr = len(idx_alltr)
                                # print('Number of trials to standardize :' + str(num_of_tr))
                                
                                # Initialize normalized_outSIG
                                # Note : normalized_outSIG and outSIG need to be the same organization (list - array - list - list)
                                normalized_outSIG = []
                                for tr in range(len(outSIG)):
                                    if not outSIG[tr].any():  # if outSIG[tr] is empty = False
                                        # You can not put an empty list because python will not append an empty list in a list 
                                        normalized_outSIG = normalized_outSIG + [np.zeros((1,3))]
                                    else:
                                        normalized_outSIG = normalized_outSIG + [np.zeros((len(outSIG[tr]),3))]
                                
                                # print('length of outSIG : ' + str(len(outSIG)))
                                # print('length of normalized_outSIG : ' + str(len(normalized_outSIG)))
                                
                                tr_c = 0
                                
                                binary_marker = np.zeros((num_of_tr,1))
                                scalar_marker = np.zeros((num_of_tr,1))
                                direction_marker = np.zeros((num_of_tr,1))
                                dir_meaning_marker = np.zeros((num_of_tr,1))  # direction marker meaning wrt binary(joy-cab follow)
                            
                            
                                for tr in idx_alltr:
                                    # print('tr :' + str(tr))
                                    
                                    # Step 1 : check if joystick moved
                                    joy_ax_dir, joy_ax_val, joy_ax_index = standarization_check_if_joy_moved(tr, outJOY, marg_joy)
                                    
                                    # Step 2 : Based on joystick movement: 1) continue analysis with cabin, 2) stop analysis --> Next trial
                                    binary_marker, scalar_marker, direction_marker, dir_meaning_marker = standarization_cabanaORnexttr(joy_ax_index, joy_ax_dir, binary_marker, scalar_marker, direction_marker, dir_meaning_marker, tr_c, NOjoyMOVE, outSIG, outJOY, tr, normalized_outSIG, marg_joy, dirC, axis_out, varr, s, strness[0], strness[1])
                                    
                                    tr_c = tr_c + 1
                               
                                
                                
                                # ------------------------------
                                # Save results per subject and condition
                                # ------------------------------
                                # Reason for separate files : the data is divided per subject because my computer can not quickly
                                # calculate with all the data of previous subjects in the RAM
                                # In column form
                                e0 = s*np.ones((num_of_tr,1))
                                e1 = orax*np.ones((num_of_tr,1))
                                e2 = yr*np.ones((num_of_tr,1))
                                e3 = dirC*np.ones((num_of_tr,1))
                                e4 = marg_joy*np.ones((num_of_tr,1))
                                e5 = binary_marker
                                e6 = scalar_marker
                                e7 = direction_marker
                                e8 = dir_meaning_marker
                                e9 = strict*np.ones((num_of_tr,1))
                                
                                # Way 1:
                                # first2 = np.concatenate((e0, e1), axis=1)
                                # next2_0 = np.concatenate((first2, e2), axis=1)
                                # next2_1 = np.concatenate((next2_0, e3), axis=1)
                                # next2_2 = np.concatenate((next2_1, e4), axis=1)
                                # next2_3 = np.concatenate((next2_2, e5), axis=1)
                                # next2_4 = np.concatenate((next2_3, e6), axis=1)
                                # X = np.concatenate((next2_4, e7), axis=1)
                                
                                # Way 2:
                                X_row = np.ravel(e0), np.ravel(e1), np.ravel(e2), np.ravel(e3), np.ravel(e4), np.ravel(e5), np.ravel(e6), np.ravel(e7), np.ravel(e8), np.ravel(e9)
                                X = np.transpose(X_row)
                                
                                # print('X : ' + str(X))
                                
                                # create a directory for saving data
                                filename = 'data_standard_%s' % (varr['which_exp'])
                                
                                if not os.path.exists("%s/%s" % (varr['main_path1'], filename)):
                                    os.mkdir("%s/%s" % (varr['main_path1'], filename))
                                # if not os.path.exists("%s\\%s" % (varr['main_path1'], filename)):
                                #     os.mkdir("%s\\%s" % (varr['main_path1'], filename))
                                
                                # First try saving a matrix per condition set : Save data matrices to file
                                file_name = "%s/%s/s%d_orax%d_yr%d_dirC%d_marg%d_st%d_outSIG.pkl" % (varr['main_path1'], filename, s, orax, yr, dirC, mj, strict)
                                # file_name = "%s\\%s\\s%d_orax%d_yr%d_dirC%d_marg%d_st%d_outSIG.pkl" % (varr['main_path1'], filename, s, orax, yr, dirC, mj, strict)
                                open_file = open(file_name, "wb")
                                pickle.dump(X, open_file)
                                open_file.close()
                                
                                
                                del binary_marker, scalar_marker, direction_marker, dir_meaning_marker
                                del joy_ax_dir, joy_ax_val, joy_ax_index, tr_c
                                del e0,e1, e2, e3, e4, e5, e6, e7, e8, e9, X_row, X
                                
                            # end of strict
                            
                        # end of mj        
                        
                    # end of dirC
                    
                # end of yr
                
            # end of orax 
            
        # end of s
        
    # end of exp
    
    return
