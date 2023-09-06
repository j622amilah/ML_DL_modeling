import numpy as np

# Plotting
import plotly.graph_objects as go

# Data saving
import pickle

# Importing the statistics module
from statistics import mode, mean, median, multimode
import scipy.stats

# Creating folders for image or data saving
import os

# Personal python functions
from subfunctions.standarization_check_if_joy_moved import *
from subfunctions.generate_joy_move_sign import *
from subfunctions.filter_sig3axes import *
from subfunctions.make_a_properlist import *
from subfunctions.datadriven_FRT_vs_expmatFRT import *


def detection_response(varr, filemarker):

	# ------------------------------

	# 1) Load exp : put the experiment that you wish to run
	for exp in range(2):  # 0=rotation, 1=translation
		Xexp = []
		
		if exp == 0:
		    # Rotational data - 18 participants
		    varr['which_exp'] = 'rot'
		    varr['subjects'] = 'GV-123','AW-123','CDV-456','LB-000','PJ-123','PN-509','DL-777','SS-531','MD-565','CB-161','PI-112','FD-112','JMF-123','LB-195','LM-123','MBC-777','PB-123','SA-643'
		    varr['sub_nom_rot'] = 'GV','AW','CDV','LB','PJ','PN','DL','SS','MD','CB','PI','FD','JMF','LB','LM','MBC','PB','SA'
		    varr['anom'] = 'RO', 'PI', 'YA'
		    varr['joyanom'] = '1', '2', '3'  # Numbers for standarization, because we switch axes.  For pre-processing we set an axis like in anom.
		    varr['vals'] = 0.5, 1.25, 0
		    varr['data_path'] = '%s%sDATA_10Hz_rot' % (varr['main_path'], filemarker)

		    # 1) Orientation of joystick axes (Proven by standardization test)
		    a = 17-1  # labeled (PI) - joystick movement
		    b = 16-1   # labeled (RO) - joystick movement
		    c = 18-1   # labeled (UD) - joystick movement

		    # Load data experimental preprocessed data matrix
		    file_name = "rotdat.pkl"

		elif exp == 1:
		    # Translational data - 14 participants
		    varr['which_exp'] = 'trans'
		    varr['subjects'] = '20170411-TB-463','20172206-NB-777','20170413-GP-007','20172806-SM-308','20170404-AV-280','20170308-MK-160','20172008-AS-007','20170824-GL-380','p9_16102017-RW-115','p10_12102017-SG-123','p11_17102017-LG-123','p12_10102017-HS-000','p13_13102017-GB-666','p14_20171014-SL-132'
		    varr['sub_nom_rot'] = 'TB','NB','GP','SM','AV','MK','AS','GL','RW','SG','LG','HS','GB','SL'        
		    varr['anom'] = 'LR', 'FB', 'UD'
		    varr['joyanom'] = '1', '2', '3'
		    varr['vals'] = 3.75, 15, 0
		    varr['data_path'] = '%s%sDATA_10Hz_trans' % (varr['main_path'], filemarker)

		    # 1) Orientation of joystick axes (Proven by standardization test)
		    a = 16-1  # labeled (LR) - joystick movement
		    b = 17-1   # labeled (FB) - joystick movement
		    c = 18-1   # labeled (UD) - joystick movement

		    # Load data experimental preprocessed data matrix
		    file_name = "transdat.pkl"
		    
		    
		    
		file_dir_name = "%s%s%s" % (varr['main_path2'], filemarker, file_name)
		open_file = open(file_dir_name, "rb")
		dat = pickle.load(open_file)
		open_file.close()


		# 2) Load subjects
		subs = range(len(dat))
		subs = make_a_properlist(subs)
		print('subs : ', subs)
		
		for s in subs:
		    
		    num_of_tr = len(dat[s][0])      
		    
		    speed_stim_DD_out = dat[s][0]        # 0 : speed_stim (DD)
		    axis_out = dat[s][1]                # 1 : axis_out (DD)
		    
		    new3_ind_st = dat[s][2]             # 2 : new3_ind_st
		    new3_ind_end = dat[s][3]            # 3 : new3_ind_end
		    
		    outSIGCOM = dat[s][4]               # 4 : outSIGCOM
		    outSIG = dat[s][5]                 # 5 : outSIG
		    outJOY = dat[s][6]                 # 6 : outJOY
		    outNOISE = dat[s][7]               # 7 : outNOISE
		    
		    time = dat[s][8]                   # 8 : time_org
		    
		    trialnum_org = dat[s][9]           # 9 : trialnum_org
		    SSQ = dat[s][10]                    # 10 : SSQ
		    FRT_em = dat[s][11]                 # 11 : FRT



		    # ------------------------------
		    # compare FRT measured by experiment with FRT found by joystick
		    # ------------------------------
		    FRT, FRT_ax, FRT_dir = datadriven_FRT_vs_expmatFRT(num_of_tr, outJOY, time, FRT_em)
		    
		    # ------------------------------
		    # Detection response metrics:
		    # [1] response_type (diagram categorization) : (res_type : a number from 1 to 10 identifying participant response),
		    # [2] time response (TR) with values set logically (Reaction time) : TR_correct (if response never correct : TR_correct=last time point, if no response : TR=0, if response correct : TR=time_value)
		    # ------------------------------
		    response_type = []
		    TR = []
		    
		    
		    # --------------------
		    # Filter the outJOY signal to remove noise
		    plotORnot = 0
		    outJOY_filter = filter_sig3axes(outJOY, plotORnot)
		    # --------------------
		    
		    # --------------------
		    # Find the sign of speed stim per subject
		    speed_stim_sign_dd = [np.sign(x) for x in speed_stim_DD_out] 
		    # --------------------

		    for tr in range(num_of_tr):
		    
		        print('tr : ' + str(tr))
		        
		        # Find first joystick movement point
		        marg_joy = 0.1
		        # joy_ax_dir, joy_ax_val, joy_ax_index = standarization_check_if_joy_moved(tr, outJOY, marg_joy)
		        joy_ax_dir, joy_ax_val, joy_ax_index = standarization_check_if_joy_moved(tr, outJOY_filter, marg_joy)
		        print('joy_ax_dir : ' + str(joy_ax_dir))
		        print('joy_ax_val : ' + str(joy_ax_val))
		        print('joy_ax_index : ' + str(joy_ax_index))
		        
		        # NOTE : the length of win_joy_sign is a little short than the length of outJOY because binning may not be
		        # equally divided by the length of the signal
		        win_joy_sign, win_joy_ax = generate_joy_move_sign(tr, outJOY_filter, axis_out)
		        dp_len_pertr = len(win_joy_sign)
		        # print('dp_len_pertr : ' + str(dp_len_pertr))
		        # ------------------------------
		        
		        
		        
		        
		        if speed_stim_sign_dd[tr] == 0:
		            # Sham : axis can be 1,2,3, speed = 0
		            
		            if sum(joy_ax_index) == 0:
		                # YES : Participant DID NOT RESPONSE
		                # print('YES : Participant DID NOT RESPONSE')
		                
		                res_type = 8
		                Correct = 1
		                TR_correct = 0
		            else:
		                # NO : Participant RESPONSED
		                # It is a sham trial and person moved joystick: they are wrong and timing=when they moved the joystick
		                res_type = 10
		                Correct = 0
		                TR_correct = FRT[tr]
		        else:
		            # Movement : axis can be 1,2,3, speed = sub, sup

		            # Are there any axes initially DETECTED?
		            if sum(joy_ax_index) == 0:
		                # NO : Participant DID NOT RESPONSE
		                # print('NO : Participant DID NOT RESPONSE')
		                # It is a movement trial and the person did not move joystick : they are wrong and timing is zero
		                res_type = 9
		                Correct = 0
		                TR_correct = 0
		            else:
		                # YES : Participant responded on 1 or more axes
		                # print('YES : Participant responded on 1 or more axes')
		                
		                # Is axis CORRECT ?
		                # print('axis_out[tr] : ' + str(axis_out[tr]))
		                
		                # print('FRT_ax[tr] : ' + str(FRT_ax[tr]))
		                
		                if axis_out[tr] == FRT_ax[tr]:
		                    # YES : initial axis CORRECT
		                    # print('YES : initial axis CORRECT')

		                    # Is direction initially CORRECT ?
		                    if speed_stim_sign_dd[tr] == FRT_dir[tr]:

		                        # YES : initial direction CORRECT
		                        # print('YES : initial direction CORRECT')
		                        
		                        # print('FRT_dir[tr] : ' + str(FRT_dir[tr]))
		                        
		                        # person finds correct axis, and INITIALLY finds correct direction
		                        res_type = 1
		                        Correct = 1
		                        TR_correct = FRT[tr]
		                    else:
		                        # NO : initial direction WRONG
		                        # print('NO : initial direction WRONG')
		                        # Did the direction eventually become CORRECT (direction search) ?
		                        
		                        # Here FRT_dir is not equal to the stimulus axis 
		                        # So we need to get the joystick direction movement of the stimulus axis (axis_out)
		                        
		                        # Then, search across the trial until the joystick direction is opposite to the stimulus direction because  (cab = -joy)
		                        for dp in range(dp_len_pertr):
		                            if (win_joy_sign[dp] == -speed_stim_sign_dd[tr]):
		                                # person finds correct axis, and eventually finds correct direction
		                                res_type = 2
		                                Correct = 1
		                                TR_correct = time[tr][dp]  # time when detected
		                                break
		                            else:
		                                # Person finds correct axis, but never finds correct direction
		                                res_type = 3
		                                Correct = 0
		                                TR_correct = time[tr][-1] # finial trial time

		                elif axis_out[tr] != FRT_ax[tr]:
		                    # NO : initial axis WRONG
		                    # print('NO : initial axis WRONG')

		                    # Did the axis eventually become CORRECT (axis search) ? (we decouple 'axis eventually correct' and 'direction eventually correct')
		                    
		                    flagger = 1
		                    for dp in range(dp_len_pertr):
		                        if (win_joy_ax[dp] == 1):
		                            # print('YES : eventual axis CORRECT')
		                            # person eventually moved the joystick on the correct axis
		                            
		                            # Is direction initially CORRECT ?
		                            if (win_joy_sign[dp] == -speed_stim_sign_dd[tr]):
		                                # print('YES : initial direction CORRECT')
		                                # person eventually finds correct axis, and initially finds correct direction
		                                res_type = 4
		                                Correct = 1
		                                TR_correct = time[tr][dp]  # time when detected
		                                break
		                            else:
		                                # NO : initial direction WRONG
		                                # print('NO : initial direction WRONG')
		                                
		                                # Did the direction eventually become CORRECT ?
		                                for dp2 in range(dp, dp_len_pertr):
		                                    
		                                    if (win_joy_sign[dp2] == -speed_stim_sign_dd[tr]):
		                                        # YES : eventual direction CORRECT
		                                        # print('YES : eventual direction CORRECT')
		                                        # Person eventually finds correct axis, and eventually finds correct direction
		                                        res_type = 5
		                                        Correct = 1
		                                        TR_correct = time[tr][dp2-1]  # time when detected
		                                        flagger = 0
		                                        break
		                                    else:
		                                        # NO : CORRECT direction was never found
		                                        # print('NO : CORRECT direction was never found')
		                                        # Person eventually finds correct axis, but never finds correct direction
		                                        res_type = 6
		                                        Correct = 0
		                                        TR_correct = time[tr][-1] # finial trial time
		                                    
		                                    # To stop the dp for loop because the axis was found, but the direction was never found
		                                    flagger = 0
		                        else:
		                            # NO : never found the correct axis
		                            # print('NO : never found the correct axis')
		                            res_type = 7
		                            Correct = 0
		                            TR_correct = time[tr][-1] # finial trial time
		                            
		                        # To stop the dp for loop if eventual direction is CORRECT
		                        # the single break only stops the dp2 for loop, but not the dp for loop.
		                        # And, we do no want to break the tr for loop, so we use a flag to stop the dp for loop.
		                        if flagger == 0:
		                            break
		        # ------------------------------
		        
		        print('res_type : ' + str(res_type))
		        print('Correct : ' + str(Correct))
		        print('TR_correct : ' + str(TR_correct))
		        
		        response_type = response_type + [res_type]
		        TR = TR + [TR_correct]

		    
		    # ------------------------------
		    # Put metrics in a matrix
		    # ------------------------------
		    # Concatenate trial vector per subject : vector need to be column vectors
		    e0 = s*np.ones((num_of_tr,1))

		    e1 = np.reshape(range(num_of_tr), (num_of_tr,1))

		    e2 = np.reshape(trialnum_org, (num_of_tr,1))  # tr_num_org : To know if experiment was administered at the start or end : could influence response due to learning or fatigue
		    
		    # print('axis_out : ' + str(axis_out))
		    e3 = np.reshape(axis_out, (num_of_tr,1))
		    
		    # print('speed_stim_DD_out : ' + str(speed_stim_DD_out))
		    e4 = np.reshape(speed_stim_DD_out, (num_of_tr,1))
		    
		    e5 = np.reshape(response_type, (num_of_tr,1))
		    e6 = np.reshape(TR, (num_of_tr,1))
		    
		    X_row = np.ravel(e0), np.ravel(e1), np.ravel(e2), np.ravel(e3), np.ravel(e4), np.ravel(e5), np.ravel(e6)
		    
		    Xsub = np.transpose(X_row)
		    

		    Xexp = Xexp + [Xsub]
		
		
		# ------------------------------
		# Save data matrices to file per experiment
		file_name = "%s_Xexp.pkl" % (varr['which_exp'])
		file_dir_name = "%s%s%s" % (varr['main_path3'], filemarker, file_name)
		open_file = open(file_dir_name, "wb")
		pickle.dump(Xexp, open_file)
		open_file.close()
		del Xexp
		# ------------------------------

	return
