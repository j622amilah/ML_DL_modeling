import numpy as np
import pandas as pd

# Plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

# Data saving
import pickle

# Importing the statistics module
from statistics import mode, mean, median, multimode
import scipy.stats

# module for counting values in a list
from collections import Counter

from sklearn.cluster import KMeans

# Personal python functions
from subfunctions.check_axes_assignmentPLOT import *
from subfunctions.cut_initial_trials import *
from subfunctions.findall import *
from subfunctions.main_preprocessing_steps import *
from subfunctions.make_a_properlist import *
from subfunctions.size import *
from subfunctions.standarization_notebadtrials import *
from subfunctions.saveSSQ import *
from subfunctions.check_axes_assignmentPLOT_final import *

# Creating folders for image or data saving
import os



def preprocessing_pipeline(varr, plot_sub_trials, filemarker, plotORnot_ALL, plot_sub_trials_FINAL):

    # ------------------------------

	# Selecting a method of removing bad trials from the data
	# Saving the data per participant per experiment

	NUMmark = 0   #  an arbitrarily large number to denote a non-valid entry

	mj_val = [0.1, 0.05, 0.01]

	tr_type_sub_exp = []

	# ------------------------------

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
		   	
			varr['data_path'] = '%s%sDATA_10Hz_rot' % (varr['main_path'], filemarker)
			
			# 1) Orientation of joystick axes (Proven by standardization test)
			a = 17-1  # labeled (PI) - joystick movement
			b = 16-1   # labeled (RO) - joystick movement
			c = 18-1   # labeled (YA) - joystick movement
			
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
			
		
		
		datout_temp = []
		
		# ------------------------------
		
		# 2) Directional control orientation :
		# Convention 1 : (cab = cab_com - joystick), thus (cab = -joy)
		# (Proven by standardization test)
		dirC = 0
		
		# ------------------------------
		
		# 3) Deadzone margin : 0.1 (Proven by standardization test)
		marg_joy = 0.1

		# ------------------------------

		# 4) Load subjects
		subs = np.arange(len(varr['subjects']))
		
		exp_reject_cat_vec = []
		exp_acc_tr_per = []
		
		tr_type_sub = []
		
		for s in subs:
			print('s : ' + str(s))

			# ------------------------------
			# (1) Load data
			A = np.loadtxt("%s%s%s.txt" % (varr['data_path'], filemarker, varr['subjects'][s]))

			# print('Size of A matrix : ' + str(size(A)))     # result(row00=9445, col00=22)
			# ------------------------------
			
			
			# ------------------------------
			# There was confusion if yaw joystick direction was reversed
			yr = 0  # 0=keep the sign of yaw/UD joystick, 1=reverse the sign of yaw/UD joystick
			# ------------------------------
			
			
			# ------------------------------
			# a) Pre-process the data using the selected orax, dirC, and mj_val 
			starttrial_index, stoptrial_index, speed_stim_org_sign, speed_stim_org, tar_ang_speed_val_co, speed_stim_tas_sign, axis_out, axis_org, new3_ind_st, new3_ind_end, g_rej_vec, outJOY, outSIG, outSIGCOM, outNOISE, corr_axis_out, trialnum_org, time_org, FRT, good_tr = main_preprocessing_steps(varr, A, a, b, c, s, NUMmark, yr, plotORnot_ALL, filemarker)
			# ------------------------------

			dt = time_org[0][1] - time_org[0][0]
			fs = 1/dt

			# ------------------------------
			# Check if axes are assigned correctly to trials: PLOTTING
			# ------------------------------
			if plot_sub_trials == 1:
			    filename = 'images_cutfinal_%s' % (varr['which_exp'])
			    check_axes_assignmentPLOT(s, outJOY, outSIG, axis_out, varr, filename, time_org)
			# ------------------------------
			
			
			# ------------------------------
			# Note trial to remove based on STANDARDIZATION
			# ------------------------------
			# If cabin does not follow the cabin in the correct direction, mark trial for removal
			# Same analysis as Standardization test 
			strictness = 1  # 0=Lenient (majority of axes correct), 1=Strict (all axes correct)
			if strictness == 0:
			    strictness_text = 'ln_maj_ax'
			elif strictness == 1:
			    strictness_text = 'st_all_ax'
			cut_trial_standard_move, cut_trial_standard_dir = standarization_notebadtrials(starttrial_index, stoptrial_index, axis_out, outSIG, outJOY, marg_joy, varr, s, strictness, dirC, good_tr)
			
			# print('cut_trial_standard_move : ' + str(cut_trial_standard_move))
			# print('cut_trial_standard_dir : ' + str(cut_trial_standard_dir))
			
			# Decide to remove trials by joy->cabin movement or joy->cabin movementANDdirection
			strictness1 = 1  # 0=Lenient (joy->cabin movement), 1=Strict (joy->cabin movementANDdirection)
			
			if strictness1 == 0:
			    strictness1_text = 'ln_mov'
			    cut_trial_standard = cut_trial_standard_move
			elif strictness1 == 1:
			    strictness1_text = 'st_movANDdir'
			    cut_trial_standard = np.unique([cut_trial_standard_move, cut_trial_standard_dir])
			
			cut_trial_standard = [int(x) for x in cut_trial_standard] 
			g_rej_vec[cut_trial_standard] = 50
			
			# ------------------------------
			# Quick statistical analysis of how many trials were removed and why
			# ------------------------------
			cou = Counter(g_rej_vec)
			if exp == 0:
			    # 30 = cut_trial_ver_short
			    # 40 = cut_trial_hor_short
			    # 50 = cut_trial_standard
			    # bar_title = ['ver_short', 'hor_short', 'standardization']
			    reject_cat_vec = [cou[30], cou[40], cou[50]]
			elif exp == 1:
			    # 10 = robotjump_cutlist
			    # 15 = robotstall_cutlist
			    # 20 = FB_nonzero_start
			    # 25 = UD_initialization
			    # 30 = cut_trial_ver_short
			    # 40 = cut_trial_hor_short
			    # 50 = cut_trial_standard
			    # bar_title = ['robotjump', 'robotstall', 'FB_nonzero_start', 'UD_initialization', 'ver_short', 'hor_short', 'standardization']
			    reject_cat_vec = [cou[10], cou[15], cou[20], cou[25], cou[30], cou[40], cou[50]]
			
			
			# print('g_rej_vec : ' + str(g_rej_vec))
			print('reject_cat_vec : ' + str(reject_cat_vec))
			
			# Must be a column vector
			colvec = np.reshape(reject_cat_vec, (len(reject_cat_vec),1))
			acc_tr = cou[0]
			acc_tr_per = acc_tr/len(g_rej_vec)
			# fig = px.bar(colvec, title="s=%d, %s, acc tr per=%f" % (s, varr['which_exp'], acc_tr_per))
			# fig.show()
			
			exp_reject_cat_vec = exp_reject_cat_vec + [reject_cat_vec]
			exp_acc_tr_per = exp_acc_tr_per + [acc_tr_per]
			# ------------------------------
			
			
			# ------------------------------
			# j06/m01/a22 : need better statistic on how many removed trials : save g_rej_vec
			# ------------------------------
			tr_type_sub.append(g_rej_vec)
			
			
			# ------------------------------
			# Check if axes are assigned correctly to trials: PLOTTING
			# All trials, it says the type of rejection on the plot.
			# ------------------------------
			if plot_sub_trials_FINAL == 1:
			    filename = 'images_cutfinal_%s' % (varr['which_exp'])
			    check_axes_assignmentPLOT_final(s, outJOY, outSIG, axis_out, varr, filename, time_org, g_rej_vec)
			# ------------------------------
			
			# ------------------------------
			# Get list of good trials to save
			# ------------------------------
			newvec, good_tr = findall(g_rej_vec, '==', 0)
			good_tr_fin = [int(x) for x in good_tr]
			
			
			# ------------------------------
			# Save good trial subject data per experiment 
			# ------------------------------
			subdat = []
			
			speed_stim_org_sign = np.array(speed_stim_org_sign)
			e0 = speed_stim_org_sign[good_tr_fin] # experimental matrix direction
			
			speed_stim_tas_sign = np.array(speed_stim_tas_sign)
			e1 = speed_stim_tas_sign[good_tr_fin] # experimental matrix direction
			
			speed_stim_org = np.array(speed_stim_org)
			e2 = speed_stim_org[good_tr_fin]  # experimental matrix stimulus magnitude and direction
			
			tar_ang_speed_val_co = np.array(tar_ang_speed_val_co)
			e3 = tar_ang_speed_val_co[good_tr_fin] # experimental matrix stimulus magnitude and direction
			
			axis_out = np.array(axis_out)
			e4 = axis_out[good_tr_fin]  # data-driven measured axis
			
			axis_org = np.array(axis_org)
			e5 = axis_org[good_tr_fin] # experimental matrix axis

			# correlation between axis value in experimental matrix in comparison to axis measured from data via start-stop index
			e6 = corr_axis_out

			# start-stop trial cut index
			new3_ind_st = np.array(new3_ind_st)
			e7 = new3_ind_st[good_tr_fin]
			
			new3_ind_end = np.array(new3_ind_end)
			e8 = new3_ind_end[good_tr_fin]

			# Parameters with 3 axis
			e9 = []
			e10 = []
			e11 = []
			e12 = []
			e13 = []
			for tr in good_tr_fin:
			    e9 = e9 + [outSIGCOM[tr]]  # a list (num of dp, 3) per trial
			    e10 = e10 + [outSIG[tr]]  # a list (num of dp, 3) per trial
			    e11 = e11 + [outJOY[tr]]  # a list (num of dp, 3) per trial
			    e12 = e12 + [outNOISE[tr]]  # a list (num of dp, 3) per trial
			    e13 = e13 + [time_org[tr]]  # a list (num of dp, 1) per trial
			
			trialnum_org = np.array(trialnum_org)
			e14 = trialnum_org[good_tr_fin]
			
			# Each [] is a 4x1 vector=[nausee, oculo_moteur, disorientation, sickness]
			e15 = saveSSQ(s, varr['which_exp']) # SSQ [[before], [after], [diff=before-after]]
			
			e16 = FRT[good_tr_fin]
			
			subdat = e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16
			
			datout_temp.append(subdat)
			
		# end of s
		# ------------------------------
		
		
		# ------------------------------
		# Correlation study 1 : to determine if data driven axis measure is similar to experimental matrix axis : across subjects
		corr_axis_out_exp = [datout_temp[i][6] for i in range(len(datout_temp))]
	
		fig = go.Figure()
		config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
		xxORG = list(range(len(corr_axis_out_exp)))
		fig.add_trace(go.Scatter(x=xxORG, y=corr_axis_out_exp, name='corr_axis_out_exp', line = dict(color='red', width=2, dash='dash'), showlegend=True))
		fig.update_layout(title='Correlation of axis experimental matrix and axis data-driven (across trials) per subject', xaxis_title='subjects', yaxis_title='correlation (%s): axis' % (varr['which_exp']))
		fig.show(config=config)
		fig.write_image("%s%scorr_axis_out_exp_%s.png" % (varr['main_path2'], filemarker, varr['which_exp']))
		# ------------------------------
		
		
		# ------------------------------
		# Correlation study 2 : to determine which speed stim measure is most reliable
		# We correlate directional and magnitude speed measures with each other, the measures that are
		# most similar to each other are likely the correct speed stimuli eventhough there were recording delays 
		# for this measure.
		
		# Speed stim was the most unreliably saved measure to file, the output value to the .txt file was delayed with
		# respect to the cabin and joystick values for certain participants, due to the real-time functioning of 
		# the simulator system. The correlation analysis is a way to numerically determine the speed stim for all 
		# participant data files, despite the simulator delays.
		# ------------------------------
		# e2, e4 # Way 0 : speed_stim_org : experimental matrix stimulus magnitude and direction
		# e5, e3 # Way 1 : tar_ang_speed_val_co : experimental matrix stimulus magnitude and direction
		
		# Way 2 : Global : to calculate speed stim magnitude and direction : Plot all data and perform kmeans 1D for two groups (sub, sup) 
		speed_stim_dd, slope_per_exp = s3_calc_datadriven_speedstim_4pipeline(datout_temp)   # data-driven stimulus magnitude and direction 
		
		# e0, e1, Way 3 : Per trial : data-driven using the initial slope of each trial to get the direction. Magnitude was designated by a 
		# threshold value per trial with respect to group min and max values.
		
		allsubmag = []
		allsubdir = []
		allsubmagdir = []
		for s in subs:
			# Both magnitude & direction
			speed_stim_org = datout_temp[s][2]
			tar_ang_speed_val_co = datout_temp[s][3]
			
			# -----------------------------------
			
			# Magnitude
			speed_stim_dd_mag = speed_stim_dd[s]    # data-driven measured stimulus magnitude : kmeans (BEST)
			
			speed_stim_org = [float("{:.2f}".format(i)) for i in list(speed_stim_org)]
			speed_stim_org_mag = transform_ss2int_persub(varr, speed_stim_org)
			
			speed_stim_tas_mag = transform_ss2int_persub(varr, tar_ang_speed_val_co)
			
			# -----------------------------------
			
			# Direction
			speed_stim_dd_sign = np.sign(slope_per_exp[s]) # data-driven measured stimulus magnitude : kmeans (BEST)
			speed_stim_org_sign = datout_temp[s][0]
			speed_stim_tas_sign = datout_temp[s][1]
			
			# -----------------------------------
			
			# Magnitude & Direction
			speed_stim_dd_magsign = [speed_stim_dd_mag[i]*speed_stim_dd_sign[i] for i in range(len(speed_stim_dd_mag))]
			speed_stim_org_magsign = [speed_stim_org_mag[i]*speed_stim_org_sign[i] for i in range(len(speed_stim_org_mag))]
			speed_stim_tas_magsign = [speed_stim_tas_mag[i]*speed_stim_tas_sign[i] for i in range(len(speed_stim_tas_mag))]
			
			# -----------------------------------
			
			qmag = [speed_stim_dd_mag, speed_stim_org_mag, speed_stim_tas_mag]
			qdir = [speed_stim_dd_sign, speed_stim_org_sign, speed_stim_tas_sign]
			qmagdir = [speed_stim_dd_magsign, speed_stim_org_magsign, speed_stim_tas_magsign]
			
			mat = np.zeros((len(qmag),len(qmag)))
			matdir = np.zeros((len(qdir),len(qdir)))
			matmagdir = np.zeros((len(qmagdir),len(qmagdir)))
			
			for ind1, vec1 in enumerate(qmag):
				for ind2, vec2 in enumerate(qmag):
					corrvals = np.corrcoef(vec1, vec2) # outputs a correlation matrix
					mat[ind1:ind1+1,ind2:ind2+1] = corrvals[0,1]
					corrvals_dir = np.corrcoef(qdir[ind1], qdir[ind2]) # outputs a correlation matrix
					matdir[ind1:ind1+1,ind2:ind2+1] = corrvals_dir[0,1]
					corrvals_magdir = np.corrcoef(qmagdir[ind1], qmagdir[ind2]) # outputs a correlation matrix
					matmagdir[ind1:ind1+1,ind2:ind2+1] = corrvals_magdir[0,1]
			out = np.triu(mat,1)
			out = out.flatten() # ssdd_ssorg, ssdd_sstas, ssorg_sstas
			out_mag = [i for i in out if i != 0]# ne prends pas des valeurs 0
			out = np.triu(matdir,1)
			out = out.flatten() # ssdd_ssorg, ssdd_sstas, ssorg_sstas
			out_dir = [i for i in out if i != 0]# ne prends pas des valeurs 0
			out = np.triu(matmagdir,1)
			out = out.flatten() # ssdd_ssorg, ssdd_sstas, ssorg_sstas
			out_magdir = [i for i in out if i != 0]# ne prends pas des valeurs 0

			allsubmag.append(out_mag)
			allsubdir.append(out_dir)
			allsubmagdir.append(out_magdir)
		
		# ------------------------------
		# Plot results for selection of prefered speed stim
		# ------------------------------
		dftmp2 = pd.DataFrame(allsubmag)
		dftmp2.columns = ['ssdd_ssorg', 'ssdd_sstas', 'ssorg_sstas']

		condi0 = dftmp2.ssdd_ssorg.to_numpy()
		condi1 = dftmp2.ssdd_sstas.to_numpy()
		condi2 = dftmp2.ssorg_sstas.to_numpy()

		fig = go.Figure()
		config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
		xxORG = list(range(len(condi0)))
		fig.add_trace(go.Scatter(x=xxORG, y=condi0, name='ssdd_ssorg', line = dict(color='red', width=2, dash='dash'), showlegend=True))
		fig.add_trace(go.Scatter(x=xxORG, y=condi1, name='ssdd_sstas', line = dict(color='blue', width=2, dash='dash'), showlegend=True))
		fig.add_trace(go.Scatter(x=xxORG, y=condi2, name='ssorg_sstas', line = dict(color='orange', width=2, dash='dash'), showlegend=True))
		fig.update_layout(title='Magnitude correlation of speed stim measures per subject', xaxis_title='subjects', yaxis_title='correlation (%s): axis' % (varr['which_exp']))
		fig.show(config=config)
		fig.write_image("%s%scorr_ssmag_%s.png" % (varr['main_path2'], filemarker, varr['which_exp']))
		
		# ------------------------------
		
		dftmp3 = pd.DataFrame(allsubdir)
		dftmp3.columns = ['ssdd_ssorg', 'ssdd_sstas', 'ssorg_sstas']

		condi0 = dftmp3.ssdd_ssorg.to_numpy()
		condi1 = dftmp3.ssdd_sstas.to_numpy()
		condi2 = dftmp3.ssorg_sstas.to_numpy()

		fig = go.Figure()
		config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
		xxORG = list(range(len(condi0)))
		fig.add_trace(go.Scatter(x=xxORG, y=condi0, name='ssdd_ssorg', line = dict(color='red', width=2, dash='dash'), showlegend=True))
		fig.add_trace(go.Scatter(x=xxORG, y=condi1, name='ssdd_sstas', 
					             line = dict(color='blue', width=2, dash='dash'), showlegend=True))
		fig.add_trace(go.Scatter(x=xxORG, y=condi2, name='ssorg_sstas', 
					             line = dict(color='orange', width=2, dash='dash'), showlegend=True))

		fig.update_layout(title='Direction correlation of speed stim measures per subject', 
					      xaxis_title='subjects', yaxis_title='correlation (%s)' % (varr['which_exp']))
		fig.show(config=config)
		fig.write_image("%s%scorr_ssdir_%s.png" % (varr['main_path2'], filemarker, varr['which_exp']))

		# ------------------------------
		
		dftmp4 = pd.DataFrame(allsubmagdir)
		dftmp4.columns = ['ssdd_ssorg', 'ssdd_sstas', 'ssorg_sstas']

		condi0 = dftmp4.ssdd_ssorg.to_numpy()
		condi1 = dftmp4.ssdd_sstas.to_numpy()
		condi2 = dftmp4.ssorg_sstas.to_numpy()

		fig = go.Figure()
		config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
		xxORG = list(range(len(condi0)))
		fig.add_trace(go.Scatter(x=xxORG, y=condi0, name='ssdd_ssorg', line = dict(color='red', width=2, dash='dash'), showlegend=True))
		fig.add_trace(go.Scatter(x=xxORG, y=condi1, name='ssdd_sstas', line = dict(color='blue', width=2, dash='dash'), showlegend=True))
		fig.add_trace(go.Scatter(x=xxORG, y=condi2, name='ssorg_sstas', line = dict(color='orange', width=2, dash='dash'), showlegend=True))
		fig.update_layout(title='Magnitude & direction correlation of speed stim measures per subject', xaxis_title='subjects', yaxis_title='correlation (%s)' % (varr['which_exp']))
		fig.show(config=config)
		fig.write_image("%s%scorr_ssmagdir_%s.png" % (varr['main_path2'], filemarker, varr['which_exp']))
		
		# ------------------------------

		
		
		# ------------------------------
		# Refold all the data into a new condensed dataout matrix : including data-driven global
		# ------------------------------
		datout = []
		for s in subs:
			# We calculate kmeans of the initial cabin slope values and make two groups (sub, sup)
			speed_stim_DD = np.sign(slope_per_exp[s])*speed_stim_dd[s]    # 1=sub, 2=sup
			e0 = [np.round(int(x), 0) for x in speed_stim_DD]   # 0 : speed_stim (DD)
			
			e1 = datout_temp[s][4]  # 1 : axis_out (DD)
			
			e2 = datout_temp[s][7]  # 2 : new3_ind_st
			e3 = datout_temp[s][8]  # 3 : new3_ind_end
			
			e4 = datout_temp[s][9]  # 4 : outSIGCOM   
			e5 = datout_temp[s][10]  # 5 : outSIG
			e6 = datout_temp[s][11]  # 6 : outJOY
			e7 = datout_temp[s][12]  # 7 : outNOISE
			
			e8 = datout_temp[s][13]  # 8 : time_org
			e9 = datout_temp[s][14]  # 9 : trialnum_org
			e10 = datout_temp[s][15]  # 10 : SSQ
			e11 = datout_temp[s][16]  # 11 : FRT
			
			subdat = e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11

			datout.append(subdat)
		# ------------------------------
		
		
		# ------------------------------
		# Save data matrices to file per experiment
		# ------------------------------
		file_name = "%s%s%sdat.pkl" % (varr['main_path2'], filemarker, varr['which_exp'])
		open_file = open(file_name, "wb")
		pickle.dump(datout, open_file)
		del datout, datout_temp
		open_file.close()
		# ------------------------------
		
		
		
		# ------------------------------
		# Bar graph
		# ------------------------------
		print('exp_reject_cat_vec : ' + str(exp_reject_cat_vec))

		exp_acc_tr_per = make_a_properlist(exp_acc_tr_per)
		# print('exp_acc_tr_per : ' + str(exp_acc_tr_per))
		
		if exp == 0:
		    ver_short = []
		    hor_short = []
		    standard = []
		    subname = []
		    for subs in range(len(exp_reject_cat_vec)):
		        ver_short = ver_short + [exp_reject_cat_vec[subs][0]]
		        hor_short = hor_short + [exp_reject_cat_vec[subs][1]]
		        standard = standard + [exp_reject_cat_vec[subs][2]]
		        subname = subname + ['s%d_%f' % (subs, exp_acc_tr_per[subs])]

		    ver_short = make_a_properlist(ver_short)
		    hor_short = make_a_properlist(hor_short)
		    standard = make_a_properlist(standard)
		    subname = make_a_properlist(subname)
		elif exp == 1:
		    robotjump = []
		    robotstall = []
		    FB_nonzero = []
		    UD_init = []
		    ver_short = []
		    hor_short = []
		    standard = []
		    subname = []
		    for subs in range(len(exp_reject_cat_vec)):
		        robotjump = robotjump + [exp_reject_cat_vec[subs][0]]
		        robotstall = robotstall + [exp_reject_cat_vec[subs][1]]
		        FB_nonzero = FB_nonzero + [exp_reject_cat_vec[subs][2]]
		        UD_init = UD_init + [exp_reject_cat_vec[subs][3]]
		        ver_short = ver_short + [exp_reject_cat_vec[subs][4]]
		        hor_short = hor_short + [exp_reject_cat_vec[subs][5]]
		        standard = standard + [exp_reject_cat_vec[subs][6]]
		        subname = subname + ['s%d_%f' % (subs, exp_acc_tr_per[subs])]   

		    robotjump = make_a_properlist(robotjump)
		    robotstall = make_a_properlist(robotstall)
		    FB_nonzero = make_a_properlist(FB_nonzero)
		    UD_init = make_a_properlist(UD_init)
		    ver_short = make_a_properlist(ver_short)
		    hor_short = make_a_properlist(hor_short)
		    standard = make_a_properlist(standard)
		    subname = make_a_properlist(subname)
		    
		if exp == 0:
		    fig = go.Figure(data=[
		        go.Bar(name='ver_short', x=subname, y=ver_short),
		        go.Bar(name='hor_short', x=subname, y=hor_short),
		        go.Bar(name='standard', x=subname, y=standard)])
		elif exp == 1:
		    fig = go.Figure(data=[
		        go.Bar(name='robotjump', x=subname, y=robotjump),
		        go.Bar(name='robotstall', x=subname, y=robotstall),
		        go.Bar(name='FB_nonzero', x=subname, y=FB_nonzero),
		        go.Bar(name='UD_init', x=subname, y=UD_init),
		        go.Bar(name='ver_short', x=subname, y=ver_short),
		        go.Bar(name='hor_short', x=subname, y=hor_short),
		        go.Bar(name='standard', x=subname, y=standard)])
		titlestr = '%s : mean acceptance=%f, strictness_axes=%s, strictness_remove=%s' % (varr['which_exp'], mean(exp_acc_tr_per), strictness_text, strictness1_text)
		fig.update_layout(title=titlestr, xaxis_title='subjects (trial acceptance percentage)', yaxis_title='Trial rejection count', barmode='stack')
		fig.show()
		
		naame1 = '%s%strial_reject_stats_%s_%s_%s' % (varr['main_path2'], filemarker, varr['which_exp'], strictness, strictness1)
		
		fig.write_image("%s.png" % (naame1))
		# ------------------------------
		
		
		# ------------------------------
		# Save data matrices to file per experiment
		# file_name = "%s%sexp_reject_cat_vec_%s.pkl" % (varr['main_path2'], filemarker, varr['which_exp'])
		# open_file = open(file_name, "wb")
		# pickle.dump(exp_reject_cat_vec, open_file)
		# open_file.close()
		# ------------------------------
		
		
		# ------------------------------
		# j06/m01/a22 : need better statistic on how many removed trials : save g_rej_vec
		# ------------------------------
		tr_type_sub_exp.append(tr_type_sub)
		
	# end of exp

	# ------------------------------
	# j06/m01/a22 : Save statistic results to file - 0=good trial
	# ------------------------------
	# if exp == 0:
		# 30 = cut_trial_ver_short
		# 40 = cut_trial_hor_short
		# 50 = cut_trial_standard
		# bar_title = ['ver_short', 'hor_short', 'standardization']
	# elif exp == 1:
		# 10 = robotjump_cutlist
		# 15 = robotstall_cutlist
		# 20 = FB_nonzero_start
		# 25 = UD_initialization
		# 30 = cut_trial_ver_short
		# 40 = cut_trial_hor_short
		# 50 = cut_trial_standard

	# file_name = "%s%sexp_sub_statistic.pkl" % (varr['main_path2'], filemarker)
	# open_file = open(file_name, "wb")
	# pickle.dump(tr_type_sub_exp, open_file)
	# open_file.close()
	# ------------------------------
	
	
	return


# ------------------------------
# SUBFUNCTIONS
# ------------------------------
	
def transform_ss2int_persub(varr, ss_persub):

    outlier_val = 0
    sup_val = 2
    sub_val = 1

    ss_int_persub = []

    # print('len(ss_persub) : ', len(ss_persub))
        
    for tr in range(len(ss_persub)):
        # print('tr : ', tr)
        
        # Turn target angular speed values into 0, 1, or outlier_val
        if varr['which_exp'] == 'rot':
            if abs(ss_persub[tr]) == 1.25 or abs(ss_persub[tr]) == 1.5:
                ss_int_persub.append(sup_val)   # sup
            elif abs(ss_persub[tr]) == 0.5:
                ss_int_persub.append(sub_val)   # sub
            else:
                ss_int_persub.append(outlier_val)    # outlier

        elif varr['which_exp'] == 'trans':
            if abs(ss_persub[tr]) == 15 or abs(ss_persub[tr]) == 17.5:
                ss_int_persub.append(sup_val)   # sup
            elif abs(ss_persub[tr]) == 3.75:
                ss_int_persub.append(sub_val)   # sub
            else:
                ss_int_persub.append(outlier_val)    # outlier


    return ss_int_persub

# ------------------------------

# PURPOSE : Calculates data-driven speed stimulus (speed_stim_dd).
def s3_calc_datadriven_speedstim_4pipeline(dat):

    # ------------------------------
    
    slope_per_exp = []
    
    # 2) Load subjects
    subs = np.arange(len(dat))

    for s in subs:
        num_of_tr = len(dat[s][0])

        axis_out = dat[s][4]                # e4 : axis_out

        outSIG = dat[s][10]                 # e10 : outSIG
        outJOY = dat[s][11]                 # e11 : outJOY
		
        time = dat[s][13]                   # e13 : time_org
        
        # ------------------------------
        
        slope_val = []
        for tr in range(num_of_tr):
            
            t = time[tr]
            tr_JOY = outJOY[tr][:, axis_out[tr]]
            tr_SIG = outSIG[tr][:, axis_out[tr]]
            
            tr_len = len(tr_SIG)  # check if it gives the rows, not the columns
            
            # If the joystick is not moved, we can see the true stimulus signal
            if abs(np.max(tr_JOY)) < 0.1: # deadzone
                
                num_of_init_pts = int(tr_len/5)
            else:
                # When the joystick was moved
                baseline = 0
                marg =  0.1
                dpOFsig_in_zone, indexOFsig_in_zone, dp_sign_not_in_zone, indexOFsig_not_in_zone = detect_sig_change_wrt_baseline(tr_JOY, baseline, marg)
                
                # Find when the joystick was moved - this is the stop pt to take the slope
                num_of_init_pts = indexOFsig_not_in_zone[0]
                
                # Corrections
                if not num_of_init_pts.any():
                    num_of_init_pts = 2   # if the joystick is moved initially, just use first two points
            
            
            vec = tr_SIG[0:num_of_init_pts]
            xx = list(range(len(vec)))
            P = np.polyfit(xx, vec, 1)# approximates coefficients for a linear function y = a1*x + a2 so first coefficient is the slope
            w_MP = P[0]		# a1 (slope), linear parameter
            b_MP = P[1]		# a2 (y-intercept), linear parameter
            
            # The initial slope across trials for each subject
            slope_val = slope_val + [w_MP]
        
        slope_val = make_a_properlist(slope_val)
        slope_per_exp = slope_per_exp + [slope_val]
    # ------------------------------
    
    # ------------------------------
    # Append all the data across subjects and trials
    totsub_sig_val = make_a_properlist(slope_per_exp)
    
    # Calculate mean and min-max boundaries for grouped data
    fs = 10
    max_clu_0, min_clu_0, max_clu_1, min_clu_1 = kmeans_minmax_boundary(totsub_sig_val, fs)
    
    # Calculate speed stim categories from initial slope data, using the max and min boundary thresholds.
    speed_stim_dd = speed_stim_from_initslope(max_clu_0, min_clu_0, max_clu_1, min_clu_1, slope_per_exp)
    # ------------------------------
    
    return speed_stim_dd, slope_per_exp
    
# ------------------------------

# PURPOSE : to calculate two distinct groups with a dataset using kmeans, it returns the max and min boundaries around the mean.
def kmeans_minmax_boundary(alldata, fs):
    
    ts = 1/fs
    
    # ------------------------------
    # Make all the points positive, to include both negative and positive stimulation
    alldata = np.array(alldata)
    alldata_abs = np.abs(alldata)

    # remove outliers before performing kmeans, so that the sup and sub estimate is more exact
    avg_alldata_abs = np.mean(alldata_abs)

    thresh = 4*np.std(alldata_abs)  # outliers are 4 standard deviations from the mean

    newvec, ind_newvec =  findall(alldata_abs, '<', thresh)
    pos_alldata = []
    for i in ind_newvec:
        pos_alldata.append(alldata_abs[i])
    # ------------------------------


    # ------------------------------
    pos_alldata = make_a_properlist(pos_alldata)
    pos_alldata = np.array(pos_alldata)

    # Reshape into a column
    pos_alldata = np.reshape(pos_alldata, (len(pos_alldata), 1))
    # OR
    # pos_alldata.reshape(-1,1)

    # ------------------------------
    # Need to find two 1-dimensional lines that are positve and then the same two lines that are negative
    # ------------------------------
    # kmeans = KMeans(n_clusters=2)
    # clusters_out = kmeans.fit_predict(pos_alldata)   # this is how you get labels/features from the data

    # OR 

    kmeans = KMeans(n_clusters=2).fit(pos_alldata)
    clusters_out = kmeans.predict(pos_alldata)
    # print('clusters_out : ' + str(clusters_out))
    # ------------------------------

    # ------------------------------
    # Centroid values : this is the 
    centroids = kmeans.cluster_centers_
    # print('centroids org : ' + str(centroids))
    mean_clu_0 = centroids[0]
    mean_clu_1 = centroids[1]

    # Note to self : confirming that the centroids are the mean of each category
    newvec, ind_newvec =  findall(clusters_out, '==', 0)
    clu_0 = []
    for i in ind_newvec:
        clu_0.append(pos_alldata[i])
    # mean_clu_0 = np.mean(clu_0)
    # print('mean_clu_0 : ' + str(mean_clu_0))

    newvec, ind_newvec =  findall(clusters_out, '==', 1)
    clu_1 = []
    for i in ind_newvec:
        clu_1.append(pos_alldata[i])
    # mean_clu_1 = np.mean(clu_1)
    # print('mean_clu_1 : ' + str(mean_clu_1))
    # ------------------------------

    # ------------------------------
    # Label the clusters correctly : from min to max; 0=min cluster, 1=max cluster.  the function kmeans returns the clusters in random order, 
    # sometimes max first and min, or vice versa.
    # ------------------------------

    if mean_clu_0 > mean_clu_1:
        # reassign cluster 0 (lowest cluster) with mean_clu_1
        temp = mean_clu_0
        mean_clu_0 = mean_clu_1
        mean_clu_1 = temp
        temp = clu_0
        clu_0 = clu_1
        clu_1 = temp
    # print('centroids min-max: ' + str(mean_clu_0) + ', ' + str(mean_clu_1))

    # ------------------------------
    # Find the margin boundary that separates the two clusters
    # ------------------------------
    # want min_clu_1 >= max_clu_0 by adjusting numm
    start_val = 1
    stop_val = 2
    N = int(fs*stop_val)
    out = np.multiply(range(start_val, N), ts) 
    std_list = out[::-1]  # Invert list

    flag = 0
    for numm in std_list:
        max_clu_0 = numm*np.std(clu_0) + mean_clu_0
        min_clu_0 = mean_clu_0 - numm*np.std(clu_0)
        max_clu_1 = numm*np.std(clu_1) + mean_clu_1
        min_clu_1 = mean_clu_1 - numm*np.std(clu_1)
        
        if min_clu_1 >= max_clu_0 and flag == 0:
            flag = 1
            break
    # print('numm : ' + str(numm))
    # ------------------------------
    
    
    # ------------------------------
    # Plot all data and the min max boundaries for sub-sup labeling
    plotORnot = 0
    
    if plotORnot == 1:
        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        
        # All initial slope estimates
        xxORG = list(range(len(alldata)))
        fig.add_trace(go.Scatter(x=xxORG, y=alldata, name='slope', mode='markers', marker=dict(color='yellow', size=10, symbol=5, line=dict(color='black', width=0)), showlegend=True))
        
        # 4 standard deviations from mean of all initial slope estimates
        thresh_vec = thresh*np.ones((len(xxORG),1))
        thresh_vec = make_a_properlist(thresh_vec)
        fig.add_trace(go.Scatter(x=xxORG, y=thresh_vec, name='thresh_vec', line = dict(color='green', width=1, dash='dash'), showlegend=True))
        
        # All initial slope estimates made positive 
        pos_alldata = make_a_properlist(pos_alldata)
        fig.add_trace(go.Scatter(x=xxORG, y=pos_alldata, name='slope', mode='markers', marker=dict(color='black', size=10, symbol=3, line=dict(color='black', width=0)), showlegend=True))
        
        # Plotting first cluster : min range
        mean_clu_0_vec = mean_clu_0*np.ones((len(xxORG),1))
        mean_clu_0_vec = make_a_properlist(mean_clu_0_vec)
        fig.add_trace(go.Scatter(x=xxORG, y=mean_clu_0_vec, name='mean_clu_0', line = dict(color='red', width=2, dash='dash'), showlegend=True))
        
        max_clu_0_vec = max_clu_0*np.ones((len(xxORG),1))
        max_clu_0_vec = make_a_properlist(max_clu_0_vec)
        fig.add_trace(go.Scatter(x=xxORG, y=max_clu_0_vec, name='max_clu_0', line = dict(color='red', width=1, dash='dash'), showlegend=True))
        
        min_clu_0_vec = min_clu_0*np.ones((len(xxORG),1))
        min_clu_0_vec = make_a_properlist(min_clu_0_vec)
        fig.add_trace(go.Scatter(x=xxORG, y=min_clu_0_vec, name='min_clu_0', line = dict(color='red', width=1, dash='dash'), showlegend=True))
        
        # Plotting second cluster : max range
        mean_clu_1_vec = mean_clu_1*np.ones((len(xxORG),1))
        mean_clu_1_vec = make_a_properlist(mean_clu_1_vec)
        fig.add_trace(go.Scatter(x=xxORG, y=mean_clu_1_vec, name='mean_clu_1', line = dict(color='blue', width=2, dash='dash'), showlegend=True))
        
        max_clu_1_vec = max_clu_1*np.ones((len(xxORG),1))
        max_clu_1_vec = make_a_properlist(max_clu_1_vec)
        fig.add_trace(go.Scatter(x=xxORG, y=max_clu_1_vec, name='max_clu_1', line = dict(color='blue', width=1, dash='dash'), showlegend=True))
        
        min_clu_1_vec = min_clu_1*np.ones((len(xxORG),1))
        min_clu_1_vec = make_a_properlist(min_clu_1_vec)
        fig.add_trace(go.Scatter(x=xxORG, y=min_clu_1_vec, name='min_clu_1', line = dict(color='blue', width=1, dash='dash'), showlegend=True))
        
        title_str = 'All data : exp %d' % (exp)
        fig.update_layout(title=title_str, xaxis_title='data points', yaxis_title='outSIG at index of max(abs(outSIG))')
        fig.show(config=config)
    # ------------------------------
    
    return max_clu_0, min_clu_0, max_clu_1, min_clu_1

# ------------------------------

# PURPOSE : calculates speed stim categories from initial slope data, using the max and min boundary thresholds.
def speed_stim_from_initslope(max_clu_0, min_clu_0, max_clu_1, min_clu_1, totsub_slope_ORG):

    # sup trial = trials with initial slope points between min_clu_1_OR_0.1 and max_clu_1
    # sub trial = trials with initial slope points between min_clu_0_OR_0.1 and max_clu_0

    sub_range = [min_clu_0, max_clu_0]
    sup_range = [min_clu_1, max_clu_1]

    max_val = sup_range[1]
    mid_val = (sup_range[0] + sub_range[1])/2
    min_val = sub_range[0]

    # ------------------------------
    # Divide the data into sub and sup based on the margin boundaries : 
    # ------------------------------
    outlier_val = 0
    sup_val = 2
    sub_val = 1

    speed_stim_per_exp = []

    # print('len(totsub_slope_ORG) : ' + str(len(totsub_slope_ORG)))

    for s in range(len(totsub_slope_ORG)):  # loop over the number of subjects and label each trial sup=1,  sub=0, or outlier
        
        num_of_tr = len(totsub_slope_ORG[s])
        
        stim_speed_pertr = []
        
        for tr in range(num_of_tr):
            if np.abs(totsub_slope_ORG[s][tr]) < 0.001:  # zero stim is considered an outlier
                sstim = outlier_val   # outlier
            else:
                if (np.abs(totsub_slope_ORG[s][tr]) > mid_val) and (np.abs(totsub_slope_ORG[s][tr]) < max_val):
                    sstim = sup_val   # sup
                elif (np.abs(totsub_slope_ORG[s][tr]) < mid_val) and (np.abs(totsub_slope_ORG[s][tr]) > min_val):
                    # print('sub detected : ' + str(np.abs(totsub_slope_ORG[s][tr])))
                    sstim = sub_val   # sub
                else:
                    sstim = outlier_val   # outlier: vaules above the max_clu_1 line and below the min_clu_0 line
            
            stim_speed_pertr = stim_speed_pertr + [sstim]
            
        stim_speed_pertr = make_a_properlist(stim_speed_pertr)
        speed_stim_per_exp = speed_stim_per_exp + [stim_speed_pertr]  # per subject
        # print('s : ' + str(s) + ', ' + 'speed_stim_per_exp : ' + str(speed_stim_per_exp))
    # ------------------------------

    return speed_stim_per_exp

# ------------------------------




# ------------------------------

