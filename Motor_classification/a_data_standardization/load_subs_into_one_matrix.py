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
from subfunctions.make_a_properlist import *
from subfunctions.size import *




def load_subs_into_one_matrix(varr):
    
    # ------------------------------

    # Way 1 : Count up correct joy-cabin follow for each category
    # the category with the most correct joy-cabin follows is the correct configuation 
    # given to the majority of participants=

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
            varr['data_path'] = '%s\\DATA_10Hz_rot' % (varr['main_path'])  # Windows
            # varr['data_path'] = '%s/DATA_10Hz_rot' % (varr['main_path'])

        elif exp == 1:
            # Translational data - 14 participants
            varr['which_exp'] = 'trans'
            varr['subjects'] = '20170411-TB-463','20172206-NB-777','20170413-GP-007','20172806-SM-308','20170404-AV-280','20170308-MK-160','20172008-AS-007','20170824-GL-380','p9_16102017-RW-115','p10_12102017-SG-123','p11_17102017-LG-123','p12_10102017-HS-000','p13_13102017-GB-666','p14_20171014-SL-132'
            varr['sub_nom_rot'] = 'TB','NB','GP','SM','AV','MK','AS','GL','RW','SG','LG','HS','GB','SL'        
            varr['anom'] = 'LR', 'FB', 'UD'
            varr['joyanom'] = '1', '2', '3'
            varr['vals'] = 3.75, 15, 0
            varr['data_path'] = '%s\\DATA_10Hz_trans' % (varr['main_path'])  # Windows
            # varr['data_path'] = '%s/DATA_10Hz_trans' % (varr['main_path'])


        # orax*yr*dirC*mj*strict
        cat_comb = 6*2*2*3*4  # equals 288 catigory combinations

        subs = range(0,len(varr['subjects']))
        subs = make_a_properlist(subs)

        Cor = np.zeros((len(subs),cat_comb))
        Wrg = np.zeros((len(subs),cat_comb))
        Cor_tdelay = np.zeros((len(subs),cat_comb))
        Wrg_tdelay = np.zeros((len(subs),cat_comb))

        Cor_dir = np.zeros((len(subs),cat_comb))
        Wrg_dir = np.zeros((len(subs),cat_comb))

        # 2) Load subjects
        for s in subs:
            cat = 0

            for orax in range(6):
                for yr in range(2):
                    for dirC in range(2):
                        for mj in range(3):
                            for strict in range(4):
                                
                                filename = 'data_standard_%s' % (varr['which_exp'])
                                file_name = "%s\\%s\\s%d_orax%d_yr%d_dirC%d_marg%d_st%d_outSIG.pkl" % (varr['main_path1'], filename, s, orax, yr, dirC, mj, strict)
                                open_file = open(file_name, "rb")
                                X = pickle.load(open_file)
                                open_file.close()

                                # print('X : ' + str(X))
                                # Index the X matrix
                                bm_Xind = 5 # binary_marker X matrix index
                                sc_Xind = 6 # scalar_marker X matrix index
                                dirm_Xind = 8 # dir_meaning_marker X matrix index

                                # binary_marker = [70=no joy move, 1=joy-cabin correct, 0=joy-cabin wrong]
                                cor_cabjoy = 0
                                wrg_cabjoy = 0
                                cor_tdelay = []
                                wrg_tdelay = []
                                cor_dir = 0
                                wrg_dir = 0
                                for tr in range(len(X)):
                                    if X[tr,bm_Xind] == 1:
                                        cor_cabjoy = cor_cabjoy + 1
                                        cor_tdelay = cor_tdelay + [X[tr,sc_Xind]]

                                        # Then of the correct joy-cabin follow trials - count up the direction correct
                                        # dir_meaning_marker:[80=ignore, 1=direction correct, 0=direction wrong]
                                        if X[tr,dirm_Xind] == 1:
                                            cor_dir = cor_dir + 1
                                        elif X[tr,dirm_Xind] == 0:
                                            wrg_dir = wrg_dir + 1

                                    elif X[tr,bm_Xind] == 0:
                                        wrg_cabjoy = wrg_cabjoy + 1
                                        wrg_tdelay = wrg_tdelay + [X[tr,sc_Xind]]


                                # print('cor_cabjoy : ' + str(cor_cabjoy))
                                # print('wrg_cabjoy : ' + str(wrg_cabjoy))

                                # print('cor_tdelay : ' + str(cor_tdelay))
                                # print('wrg_tdelay : ' + str(wrg_tdelay))

                                # Put values into matricies
                                Cor[s,cat] = cor_cabjoy
                                Wrg[s,cat] = wrg_cabjoy

                                # Put values into matricies
                                Cor_dir[s,cat] = cor_dir
                                Wrg_dir[s,cat] = wrg_dir

                                # average joy-cab delay across trials for correct follow
                                cor_tdelay = np.array(cor_tdelay)
                                if not cor_tdelay.any():
                                    # If cor_tdelay is empty
                                    Cor_tdelay[s,cat] = 0 
                                else:
                                    Cor_tdelay[s,cat] = mean(cor_tdelay)

                                wrg_tdelay = np.array(wrg_tdelay)
                                if not wrg_tdelay.any():
                                    # If wrg_tdelay is empty
                                    Wrg_tdelay[s,cat] = 0 
                                else:
                                    Wrg_tdelay[s,cat] = mean(wrg_tdelay)


                                cat = cat + 1

        print('Cor : ' + str(Cor))

        print('size of Cor : ' + str(size(Cor)))


        # Sum across subjects to get total count per category
        Cor_tot_across_cat = np.sum(Cor, axis=0)
        Wrg_tot_across_cat = np.sum(Wrg, axis=0)
        print('Cor_tot_across_cat : ' + str(Cor_tot_across_cat))  

        Cor_tdelay_tot_across_cat = np.sum(Cor_tdelay, axis=0)
        Wrg_tdelay_tot_across_cat = np.sum(Wrg_tdelay, axis=0)

        Cor_dir_tot_across_cat = np.sum(Cor_dir, axis=0)
        Wrg_dir_tot_across_cat = np.sum(Wrg_dir, axis=0)

        # -------------------------------

        # Generate a list of strings for categories
        cout = 0
        strict_iden = np.zeros((cat_comb,1))
        x_ax_name = []
        ct = 0
        for orax in range(6):
            for yr in range(2):
                for dirC in range(2):
                    for mj in range(3):
                        for strict in range(4):
                            if strict == 3: # [movement, direction] : 0=Strict, 1=Strict
                                strict_iden[ct] = 1
                            elif strict == 2: # [movement, direction] : 0=Strict, 1=Lenient (direction calc could have anomolies, like same stim and joystick movement axis is difficult to determine following due to delay and gain of stim versus joystick) - the majority of axes in the correct direction is probably more accurate
                                strict_iden[ct] = 2
                            x_ax_name = x_ax_name + ['o%d,y%d,d%d,m%d,st%d' % (orax, yr, dirC, mj, strict)]
                            ct = ct + 1

        # print('x_ax_name : ' + str(x_ax_name))
        x_ax_name = np.array(x_ax_name)
        
        # Highlighting conditions that are difficult to see
        prt_ind2 = np.zeros((cat_comb,1))
        prt_ind3 = np.zeros((cat_comb,1))
        ct1 = 0
        ct2 = 0
        for i in range(len(x_ax_name)):
            if strict_iden[i] == 1:
                prt_ind2[ct1] = i
                ct1 = ct1 + 1
            elif strict_iden[i] == 2:
                prt_ind3[ct2] = i
                ct2 = ct2 + 1
        prt_ind2 = [int(x) for x in prt_ind2] # convert to integer
        prt_ind3 = [int(x) for x in prt_ind3] # convert to integer
        
        # if varr['which_exp'] == 'rot':
            # Find key category indexes for plotting
                # if x_ax_name[i] == 'o2,d0,m0':
                    # prt_ind0 = i
                # elif x_ax_name[i] == 'o2,d1,m0':
                    # prt_ind1 = i
        # elif varr['which_exp'] == 'trans':
            # Find key category indexes for plotting
            # for i in range(len(x_ax_name)):
                # if x_ax_name[i] == 'o0,d0,m0':
                    # prt_ind0 = i
                # elif x_ax_name[i] == 'o0,d1,m0':
                    # prt_ind1 = i


        # print('x_ax_name[prt_ind0] : ' + str(x_ax_name[prt_ind0]))
        # print('x_ax_name[prt_ind1] : ' + str(x_ax_name[prt_ind1]))

        # -------------------------------


        save_filename0 = 'corcount_%s' % (varr['which_exp'])
        save_filename1 = 'tdelay_%s' % (varr['which_exp'])
        save_filename2 = 'dir_%s' % (varr['which_exp'])

        # -------------------------------

        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})

        fig.add_trace(go.Scatter(x=x_ax_name, y=Cor_tot_across_cat, name='Cor_tot_across_cat', line = dict(color='red', width=2, dash='solid'), showlegend=True))
        fig.add_trace(go.Scatter(x=x_ax_name, y=Wrg_tot_across_cat, name='Wrg_tot_across_cat', line = dict(color='blue', width=2, dash='solid'), showlegend=True))

        # Top counted categories
        # pt0 = prt_ind0*np.ones((2)) # must double the point : can not plot a singal point
        # pt0 = [int(x) for x in pt0] # convert to integer
        # fig.add_trace(go.Scatter(x=x_ax_name[pt0], y=Cor_tot_across_cat[pt0], name='pt0', mode='markers', marker=dict(color='red', size=15, symbol=5, line=dict(color='red', width=0)), showlegend=True))

        # pt1 = prt_ind1*np.ones((2)) # must double the point : can not plot a singal point
        # pt1 = [int(x) for x in pt1] # convert to integer
        # fig.add_trace(go.Scatter(x=x_ax_name[pt1], y=Cor_tot_across_cat[pt1], name='pt1', mode='markers', marker=dict(color='red', size=10, symbol=0, line=dict(color='red', width=0)), showlegend=True))
        
        fig.add_trace(go.Scatter(x=x_ax_name[prt_ind2], y=Cor_tot_across_cat[prt_ind2], name='strict-lenient', mode='markers', marker=dict(color='black', size=10, symbol=0, line=dict(color='red', width=0)), showlegend=True))
        fig.add_trace(go.Scatter(x=x_ax_name[prt_ind3], y=Cor_tot_across_cat[prt_ind3], name='strict-strict', mode='markers', marker=dict(color='cyan', size=10, symbol=0, line=dict(color='red', width=0)), showlegend=True))
        
        fig.update_layout(title='Total correct/wrong trials per category - %s' % (save_filename0), xaxis_title='category', yaxis_title='correct trial count')
        fig.show(config=config)

        fig.write_image("%s\\%s.png" % (varr['main_path1'], save_filename0))

        # -------------------------------

        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})

        fig.add_trace(go.Scatter(x=x_ax_name, y=Cor_tdelay_tot_across_cat, name='Cor_tdelay_tot_across_cat', line = dict(color='red', width=2, dash='solid'), showlegend=True))
        fig.add_trace(go.Scatter(x=x_ax_name, y=Wrg_tdelay_tot_across_cat, name='Wrg_tdelay_tot_across_cat', line = dict(color='blue', width=2, dash='solid'), showlegend=True))

        # Top counted categories
        # pt0 = prt_ind0*np.ones((2)) # must double the point : can not plot a singal point
        # pt0 = [int(x) for x in pt0] # convert to integer
        # fig.add_trace(go.Scatter(x=x_ax_name[pt0], y=Cor_tdelay_tot_across_cat[pt0], name='pt0', mode='markers', marker=dict(color='red', size=15, symbol=5, line=dict(color='red', width=0)), showlegend=True))

        # pt1 = prt_ind1*np.ones((2)) # must double the point : can not plot a singal point
        # pt1 = [int(x) for x in pt1] # convert to integer
        # fig.add_trace(go.Scatter(x=x_ax_name[pt1], y=Cor_tdelay_tot_across_cat[pt1], name='pt1', mode='markers', marker=dict(color='red', size=10, symbol=0, line=dict(color='red', width=0)), showlegend=True))
        
        fig.add_trace(go.Scatter(x=x_ax_name[prt_ind2], y=Cor_tdelay_tot_across_cat[prt_ind2], name='strict-lenient', mode='markers', marker=dict(color='black', size=10, symbol=0, line=dict(color='red', width=0)), showlegend=True))
        fig.add_trace(go.Scatter(x=x_ax_name[prt_ind3], y=Cor_tdelay_tot_across_cat[prt_ind3], name='strict-strict', mode='markers', marker=dict(color='cyan', size=10, symbol=0, line=dict(color='red', width=0)), showlegend=True))
        
        fig.update_layout(title='Time delay of correct/wrong trials per category - %s' % (save_filename1), xaxis_title='category', yaxis_title='Average time delay')
        fig.show(config=config)

        fig.write_image("%s\\%s.png" % (varr['main_path1'], save_filename1))

        # -------------------------------

        fig = go.Figure()
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})

        fig.add_trace(go.Scatter(x=x_ax_name, y=Cor_dir_tot_across_cat, name='Cor_dir_tot_across_cat', line = dict(color='red', width=2, dash='solid'), showlegend=True))
        fig.add_trace(go.Scatter(x=x_ax_name, y=Wrg_dir_tot_across_cat, name='Wrg_dir_tot_across_cat', line = dict(color='blue', width=2, dash='solid'), showlegend=True))

        # Top counted categories
        # pt0 = prt_ind0*np.ones((2)) # must double the point : can not plot a singal point
        # pt0 = [int(x) for x in pt0] # convert to integer
        # fig.add_trace(go.Scatter(x=x_ax_name[pt0], y=Cor_dir_tot_across_cat[pt0], name='pt0', mode='markers', marker=dict(color='red', size=15, symbol=5, line=dict(color='red', width=0)), showlegend=True))

        # pt1 = prt_ind1*np.ones((2)) # must double the point : can not plot a singal point
        # pt1 = [int(x) for x in pt1] # convert to integer
        # fig.add_trace(go.Scatter(x=x_ax_name[pt1], y=Cor_dir_tot_across_cat[pt1], name='pt1', mode='markers', marker=dict(color='red', size=10, symbol=0, line=dict(color='red', width=0)), showlegend=True))
        
        fig.add_trace(go.Scatter(x=x_ax_name[prt_ind2], y=Cor_dir_tot_across_cat[prt_ind2], name='strict-lenient', mode='markers', marker=dict(color='black', size=10, symbol=0, line=dict(color='red', width=0)), showlegend=True))
        fig.add_trace(go.Scatter(x=x_ax_name[prt_ind3], y=Cor_dir_tot_across_cat[prt_ind3], name='strict-strict', mode='markers', marker=dict(color='cyan', size=10, symbol=0, line=dict(color='red', width=0)), showlegend=True))

        fig.update_layout(title='Considering the correct trials - %s : correct/wrong direction per category' % (save_filename2), xaxis_title='category', yaxis_title='Direction count')
        fig.show(config=config)

        fig.write_image("%s\\%s.png" % (varr['main_path1'], save_filename2))

        # -------------------------------
        
    return