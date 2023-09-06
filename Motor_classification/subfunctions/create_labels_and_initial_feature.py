import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from subfunctions.indexit import *


def create_labels_and_initial_feature(df):
    # IC = 1
    # EC = 2, 4, 5
    # NC = 3, 6, 7
    # NR = 9
    # (IC) - sham (do not use) = 8
    # (NC) - sham (do not use) = 10

    # Just to confirm, what are the unique values of res_type
    df.res_type.value_counts(ascending=True)

    # Construction of SD_label : How do we define disorientation?

    # Way 0 : lenient
    # 0 = If participates got the result CORRECT for the trial, they were NOT disoriented. (IC, EC)
    # 1 = If participates got the result WRONG or did not respond, they were disoriented. (NC, NR)

    idx_NDS = df.index[(df.res_type == 1) | (df.res_type == 2) | (df.res_type == 4) | (df.res_type == 5)].to_list()
    idx_DS = df.index[(df.res_type == 3) | (df.res_type == 6) | (df.res_type == 7) | (df.res_type == 9)].to_list()
    df.lenient = ''  # define a new column , rows 8 and 10 will be NaN, need to do dropna for rows
    df.loc[idx_NDS, 'lenient'] = 0
    df.loc[idx_DS, 'lenient'] = 1

    # -------------------------------------

    # Way 1 : strict simple
    # 0 = If participants got the result initially CORRECT, they were NOT disoriented. (IC)
    # 1 = If participants were WRONG at any point, they were disoriented. (EC, NC, NR)

    idx_NDS = df.index[(df.res_type == 1)].to_list()
    idx_DS = df.index[(df.res_type == 2) | (df.res_type == 4) | (df.res_type == 5) | (df.res_type == 3) | (df.res_type == 6) | (df.res_type == 7) | (df.res_type == 9)].to_list()
    df.strict = ''  # define a new column
    df.loc[idx_NDS, 'strict'] = 0
    df.loc[idx_DS, 'strict'] = 1

    # Way 2 : st_complex
    # 0 = If participants got the result initially CORRECT, they were NOT disoriented. (IC)
    # 1 = If participants got the result eventually CORRECT, they were MILDLY disoriented. (EC)
    # 2 = If participants were WRONG for the trial, they were disoriented. (NC, NR)

    idx_NDS = df.index[(df.res_type == 1)].to_list()
    idx_MDS = df.index[(df.res_type == 2) | (df.res_type == 4) | (df.res_type == 5)].to_list()
    idx_DS = df.index[(df.res_type == 3) | (df.res_type == 6) | (df.res_type == 7) | (df.res_type == 9)].to_list()
    df.st_complex = ''  # define a new column
    df.loc[idx_NDS, 'st_complex'] = 0
    df.loc[idx_MDS, 'st_complex'] = 1
    df.loc[idx_DS, 'st_complex'] = 2

    # -------------------------------------

    # Create features :  (1) position
    df['joy_stim'] = df.apply(indexit, axis='columns')  # fill in joy_stim
    
    # -------------------------------------

    # Make DataFrame for trial start-stop index
    # Cut the data up per trial across subjects
    tr_vec = df.tr.to_numpy()

    st = [0]
    ender = []
    for i in range(len(tr_vec)-1):
        if tr_vec[i] != tr_vec[i+1]:
            st = st + [i+1]
            ender = ender + [i]
    ender = ender + [len(tr_vec)-1]

    # See start-stop index clearly
    e0 = np.reshape(st, (len(st),1))
    e1 = np.reshape(ender, (len(st),1))
    data = np.ravel(e0), np.ravel(e1)
    data = np.transpose(data)
    columns = ['stind', 'endind']
    temp = pd.DataFrame(data=data, columns=columns)

    # -------------------------------------

    # Find the longest trial signal in df_rot['joy_stim']
    temp['diff'] = temp.endind - temp.stind
    temp['timediff'] = [df.time.iloc[temp.endind[i]] - df.time.iloc[temp.stind[i]] for i in range(len(temp.endind))]
    outmin = temp['diff'].min()
    outmax = temp['diff'].max()

    tomin = temp['timediff'][(temp['diff'] == outmin)]
    tomax = temp['timediff'][(temp['diff'] == outmax)]
    # print('outmin : ', outmin, 't :', tomin)
    # print('outmax : ', outmax, 't :', tomax)

    # -------------------------------------

    # Interpolate : make each trial the same number of data points
    feat0 = []
    t_feat0 = []
    y1_feat0 = []
    y2_feat0 = []
    y3_feat0 = []
    for i in range(len(temp.stind)):
        
        # X
        sSIG = df['joy_stim'][temp.stind.iloc[i]:temp.endind.iloc[i]].to_numpy()
        t_sSIG = df['time'][temp.stind.iloc[i]:temp.endind.iloc[i]].to_numpy()

        # labels
        y1 = df['lenient'][temp.stind.iloc[i]:temp.endind.iloc[i]].to_numpy()
        y2 = df['strict'][temp.stind.iloc[i]:temp.endind.iloc[i]].to_numpy()
        y3 = df['st_complex'][temp.stind.iloc[i]:temp.endind.iloc[i]].to_numpy()
        
        # Check if trial data is less than the maximum length
        if len(df['joy_stim'][temp.stind.iloc[i]:temp.endind.iloc[i]]) < outmax:
            
            # The trial length is different so interpolate the time-series to make them the same length signal 
            x = np.linspace(sSIG[0], len(sSIG), num=len(sSIG), endpoint=True)
            xnew = np.linspace(sSIG[0], len(sSIG), num=outmax, endpoint=True)

            # joystick on stim
            f = interp1d(x, sSIG)
            sSIGl = f(xnew)

            # time
            f = interp1d(x, t_sSIG)
            t_sSIGl = f(xnew)

            # y1
            f = interp1d(x, y1)
            y1_sSIGl = f(xnew)

            # y2
            f = interp1d(x, y2)
            y2_sSIGl = f(xnew)

            # y3
            f = interp1d(x, y3)
            y3_sSIGl = f(xnew)

            # python : you can not create a matrix in real-time in pandas
            # you only assign the full matrix at the end
            # (0) position
            feat0 = feat0 + [sSIGl]
            t_feat0 = t_feat0 + [t_sSIGl]
            y1_feat0 = y1_feat0 + [np.ravel(y1_sSIGl)]
            y2_feat0 = y2_feat0 + [np.ravel(y2_sSIGl)]
            y3_feat0 = y3_feat0 + [np.ravel(y3_sSIGl)]
            
            del x, f, sSIGl, t_sSIGl, y1_sSIGl, y2_sSIGl, y3_sSIGl
        else:
            feat0 = feat0 + [sSIG]
            t_feat0 = t_feat0 + [t_sSIG]
            y1_feat0 = y1_feat0 + [np.ravel(y1)]
            y2_feat0 = y2_feat0 + [np.ravel(y2)]
            y3_feat0 = y3_feat0 + [np.ravel(y3)]

    # Clean up
    del df
    # -------------------------------------
    
    return feat0, t_feat0, y1_feat0, y2_feat0, y3_feat0
