import pandas as pd
import pickle
import numpy as np


def put_timeseries_trialdata_into_pandas(varr, filemarker):
    
    df_timeseries_exp = {}

    # 1) Load exp : put the experiment that you wish to run
    for exp in range(2):  # 0=rotation, 1=translation
        # print('exp : ', exp)

        if exp == 0:
            # Rotational data - 18 participants
            varr['which_exp'] = 'rot'
            varr['anom'] = 'RO', 'PI', 'YA'
            
            # Time series data per subject per trial
            file_name1 = "%s%srotdat.pkl" % (varr['main_path2'], filemarker)

            # Load data experimental preprocessed data matrix
            file_name2 = "%s%srot_Xexp.pkl" % (varr['main_path3'], filemarker)
        elif exp == 1:
            # Translational data - 14 participants
            varr['which_exp'] = 'trans'
            varr['anom'] = 'LR', 'FB', 'UD'

            # Time series data per subject per trial
            file_name1 = "%s%stransdat.pkl" % (varr['main_path2'], filemarker)

            # Experimental preprocessed : a scalar metric per subject per trial
            file_name2 = "%s%strans_Xexp.pkl" % (varr['main_path3'], filemarker)

        open_file = open(file_name1, "rb")
        dat = pickle.load(open_file)
        open_file.close()

        open_file = open(file_name2, "rb")
        X = pickle.load(open_file)
        open_file.close()
        
        num_of_subs = len(X)    # OR dat.shape[0]

        Xsub = []
        
        for s in range(num_of_subs):
            # print('s : ', s)
            
            num_of_tr = len(dat[s][0])  # OR X[0].shape[0]

            for tr in range(num_of_tr):
                # print('tr : ', tr)
                
                # time series dataFrame (ensure vectors are a column): 
                num_dp = len(dat[s][4][tr][:,0])

                subject = s*np.ones((num_dp,1))
                trial = tr*np.ones((num_dp,1))
                ss = dat[s][0][tr]*np.ones((num_dp,1))
                ax = dat[s][1][tr]*np.ones((num_dp,1))

                dp = np.reshape(list(range(num_dp)), (num_dp,1))
                time = np.reshape(dat[s][8][tr], (num_dp,1))

                res_type = X[s][:,5][tr]*np.ones((num_dp,1))
                
                outSIGCOM_ax0 = np.reshape(dat[s][4][tr][:,0], (num_dp,1))
                outSIGCOM_ax1 = np.reshape(dat[s][4][tr][:,1], (num_dp,1))
                outSIGCOM_ax2 = np.reshape(dat[s][4][tr][:,2], (num_dp,1))

                outSIG_ax0 = np.reshape(dat[s][5][tr][:,0], (num_dp,1))
                outSIG_ax1 = np.reshape(dat[s][5][tr][:,1], (num_dp,1))
                outSIG_ax2 = np.reshape(dat[s][5][tr][:,2], (num_dp,1))

                outJOY_ax0 = np.reshape(dat[s][6][tr][:,0], (num_dp,1))
                outJOY_ax1 = np.reshape(dat[s][6][tr][:,1], (num_dp,1))
                outJOY_ax2 = np.reshape(dat[s][6][tr][:,2], (num_dp,1))

                outNOISE_ax0 = np.reshape(dat[s][7][tr][:,0], (num_dp,1))
                outNOISE_ax1 = np.reshape(dat[s][7][tr][:,1], (num_dp,1))
                outNOISE_ax2 = np.reshape(dat[s][7][tr][:,2], (num_dp,1))
                
                X_row = np.ravel(subject), np.ravel(trial), np.ravel(ss), np.ravel(ax), np.ravel(dp), np.ravel(time), np.ravel(res_type), np.ravel(outSIGCOM_ax0), np.ravel(outSIGCOM_ax1), np.ravel(outSIGCOM_ax2), np.ravel(outSIG_ax0), np.ravel(outSIG_ax1), np.ravel(outSIG_ax2), np.ravel(outJOY_ax0), np.ravel(outJOY_ax1), np.ravel(outJOY_ax2), np.ravel(outNOISE_ax0), np.ravel(outNOISE_ax1), np.ravel(outNOISE_ax2),
                Xtr = np.transpose(X_row)
                # print('shape of Xtr : ', Xtr.shape)
                
                # concatenate accumulated matrix with new
                if s == 0 and tr == 0:
                    Xsub = Xtr
                else:
                    Xsub = np.concatenate((Xsub, Xtr), axis=0)
                
                # print('len of Xsub : ', len(Xsub))
                
        
        columns = ['subject', 'tr', 'ss', 'ax', 'dp', 'time', 'res_type', 'SIGCOM_ax0', 'SIGCOM_ax1', 'SIGCOM_ax2', 'SIG_ax0', 'SIG_ax1', 'SIG_ax2', 'JOY_ax0', 'JOY_ax1', 'JOY_ax2', 'NOISE_ax0', 'NOISE_ax1', 'NOISE_ax2']
        out1 = np.reshape(Xsub, (len(Xsub), len(columns)))
        
        df = pd.DataFrame(out1, columns=columns)

        df_timeseries_exp[varr['which_exp']] = df

    return df_timeseries_exp
