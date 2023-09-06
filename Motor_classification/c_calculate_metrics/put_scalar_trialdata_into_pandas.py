import pandas as pd
import pickle
import numpy as np


def put_scalar_trialdata_into_pandas(varr, filemarker):
    
    df_scalarmetics_exp = {}

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

        dat = np.array(dat)
        # print('shape of dat : ', dat.shape)
        # print('length of dat : ', len(dat))
        # print('shape of dat[0] : ', len(dat[0][0]))

        X = np.array(X)
        # print('shape of X : ', X.shape)
        # print('length of X : ', len(X))
        # print('shape of X[0] : ', X[0].shape)

        num_of_subs = len(X)    # OR dat.shape[0]
        # print('num_of_subs : ', num_of_subs)

        row = []

        for s in range(num_of_subs):
            # print('s : ', s)

            num_of_tr = len(dat[s][0])  # OR X[0].shape[0]

            for tr in range(num_of_tr):
                # print('tr : ', tr)

                # scalar dataFrame
                subject = s
                trial = tr
                ss = dat[s][0][tr]
                ax = dat[s][1][tr]
                new3_ind_st = dat[s][2][tr]
                new3_ind_end = dat[s][3][tr]
                trnum_org = dat[s][9][tr]

                SSQ_b4 = dat[s][10][0]
                SSQ_af = dat[s][10][1]
                SSQ_diff = dat[s][10][2]
                FRT_em = dat[s][11][tr][0]

                res_type = X[s][:,5][tr]
                TR = X[s][:,6][tr]


                out = [subject, trial, ss, ax, new3_ind_st, new3_ind_end, trnum_org, SSQ_b4, SSQ_af, SSQ_diff, FRT_em, res_type, TR]

                # out = list(np.reshape(out, (1,len(out))))
                # print('out : ', out)

                row = row + [out]


        # print('FINAL : len of row : ', len(row))
        # print('FINAL : len of row[0] : ', len(row[0]))

        out1 = np.reshape(row, (len(row), len(row[0])))
        columns = ['subject', 'tr', 'ss', 'ax', 'new3_ind_st', 'new3_ind_end', 'trnum_org', 'SSQ_b4', 'SSQ_af', 'SSQ_diff', 'FRT_em', 'res_type', 'TR']
        df = pd.DataFrame(out1, index=range(len(row)), columns=columns)

        df_scalarmetics_exp[varr['which_exp']] = df

    return df_scalarmetics_exp
