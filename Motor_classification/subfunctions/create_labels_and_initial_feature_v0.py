def create_labels_and_initial_feature(df):

    # Add a direction column to the dataframe, for each of the bins
    ts = 0.1
    start_val = 0

    e18 = []
    for i in range(len(df)):
        # This is the joystick
        # KEY : Baseline shift the signal to zero wrt the first point for t-test statistic
        y = df.sig_win_out.iloc[0][0] - df.sig_win_out.iloc[i]   # .to_numpy() # says I don't need it, but not logical
        # print('y : ', y)

        # NOTE : at this point, some binned signals may be slightly shorter than others.
        # The measures below are not effected by signal length, so leave length for this step
        # and interpolate the lengths during the next step (feature preprocess/creation)
        N = len(y)  # this is why we use the length of y instead of binsize
        x = np.multiply(range(start_val, N), ts)

        # Direction : Take the slope of each 10 bin trace
        deg = 2
        p = np.polyfit(x, y, deg)
        slope = p[0]
        # print('slope : ', slope)
        e18.append(slope)   # slope
        
    # ---------------
        
    # Significance measure 0 : standard deviation from mean (to make stationary boundary)
    # Calculate the mean and 1 std from mean
    n = 1
    mean_pop_slope = np.mean(np.abs(e18))
    print('mean_pop_slope : ', mean_pop_slope)
    std_pop_slope = np.std(np.abs(e18))*n
    print('std_pop_slope : ', std_pop_slope)
    sigthresh_std = mean_pop_slope + std_pop_slope
    print('sigthresh_std : ', sigthresh_std)
        
    # ---------------
        
    # Significance measure 1 : t-test
    sigthresh = 0.05  # sig increase/decrease = pvalue significance of each windowed data signal wrt zero
    e19 = []
    e20 = []
    e21 = []
    for i in range(len(df)):
        # This is the joystick
        # KEY : Baseline shift the signal to zero wrt the first point for t-test statistic
        y = df.sig_win_out.iloc[0][0] - df.sig_win_out.iloc[i]   # .to_numpy() # says I don't need it, but not logical
        
        # Signficance measure 0: standard deviation from abs slope mean
        if np.abs(e18[i]) < sigthresh_std:   # stationary/bounded movement
            lab_val0 = 0   # 0=stationary
        elif np.abs(e18[i]) > sigthresh_std and np.sign(e18[i]) > 0:  # positive outlier
            lab_val0 = 1   # 1=increase
        elif np.abs(e18[i]) > sigthresh_std and np.sign(e18[i]) < 0:  # negative outlier
            lab_val0 = 2   # 2=decrease
        e19.append(lab_val0)

        # Signficance measure 1: Compare the 10 bin trace to zero - t-test 
        pop_mean = 0  # Deadzone was set to 0.1
        
        # This gives nan, if np.abs(y) equals a vector of zeros
        if np.sum(np.abs(y)) == 0:
            # np.abs(y) equals a vector of zeros, 
            # null hypothesis is accepted (the mean is equal to zero)
            sigval = 1
        else:
            result = stats.ttest_1samp(np.abs(y), pop_mean)
            sigval = result.pvalue
        # print('sigval : ', sigval)
        e20.append(sigval)   # the significance value with respect to the pop_mean value

        # Label bin :
        if np.abs(sigval) >= sigthresh:
            lab_val = 0   # 0=stationary (dir=does not matter, not sig)
        elif np.sign(e18[i]) > 0 and np.abs(sigval) < sigthresh:
            lab_val = 1   # 1=increase (dir=pos, sig)
        elif np.sign(e18[i]) < 0 and np.abs(sigval) < sigthresh:
            lab_val = 2   # 2=decrease (dir=neg, sig)
        e21.append(lab_val)

    # ---------------

    # Put the bin analysis data into DataFrame
    data = e18, e19, e20, e21
    data = np.transpose(data)
    columns = ['slope', 'asm_label', 't-testsig', 'tt_label']
    df_binana = pd.DataFrame(data=data, columns=columns)

    # NOTE : we save over df
    df = pd.concat([df, df_binana], axis=1)
    df = df.rename({0: 'subject', 1: 'tr', 2: 'ss', 3: 'ax', 4: 'i_st', 5: 'i_end', 6: 't_st', 7: 't_end', 8: 'TR', 9: 'res_type', 10: 'i_TR', 11: 'featnum', 12: 'winval', 13: 'abs_sum', 14: 'abs_sumMSE', 15: 'rel_sum', 16: 'rel_sumMSE', 17: 'sig_win_out', 18: 'slope', 19: 'asm_label', 20: 'ttestsig', 21: 'tt_label'}, axis=1)
    # df.head()

    # ---------------
        
    dat = df.sig_win_out.to_numpy()
    
    lvec = []
    for i in range(len(dat)):
        lvec.append(len(dat[i]))
    print('min lvec : ', np.min(lvec))
    print('max lvec : ', np.max(lvec))
    lvecmax = np.max(lvec)
    
    # ---------------
    
    # Features per windows (micro-signatures)
    # Interpolate : make each sub-set of data the same number of data points
    feat0 = []
    y1_feat0 = []
    y2_feat0 = []

    ts = 0.1
    start_val = 0
    t_feat0 = np.multiply(range(start_val, lvecmax), ts)

    for i in range(len(df)):
        
        sSIG = dat[i]
        y1 = np.ravel(df.asm_label.to_numpy()[i]*np.ones((len(sSIG), 1)))
        y2 = np.ravel(df.tt_label.to_numpy()[i]*np.ones((len(sSIG), 1)))

        # Check if trial data is less than the maximum length
        if len(dat[i]) < np.max(lvec):
            # Do interpolation : X
            # The trial length is different so interpolate the time-series to make them the same length signal 
            # NOTE : you give the first point of sSIG as the start_val
            x = np.linspace(sSIG[0], len(sSIG), num=len(sSIG), endpoint=True)
            xnew = np.linspace(sSIG[0], len(sSIG), num=lvecmax, endpoint=True)

            # joystick on stim
            f = interp1d(x, sSIG)
            sSIGl = f(xnew)

            # y1
            f = interp1d(x, y1)
            y1_sSIGl = f(xnew)

            # y2
            f = interp1d(x, y2)
            y2_sSIGl = f(xnew)

            # python : you can not create a matrix in real-time in pandas
            # you only assign the full matrix at the end
            # (0) position
            feat0 = feat0 + [sSIGl]
            y1_feat0 = y1_feat0 + [y1_sSIGl]
            y2_feat0 = y2_feat0 + [y2_sSIGl]
            
            # Clean up
            del x, f, sSIGl, y1_sSIGl, y2_sSIGl
        else:
            # Trial data is the same length as the maximum length
            feat0 = feat0 + [sSIG]
            y1_feat0 = y1_feat0 + [y1]
            y2_feat0 = y2_feat0 + [y2]
    # -------------------------------------
    
    return feat0, t_feat0, y1_feat0, y2_feat0, df