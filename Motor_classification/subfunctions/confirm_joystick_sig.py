def confirm_joystick_sig(feat0):
    # Realized that we need to check the correctness of the data signal AGAIN
    # Some of the signals look like cabin data movement
    
    # This joystick signals should start around 0 for each trial
    marg = 0.1
    temp = []
    gard = []
    for i in range(len(feat0)):
        if feat0[i,0] < marg and feat0[i,0] > -marg:
            temp.append(feat[i] - feat[i,0]) # baseline shift signal to zero
            gard.append(i)
    
    return temp, gard
