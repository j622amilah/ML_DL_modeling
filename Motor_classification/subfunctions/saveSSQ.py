import numpy as np


def saveSSQ(s, which_exp):

    # Save SSQ data by trial in the dat. structure
    # SSQ : [nausee, oculo_moteur, disorientation, total sickness]
    
    if which_exp == 'rot':
        if s == 0:		# GV
            before = np.array([0, 5, 2, 5])
            after = np.array([0, 4, 1, 4])
        elif s == 1:	# AW
            before = np.array([2, 2, 1, 4])
            after = np.array([2, 2, 1, 4])
        elif s == 2:	# CDV
            before = np.array([0, 2, 0, 2])
            after = np.array([0, 4, 0, 4])
        elif s == 3:	# LB
            before = np.array([0, 3, 1, 3])
            after = np.array([0, 4, 2, 4])
        elif s == 4:	# PJ
            before = np.array([1, 0, 0, 1])
            after = np.array([0, 0, 0, 0])
        elif s == 5:	# PN
            before = np.array([1, 6, 4, 7])
            after = np.array([4, 8, 6, 12])
        elif s == 6:	# DL
            before = np.array([0, 0, 0, 0])
            after = np.array([0, 0, 0, 0])
        elif s == 7:	# SS
            before = np.array([0, 1, 0, 1])
            after = np.array([0, 1, 0, 1])
        elif s == 8:	# MD
            before = np.array([2, 6, 4, 8])
            after = np.array([0, 2, 0, 2])
        elif s == 9:	# CB
            before = np.array([0, 3, 0, 3])
            after = np.array([0, 4, 0, 4])
        elif s == 10:	# PI
            before = np.array([0, 1, 0, 1])
            after = np.array([0, 1, 0, 1])
        elif s == 11:	# FD
            before = np.array([0, 0, 0, 0])
            after = np.array([0, 0, 0, 0])
        elif s == 12:	# JMF
            before = np.array([0, 0, 0, 0])
            after = np.array([0, 0, 0, 0])
        elif s == 13:	# LB
            before = np.array([1, 3, 0, 4])
            after = np.array([0, 0, 0, 0])
        elif s == 14:	# LM
            before = np.array([1, 7, 4, 8])
            after = np.array([1, 5, 3, 6])
        elif s == 15:	# MBC
            before = np.array([0, 0, 0, 0])
            after = np.array([1, 0, 1, 1])
        elif s == 16:	# PB
            before = np.array([0, 0, 0, 0])
            after = np.array([6, 6, 5, 12])
        elif s == 17:	# SA
            before = np.array([0, 1, 0, 1])
            after = np.array([2, 3, 2, 5])
           
    elif which_exp == 'trans':
        if s == 0:
            before = np.array([2, 5, 2, 7])
            after = np.array([2, 4, 2, 6])
        elif s == 1:
            before = np.array([0, 1, 0, 1])
            after = np.array([0, 2, 0, 2])
        elif s == 2:
            before = np.array([2, 3, 2, 5])
            after = np.array([0, 3, 2, 3])
        elif s == 3:
            before = np.array([0, 0, 0, 0])
            after = np.array([0, 1, 0, 1])
        elif s == 4:
            before = np.array([1, 0, 0, 1])
            after = np.array([4, 3, 3, 7])
        elif s == 5:
            before = np.array([0, 4, 1, 4])
            after = np.array([1, 6, 3, 7])
        elif s == 6:
            before = np.array([2, 1, 2, 3])
            after = np.array([0, 3, 0, 3])
        elif s == 7:
            before = np.array([0, 0, 0, 0])
            after = np.array([0, 0, 0, 0])
        elif s == 8:
            before = np.array([0, 1, 0, 1])
            after = np.array([1, 3, 0, 4])
        elif s == 9:
            before = np.array([0, 0, 0, 0])
            after = np.array([0, 0, 0, 0])
        elif s == 10:
            before = np.array([0, 0, 0, 0])
            after = np.array([0, 0, 0, 0])
        elif s == 11:
            before = np.array([2, 4, 1, 6])
            after = np.array([2, 1, 1, 3])
        elif s == 12:
            before = np.array([2, 1, 1, 3])
            after = np.array([3, 3, 2, 6])
        elif s == 13:
            before = np.array([1, 0, 0, 1])
            after = np.array([1, 0, 0, 1])
     
    print('before : ' + str(before))
    
    print('after : ' + str(after))
    
    
    diff = before - after
    
    return [before, after, diff]