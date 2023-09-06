# Created by Jamilah Foucher, 21/02/2022

import numpy as np

# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')

from make_a_properlist import *
from string_text_processing.is_sen_in_senarray import *


def unique_str_arrays(senarray):

    ind = []
    for sen in senarray:
        indy, out = is_sen_in_senarray(sen, senarray)
        
        ind.append(indy)
    #print('ind: ', ind)

    ind_new = []
    for i in ind:
        if len(i) > 1:
            # take the first value
            ind_new.append(i[0])
        else:
            ind_new.append(i)
    ind_new = make_a_properlist(np.unique(ind_new))
    #print('ind_new: ', ind_new)

    # create a unique sentence array with the ind_new index
    uq_senarray = []
    for i in ind_new:
        uq_senarray.append(senarray[i])

    return ind_new, uq_senarray