# Created by Jamilah Foucher, 26/02/2022

import numpy as np

# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')


from findall import *
from is_empty import *

from string_text_processing.make_a_properlist_str import *


def sen_to_onehot(sen, keywords):
    sen = make_a_properlist_str(sen)
    
    X = np.zeros((len(keywords), 1))
    c = 0
    for word in sen:
        newvec, ind_newvec = findall(keywords, '==', word)

        if is_empty(ind_newvec) == False:
            for i in ind_newvec:
                # print('i : ', i)
                # print('c : ', c)
                X[i,c] = 1
    c = c + 1

    X = X.T
    print('size of X (sentences, keywords): ', X.shape)
    
    return X