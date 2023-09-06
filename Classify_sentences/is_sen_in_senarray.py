# Created by Jamilah Foucher, 21/02/2022

import numpy as np

# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')

from findall import *

# Examples of how sen and senarray should look
# sen = ['How', 'do', 'I', 'hack', 'Motorola', 'DCX3400', 'for', 'free', 'internet?']

# senarray = [['How', 'can', 'I', 'be', 'a', 'good', 'geologist?'],
# ['What', 'should', 'I', 'do', 'to', 'be', 'a', 'great', 'geologist?']]

# indy is the index in which the sentence was found in the sentence array
# out is the result of whether the sentence is in the sentence array

# senarray must be a 2-positional array

def is_sen_in_senarray(sen, senarray):
    
    #print('sen: ', sen)
    val, indy = findall(senarray, '==', sen)
    indy = np.ravel(indy)
    #print('indy: ', indy)
    
    
    if not any(indy) == True and len(indy) < 1:
        # the array is empty: sen is NOT in senarray
        out = False
    else:
        # the array is NOT empty: sen is in senarray
        out = True
        
    return indy, out