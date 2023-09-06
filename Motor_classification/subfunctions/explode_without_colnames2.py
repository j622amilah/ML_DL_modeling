import numpy as np
import pandas as pd

from subfunctions.make_a_properlist import *

# AFTER mid November 2021 : 
# Pandas did an update with nested arrays ; they force you to use two instead of one

def explode_without_colnames2(df):

    # Get the rows of the DataFrame
    num_of_rows = df.shape[0]

    # Get the columns of the DataFrame
    shape_tuple = df.shape
    if len(shape_tuple) == 1:
        num_of_cols = 1
    else:
        num_of_cols = df.shape[1]
    print('original df - num_of_cols : ', num_of_cols)
    print('original df - num_of_rows : ', num_of_rows)

    vec = []
    for i in range(num_of_cols):
        temp = []
        for j in range(num_of_rows):
            #print('j : ', j)
            temp.append(df.iloc[j:j+1,i:i+1].to_numpy()[0][0])
        vec.append(make_a_properlist(temp))

    print('new df - num_of_cols : ', len(vec))
    print('new df - num_of_rows : ', len(vec[0]))

    df = pd.DataFrame(vec)
    df = df.T # or df1.transpose()
    print('shape of df : ', df.shape)
    
    return df