import numpy as np
import pandas as pd

from subfunctions.make_a_properlist import *

# Before mid November 2021, this exploded a matrix automatically
# Pandas did an update and it no longer works

def explode_without_colnames1(df):

    # Get the rows of the DataFrame
    num_of_rows = df.shape[0]
    #print('num_of_rows : ', num_of_rows)

    # Get the columns of the DataFrame
    shape_tuple = df.shape
    if len(shape_tuple) == 1:
        num_of_cols = 1
    else:
        num_of_cols = df.shape[1]

    vals = []
    for i in range(num_of_cols):
        #print('i : ', i)
        temp = []
        for j in range(num_of_rows):
            #print('j : ', j)
            temp.append(make_a_properlist(df.iloc[j:j+1,i:i+1].to_numpy()))
        vals.append(make_a_properlist(temp))
    X = np.transpose(vals)
    #print('shape of X : ', X.shape)
    #X = np.reshape(vals, (-1,num_of_cols))

    df = pd.DataFrame(X)
    print('shape of df : ', df.shape)
    
    return df