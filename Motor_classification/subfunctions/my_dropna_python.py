import pandas as pd
import numpy as np
import math


# Personal python functions
from subfunctions.make_a_properlist import *



# np.nan can not process strings,  use this.
def isnan(value):
    try:
        return math.isnan(float(value))
    except:
        return False



def my_dropna_python(df):
    # Python
    col_names = list(df.columns.values)
    # OR
    # col_names = list(df.columns)
    
    df = df.to_numpy()
    df = np.array(df)
    # print('size of df : ', df.shape)
    data = []
    num_of_cols = df.shape[1]
    for i in range(df.shape[0]):
        row_vec = df[i,:]
        
        out = [isnan(row_vec[i]) for i in range(len(row_vec))]
        # OR
        # out = []
        # for i in range(len(row_vec)):
            # #print('row_vec[i]', row_vec[i])
            # out.append(isnan(row_vec[i]))
        
        out = make_a_properlist(out)  # for dataframes with nested arrays
        
        if any(out) == False:
            data.append(df[i,:])
    
    num_of_rows = len(data)
    data0 = np.reshape(data, (num_of_rows, num_of_cols))
    
    df_new = pd.DataFrame(data=data0, columns=col_names)
    
    return df_new