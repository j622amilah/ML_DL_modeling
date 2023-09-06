# Created by Jamilah Foucher, January 26, 2022.

# Purpose: Does the same thing as fillna in pandas.  Fills nan values in a pandas Series with a value of your choice.

# Input VARIABLES:
# (1) df_col: pandas Series with nan values
# (2) fillval : value to replace nan values
# 
# Output VARIABLES:
# (1) df_col2: pandas Series with nan values filled using fillval


import pandas as pd
import numpy as np

import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')
# Personal python functions
from make_a_properlist import *
from isnan import *


def pandas_fillna_percolumn(df_col, fillval):
    
    df_col = df_col.to_numpy()
    df_col = np.array(df_col)
    
    # check entire col
    out = [isnan(df_col[j]) for j in range(len(df_col))]
    
    out = make_a_properlist(out)
        
    df_fill = []
    for ind, val in enumerate(out):
        if val == False:
            # no nan is present
            df_fill.append(df_col[ind])
        else:
            # nan is present: replace with fillval
            df_fill.append(fillval)
    
    # return a dataframe, since the input was a dataframe
    df_col2 = pd.Series(df_fill)
    
    return df_col2