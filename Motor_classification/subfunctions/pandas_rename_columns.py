import pandas as pd
import numpy as np

from collections import Counter


def pandas_rename_columns(df, col_list):
    
    way = 1
    
    if way == 0:
        # Façon difficile
        # Rename columns
        onames = df.columns.to_numpy()
        dictout = {}
        for nf in range(len(col_list)):
            dictout[onames[nf]] = '%s' % (col_list[nf])
            
        # Determinez quels columns de df repeter
        uq = Counter(onames).most_common()
        d = {}
        for i in range(len(uq)):
            temp = []
            for ind, val in enumerate(onames):
                if uq[i][0] == val:
                    temp.append(col_list[ind])
            d[i] = temp    
        
        # if the column name is a key of d pop the names in the list, else return the column name
        df.rename(columns=lambda c: d[c].pop(0) if c in d.keys() else c)
    
    elif way == 1:
        # Façon facile
        df.columns = col_list
    
    return df
