# Created by Jamilah Foucher, 21/02/2022

# Example of names of columns : col_list = ['is_duplicate', 'cosine_sim']
#col_list = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']

import pandas as pd


def pandas_rename_columns(df, col_list):
    
    # Rename columns
    onames = df.columns.to_numpy()
    dictout = {}
    for nf in range(len(col_list)):
        dictout[onames[nf]] = '%s' % (col_list[nf])
    df = df.rename(dictout, axis=1)
    
    return df