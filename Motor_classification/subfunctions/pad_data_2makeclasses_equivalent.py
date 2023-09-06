import pandas as pd
import numpy as np

from subfunctions.my_dropna_python import *
from subfunctions.pandas_rename_columns import *
from subfunctions.count_classes import *




def pad_data_2makeclasses_equivalent(df_feat):
    # Remove nan value per row
    # df_test_noNan = df_feat.dropna(axis=0)
    # OR
    df_test_noNan = my_dropna_python(df_feat)

    # ----------------

    # Confirm that there are no nan values
    out = df_test_noNan.isnull().values.any()
    print('Are there nan valeus in the data : ', out)

    # ----------------

    # Check class balance
    needed_samps_class, counted_value, count_index, st, endd = count_classes(df_test_noNan)

    # ----------------

    print('shape of dataframe before padding : ', df_test_noNan.shape)

    # ----------------
    # Pad the DataFrame
    n_classes = len(count_index)
    n_samples = len(st)

    df_2add_on = pd.DataFrame()
    
    # Le derniere sample dans df_test_noNan
    df_coupe_proche = df_test_noNan.iloc[st[-1]:endd[-1], :]

    for i in range(n_classes):
        #print('i : ', i)
        # Pad short length classes
        for j in range(needed_samps_class[i]):
            #print('j : ', j) 
            flag = 0
            while flag == 0:
                permvec = np.random.permutation(n_samples)
                index = permvec[0]  #random choosen index
                
                # look for each class : on veut le classe être le meme
                if i == int(df_test_noNan.y.iloc[st[index]]):
                    #print('Class match was found : i = ', i, ', data index = ', int(df_test_noNan.y_scalar.iloc[index]), ', index = ', index)
                    
                    # Append the data with padded data entry
                    df_coupe = df_test_noNan.iloc[st[index]:endd[index], :]
                    
                    # Le derriere sample ne sont pas le meme que le sample actuelle
                    if int(df_coupe.iloc[0,0] - df_coupe_proche.iloc[0,0]) != 0:
                        df_coupe_proche = df_coupe
                        df_2add_on = pd.concat([df_2add_on, df_coupe], axis=0)
                        flag = 1 # to brake while
                        
    # ----------------

    # DataFrame a besoin les noms de columns d'avoir le meme noms que df_test_noNan
    df_2add_on = df_2add_on.reset_index(drop=True)  # reset index : delete the old index column

    col_list = df_test_noNan.columns
    df_2add_on = pandas_rename_columns(df_2add_on, col_list)
    df_2add_on

    print('shape of dataframe to add to original dataframe: ', df_2add_on.shape)

    # ----------------

    # want to arrange the dataframe with respect to rows (stack on top of the other): so axis=0 
    # OR think of it as the rows of the df change so you put axis=0 for rows
    df_test2 = pd.concat([df_test_noNan, df_2add_on], axis=0)
    df_test2 = df_test2.reset_index(drop=True)  # reset index : delete the old index column

    print('shape of padded dataframe (original + toadd) : ', df_test2.shape)

    del df_test_noNan, df_2add_on

    # ----------------

    # Final check of class balance
    needed_samps_class, counted_value, count_index, st, endd = count_classes(df_test2)

    # ----------------
    
    # Enlevez des columns unnecessaires : num, y
    # df_test2 = df_test2.drop(['num', 'y'], axis=1)  # 1 is the axis number (0 for rows and 1 for columns)

    # ----------------

    # Rename each feature column a number, and the label column with y 
    # col_list = list(map(str, np.arange(len(df_test2.columns) - 1)))
    # col_list.append('y')
    # df_test2 = pandas_rename_columns(df_test2, col_list)
    
    return df_test2
