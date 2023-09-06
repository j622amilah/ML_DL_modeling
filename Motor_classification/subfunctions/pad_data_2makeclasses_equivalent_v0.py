import pandas as pd
import numpy as np

def pad_data_2makeclasses_equivalent(df_test):

    # DataFrame is organized : [X, y]  where each of X columns are nested arrays

    # ----------------

    # Remove nan value per row
    # df_test.dropna(axis=0)
    # OR
    df_test_noNan = my_dropna_python(df_test)

    #del df_test
    
    # ----------------
    
    # DataFrame is organized : [X, y]  where each of X columns are nested arrays

    num_of_feat = df_test_noNan.shape[1] - 1

    # ----------------

    # Add on a scalar column of the y label, for searching
    val = []
    for i in range(len(df_test_noNan.iloc[:,-1])):
        val.append(df_test_noNan.iloc[i,-1][0])
    print('length of val : ', len(val))
    valS = pd.Series(val)

    df_test_noNan = pd.concat([df_test_noNan, valS], axis=1)
    df_test_noNan = df_test_noNan.rename({0: 'y_scalar'}, axis=1)

    # ----------------

    # Count the unique values in the label
    df_vc = pd.DataFrame(df_test_noNan.y_scalar.value_counts())
    df_vc.head()
    counted_value = df_vc.iloc[:,0].to_numpy()
    count_index = df_vc.iloc[:,0].index.to_numpy()
    print('Before sorting counted_value : ', counted_value)
    print('Before sorting count_index : ', count_index)
    
    # Sort counted_value by count_index; in ascending order
    sind = np.argsort(count_index)
    count_index = count_index[sind]
    counted_value = counted_value[sind]
    print('After sorting counted_value : ', counted_value)
    print('After sorting count_index : ', count_index)
    
    out = {}
    for i, val in enumerate(counted_value):
        out[count_index[i]] = val
    print('Original class count : ', out)

    # ----------------

    # Determine how much to pad each class label
    needed_samps_class = np.max(counted_value) - counted_value
    print('needed_samps_class : ', needed_samps_class)

    # ----------------

    # Confirm that there are no nan values
    out = df_test_noNan.isnull().values.any()
    print('Are there nan valeus in the data : ', out)

    # ----------------
    
    print('shape of dataframe before padding : ', df_test_noNan.shape)

    # ----------------
    # Pad the DataFrame
    n_classes = len(count_index)
    n_samples = len(df_test_noNan)

    df_2add_on = pd.DataFrame()

    for i in range(n_classes):
        print('i : ', i)
        # Pad short length classes
        for j in range(needed_samps_class[i]):
            print('j : ', j) 
            flag = 0
            while flag == 0:
                permvec = np.random.permutation(n_samples)
                index = permvec[0]  #random choosen index

                # look for each class
                if i == int(df_test_noNan.y_scalar.iloc[index]):
                    print('Class match was found : i = ', i, ', data index = ', int(df_test_noNan.y_scalar.iloc[index]), ', index = ', index)
                    
                    # Append the data with padded data entry
                    data = []
                    for nf in range(num_of_feat+2):
                        if nf == num_of_feat+1:
                            # The last column is a scalar - can not use list
                            data.append([df_test_noNan.iloc[index, nf]])
                        else:
                            data.append([list(df_test_noNan.iloc[index, nf])])

                    df_row = pd.DataFrame(data=data)
                    df_row = df_row.T # or df1.transpose()
                    df_2add_on = pd.concat([df_2add_on, df_row], axis=0)

                    flag = 1 # to brake while

    # ----------------

    df_2add_on = df_2add_on.reset_index(drop=True)  # reset index : delete the old index column

    dfl = {}
    for nf in range(num_of_feat+2):
        if nf == num_of_feat+1:
            dfl[nf] = 'y_scalar'
        elif nf == num_of_feat:
            dfl[nf] = 'y'
        else:
            dfl[nf] = '%d' % (nf)

    df_2add_on = df_2add_on.rename(dfl, axis=1)
    
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
    df_vc = pd.DataFrame(df_test2.y_scalar.value_counts())
    df_vc.head()
    counted_value = df_vc.iloc[:,0].to_numpy()
    count_index = df_vc.iloc[:,0].index.to_numpy()
    print('Before sorting counted_value : ', counted_value)
    print('Before sorting count_index : ', count_index)
    
    # Sort counted_value by count_index; in ascending order
    sind = np.argsort(count_index)
    count_index = count_index[sind]
    counted_value = counted_value[sind]
    print('After sorting counted_value : ', counted_value)
    print('After sorting count_index : ', count_index)
    
    out = {}
    for i, val in enumerate(counted_value):
        out[count_index[i]] = val
    print('New class count : ', out)

    # Determine how much to pad each class label
    needed_samps_class = np.max(counted_value) - counted_value
    print('needed_samps_class : ', needed_samps_class)

    # ----------------
    
	return df_test2