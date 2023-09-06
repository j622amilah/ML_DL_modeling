def value_counts_vector(vec):
    
    way = 0
    
    if way == 0:
        # Pandas
        import pandas as pd
        df1 = pd.Series(vec)

        # Remove nan value per row
        # df1.dropna(axis=0)
        # OR
        df1 = my_dropna_python(df1)
        
        # Put the value_counts result in a DataFrame
        df_vc = pd.DataFrame(df1.value_counts())
        
        counted_value = df_vc.iloc[:,0].to_numpy()
        count_index = df_vc.iloc[:,0].index.to_numpy()
        
    elif way == 1:
        # Python
        # Remove nan values from the vector
        vec_nonan = [i for i in vec if  np.isnan(i) == False]

        # module for counting values in a list
        from collections import Counter
        c = Counter(vec_nonan)
        counted_value = []
        count = []
        for cn, cout in c.most_common(len(c)):
            counted_value = counted_value + [cn]
            count = count_index + [cout]
        print('counted_value : ' + str(counted_value))
        print('count_index : ' + str(count))

    out = {}
    for i, val in enumerate(counted_value):
        out[count_index[i]] = val
 
    return out