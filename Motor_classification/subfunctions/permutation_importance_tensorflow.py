import numpy as np


def sort_dict_by_value(d, reverse = False):
    return dict(sorted(d.items(), key = lambda x: x[1], reverse = reverse))



def permutation_importance_tensorflow(model, X_test, Y_test):

    Y_test_1D = [Y_test[i,0:1] for i in range(Y_test.shape[0])]

    # First, a baseline metric, defined by scoring,
    # Obtenez mean absolute error
    y_hat_test = model.predict(X_test, verbose=0)
    baseline_mae = np.mean(np.abs(y_hat_test - Y_test_1D))

    vals = {}
    # Shuffle each feature columns at a time
    for featcol in range(X_test.shape[2]):

        # Define a modifiable temporary variable
        temp = X_test

        # select a column
        feat_slice = temp[:,:,featcol]

        # Must flatten the matrix because np.random.permutation or 
        # np.random.shuffle don't work
        t = feat_slice.flatten()
        t_shuf = np.random.permutation(t)
        feat_slice =  np.reshape(t_shuf, (feat_slice.shape))

        # put feat_slice back into temp
        temp[:,:,featcol] = feat_slice

        y_hat_test = model.predict(temp, verbose=0)
        mae_per_col = np.mean(np.abs(y_hat_test - Y_test_1D))
        vals[featcol] = mae_per_col

    # Sort the columns from largest to smallest mae
    laquelle = sort_dict_by_value(vals, reverse = True)
    
    # Determinez le nombres des columns qui sont plus grande que le baseline_mae
    # C'est des marqueurs qui sont importants
    feat = list(laquelle.keys())
    cnt = [1 for i in range(len(feat)) if feat[i] > baseline_mae]
    cnt = np.sum(cnt)
    
    allout = list(laquelle.items())
    nout = [allout[i] for i in range(cnt)]
    marquers_important = dict(nout)
    
    return marquers_important
