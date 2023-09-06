# Created by Jamilah Foucher, May 09, 2021 

# Purpose: to quickly use the scikit-learn python machine learning toolbox.  The most frequently used regression and classification methods are outlined such that they can be quickly used.



# ----------------------------------------------
def plot_feature_space_2D(X, y):
    
    import matplotlib.pyplot as plt
    
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Plot the first two features
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    
    return
# ----------------------------------------------


# ----------------------------------------------
def plot_feature_space_3D_PCA(X, y):
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA
    
    # To get a better understanding of interaction of the dimensions plot the first three PCA dimensions
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    
    # PCA of the first two features
    X_reduced = PCA(n_components=3).fit_transform(X)

    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap=plt.cm.Set1, edgecolor="k", s=40)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()
    
    return
# ----------------------------------------------


# Pre-treating features or labels : 


# ----------------------------------------------
def do_train_test_split(X, y):
    
    from sklearn.model_selection import train_test_split
    import numpy as np
    seed = 0
    X_train, X_test, Y_train_1D, Y_test_1D = train_test_split(X, y, random_state = seed)

    X_train = np.array(X_train)
    print('shape of X_train : ', X_train.shape)

    Y_train_1D = np.array(Y_train_1D)
    print('shape of Y_train_1D : ', Y_train_1D.shape)

    X_test = np.array(X_test)
    print('shape of X_test : ', X_test.shape)

    Y_test_1D = np.array(Y_test_1D)
    print('shape of Y_test_1D : ', Y_test_1D.shape)
    
    return X_train, X_test, Y_train_1D, Y_test_1D
# ----------------------------------------------



# ----------------------------------------------
def check_if_Y_1D_is_correct(Y_1D):
    
    import matplotlib.pyplot as plt
    fig, (ax0) = plt.subplots(1)

    ax0.plot(Y_1D[:], 'b-', label='Y_1D')
    ax0.set_ylabel('Y_1D')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    return
# ----------------------------------------------


# ----------------------------------------------
def binarize_Y1Dvec_2_Ybin(Y):
    
    import numpy as np
    
    # Transform a 1D Y vector (n_samples by 1) to a Y_bin (n_samples by n_classes) vector

    # Ensure vector is of integers
    Y = [int(i) for i in Y]

    # Number of samples
    m_examples = len(Y)

    # Number of classes
    temp = np.unique(Y)
    unique_classes = [int(i) for i in temp]
    # print('unique_classes : ', unique_classes)

    whichone = 2
    # Binarize the output
    if whichone == 0:
        from sklearn.preprocessing import label_binarize
        Y_bin = label_binarize(Y, classes=unique_classes)  # does not work

    elif whichone == 1:
        from sklearn import preprocessing
        lb = preprocessing.LabelBinarizer()
        Y_bin = lb.fit_transform(Y)
        
    elif whichone == 2:
        # By hand
        Y_bin = np.zeros((m_examples, len(unique_classes)))
        for i in range(0, m_examples):
            if Y[i] == unique_classes[0]:
                Y_bin[i,0] = 1
            elif Y[i] == unique_classes[1]:
                Y_bin[i,1] = 1
            elif Y[i] == unique_classes[2]:
                Y_bin[i,2] = 1
            elif Y[i] == unique_classes[3]:
                Y_bin[i,3] = 1
            elif Y[i] == unique_classes[4]:
                Y_bin[i,4] = 1
            elif Y[i] == unique_classes[5]:
                Y_bin[i,5] = 1
            elif Y[i] == unique_classes[6]:
                Y_bin[i,6] = 1
                
    print('shape of Y_bin : ', Y_bin.shape)

    return Y_bin, unique_classes
# ----------------------------------------------


# ----------------------------------------------
def debinarize_Ybin_2_Y1Dvec(Y_bin):

    import numpy as np

    # Transform a Y_bin (n_samples by n_classes) vector to a 1D Y vector (n_samples by 1)

    # De-Binarize the output
    Y = np.argmax(Y_bin, axis=1)

    return Y
# ----------------------------------------------


# ----------------------------------------------
def transform_Y_bin_pp_2_Y_1D_pp(Y_1D, Y_bin_pp):

    # Y_bin_pp is size [n_samples, n_classes=2]
    # Take the column of Y_bin_pp for the class of Y_1D, because both vectors need to be [n_samples, 1]
    import numpy as np
    Y_1D_pp = []
    for q in range(len(Y_1D)):
        desrow = Y_bin_pp[q]
        Y_1D_pp.append(desrow[int(Y_1D[q])])
    Y_1D_pp = np.ravel(Y_1D_pp)
    
    Y_1D_pp = np.array(Y_1D_pp)
    
    return Y_1D_pp

# ----------------------------------------------

# Principle Component Analysis (PCA)
def get_pca(X_train, n_components):

    # X_train is size [n_samples, n_features]
    # n_components is a scalar, the number of dominate eigenvectors/components that make up the data

    from sklearn.decomposition import PCA

    # Get the n_components principle componets of the data X_train
    X_PCA = PCA(n_components=n_components).fit_transform(X_train)
    
    import matplotlib.pyplot as plt
    plt.plot(X_PCA[:])
    plt.title('PCA')
    plt.xlabel('samples')
    plt.ylabel('magnitude')
    plt.show()
    
    return X_PCA

# ----------------------------------------------

# Canonical Correlation Analysis (CCA)
def get_cca(X_train, Y_train, n_components):

    # X_train is size [n_samples, n_features]
    # Y_train is size (n_samples, n_classes)
    # n_components is a scalar, the number of dominate components that make up the data

    from sklearn.cross_decomposition import CCA

    # Get the n_components principle componets of the data X_train
    X_CCA = CCA(n_components=n_components).fit(X_train, Y_train).transform(X_train)
    
    import matplotlib.pyplot as plt
    plt.plot(X_CCA[:])
    plt.set_title('CCA')
    plt.set_xlabel('samples')
    plt.set_ylabel('magnitude')
    plt.show()

    return X_CCA

# ----------------------------------------------




# Machine Learning models :

# ----------------------------------------------
# multi-class Stochastic Gradient Descent
def multiclass_stochastic_gradient_descent(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size (n_samples, n_features)
    # X_test is size (n_samples, n_features)
    # Y_train_1D is size (n_samples, 1) where the n_classes are represented by 1,2,3,..., etc
    
    from sklearn.linear_model import SGDClassifier
    from sklearn.calibration import CalibratedClassifierCV
    
    # Can ONLY calculate the probability estimates (predict_proba), but NOT the decision_function
    lr = SGDClassifier(loss='hinge', alpha=0.001, class_weight='balanced')
    clf =lr.fit(X_train, Y_train_1D)
    calibrator = CalibratedClassifierCV(clf, cv='prefit')
    model = calibrator.fit(X_train, Y_train_1D)
    
    # -------
    # Y_predict : size [n_samples, 1]
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    # -------
    
    # -------
    # The prediction probability of each class : size is [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------
    
    # probability estimates (predict_proba) are not available for loss='hinge', but can calculate the decision_function
    model = SGDClassifier(alpha=0.001, loss='hinge') 
    model.fit(X_train, Y_train_1D)
    
    # -------
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    #  size is [n_samples, n_classes]
    Y_train_bin_score = model.decision_function(X_train)
    Y_test_bin_score = model.decision_function(X_test)

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    # -------

    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score
    
# ----------------------------------------------

# Multi-class Linear Discriminant Analysis (LDA)
def multiclass_LDA_classifier(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    model = LinearDiscriminantAnalysis()

    model.fit(X_train, Y_train_1D)

    # -------
    # Y_predict : size [n_samples, 1]
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    # -------
    
    # -------
    # The prediction probability of each class : size is [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------

    # -------
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    #  size is [n_samples, n_classes]
    Y_train_bin_score = model.decision_function(X_train)
    Y_test_bin_score = model.decision_function(X_test)

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    # -------

    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score
    
# ----------------------------------------------

# Multi-class Support Vector Machine (SVC)
def multiclass_svm(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    # -------
    # Need to binarize Y into size [n_samples, n_classes]
    Y_train_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_train_1D)
    Y_test_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_test_1D)

    Y_train_bin = np.array(Y_train_bin)
    print('shape of Y_train_bin : ', Y_train_bin.shape)

    Y_test_bin = np.array(Y_test_bin)
    print('shape of Y_test_bin : ', Y_test_bin.shape)
    # -------
    
    from sklearn import svm
    from sklearn.multiclass import OneVsRestClassifier
    
    model = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, class_weight=None, max_iter=100))  #, random_state=random_state
    # OR
    # model = svm.SVC(decision_function_shape='ovr', probability=True, max_iter=100)  # “one-versus-rest” : multi-class (need to try **)
    # Y_train_bin is (n_samples, n_classes), this transforms the ovo setup into a ovr setup
    # OR
    # model = svm.LinearSVC() # “one-versus-rest” : multi-class (need to try **) Y_train_bin is (n_samples, n_classes), direct ovr setup
    
    model.fit(X_train, Y_train_bin)
    
    Y_train_bin_predict = model.predict(X_train)
    Y_test_bin_predict = model.predict(X_test)
    #print('Y_test_bin_predict : ', Y_test_bin_predict)

    # The prediction probability of each class
    Y_train_bin_pp = model.predict_proba(X_train) # size is [n_samples, n_classes]
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)

    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    Y_train_bin_score = model.decision_function(X_train)  # size is n_samples, 1
    Y_test_bin_score = model.decision_function(X_test)

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    
    # Y_train_bin and Y_test_bin needs to be used in the evaluation of the model ALSO!!
    
    return model, Y_train_bin, Y_test_bin, Y_train_bin_predict, Y_test_bin_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score
    
# ----------------------------------------------

# Multi-class Support Vector Machine (NuSVC)
def multiclass_svm_NuSVC(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    from sklearn import svm
    
    model = svm.NuSVC(decision_function_shape='ovr', probability=True, max_iter=1) 
    #  class_weight='auto',
    # “one-versus-rest” : multi-class Y_train_bin is (n_samples, n_classes), direct ovr setup
    
    model.fit(X_train, Y_train_1D)
    
    # -------
    # Y_predict : size [n_samples, 1]
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    # -------
    
    # -------
    # The prediction probability of each class : size is [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------

    # -------
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    #  size is [n_samples, n_classes]
    Y_train_bin_score = model.decision_function(X_train)
    Y_test_bin_score = model.decision_function(X_test)

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    # -------
    
    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score
    
# ----------------------------------------------

# Faster Multi-class Support Vector Machine (SVC) : Bagging
def multiclass_svm_bagging(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    # -------
    # Need to binarize Y into size [n_samples, n_classes]
    Y_train_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_train_1D)
    Y_test_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_test_1D)

    Y_train_bin = np.array(Y_train_bin)
    print('shape of Y_train_bin : ', Y_train_bin.shape)

    Y_test_bin = np.array(Y_test_bin)
    print('shape of Y_test_bin : ', Y_test_bin.shape)
    # -------
    
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC

    n_estimators = 10
    #max_samples= 1.0 / n_estimators  # max_samples must be in (0, n_samples]
    #print('max_samples : ', max_samples)
    #model = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='auto'), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
    model = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight=None, max_iter=100), n_estimators=n_estimators))

    model.fit(X_train, Y_train_bin)

    Y_train_bin_predict = model.predict(X_train)
    Y_test_bin_predict = model.predict(X_test)
    #print('Y_test_bin_predict : ', Y_test_bin_predict)

    # The prediction probability of each class
    Y_train_bin_pp = model.predict_proba(X_train) # size is [n_samples, n_classes]
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)

    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    Y_train_bin_score = model.decision_function(X_train)  # size is n_samples, 1
    Y_test_bin_score = model.decision_function(X_test)

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    
    # Y_train_bin and Y_test_bin needs to be used in the evaluation of the model ALSO!!
    
    #score(X, y[, sample_weight]) Return the mean accuracy on the given test data and labels.
    
    return model, Y_train_bin, Y_test_bin, Y_train_bin_predict, Y_test_bin_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score
    
# ----------------------------------------------

# Multi-class RandomForest
def multiclass_RandomForest_1Dinput(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # -------
    
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(random_state=1, min_samples_leaf=50)  # min_samples_leaf is 100 by default
    model = MultiOutputClassifier(forest, n_jobs=-1) #n_jobs=-1 means apply parallel processing
    
    Y_train_1D = np.reshape(Y_train_1D, (len(Y_train_1D), 1))  # Y needs to have a defined shape ***
    model.fit(X_train, Y_train_1D)
    
    
    # -------
    # Y_predict : size [n_samples, 1]
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    # -------
    
    # -------
    # The prediction probability of each class : size is [1, n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    unique_classes = np.unique(Y_train_1D)
    Y_train_bin_pp = np.reshape(Y_train_bin_pp, (len(Y_train_1D), len(unique_classes)))
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.reshape(Y_test_bin_pp, (len(Y_test_1D), len(unique_classes)))
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------
    
    # There is NO decision_function
    # ------------------------------
    Y_train_bin_score = Y_train_bin_pp
    Y_test_bin_score = Y_test_bin_pp
    
    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    
    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score

# ----------------------------------------------

# Multi-class RandomForest : is the same as the 1Dinput
def multiclass_RandomForest_bininput(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    # -------
    # Need to binarize Y into size [n_samples, n_classes]
    Y_train_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_train_1D)
    Y_test_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_test_1D)

    Y_train_bin = np.array(Y_train_bin)
    print('shape of Y_train_bin : ', Y_train_bin.shape)

    Y_test_bin = np.array(Y_test_bin)
    print('shape of Y_test_bin : ', Y_test_bin.shape)
    # -------
    
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(random_state=1, min_samples_leaf=50)  # min_samples_leaf is 100 by default
    model = MultiOutputClassifier(forest, n_jobs=-1) #n_jobs=-1 means apply parallel processing
    
    model.fit(X_train, Y_train_bin)
    
    # -------
    # Y_predict : size [n_samples, n_classes]
    Y_train_bin_predict = model.predict(X_train)
    Y_test_bin_predict = model.predict(X_test)
    
    Y_train_bin_predict = np.array(Y_train_bin_predict)
    print('shape of Y_train_bin_predict : ', Y_train_bin_predict.shape)
    Y_test_bin_predict = np.array(Y_test_bin_predict)
    print('shape of Y_test_bin_predict : ', Y_test_bin_predict.shape)
    # -------
    
    
    # Can not use the same model with Y_train_bin, as the prediction probability is per class instead of across class
    Y_train_1D = np.reshape(Y_train_1D, (len(Y_train_1D), 1))  # Y needs to have a defined shape ***
    model.fit(X_train, Y_train_1D)
    
    # -------
    # The prediction probability of each class : size is [1, n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    unique_classes = np.unique(Y_train_1D)
    Y_train_bin_pp = np.reshape(Y_train_bin_pp, (len(Y_train_1D), len(unique_classes)))
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.reshape(Y_test_bin_pp, (len(Y_test_1D), len(unique_classes)))
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------
    
    # There is NO decision_function
    # ------------------------------
    Y_train_bin_score = Y_train_bin_pp
    Y_test_bin_score = Y_test_bin_pp
    
    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    
    return model, Y_train_bin, Y_test_bin, Y_train_bin_predict, Y_test_bin_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score

# ----------------------------------------------

# ----------------
# XGBoost
# ----------------
# Ensemble methods : they combine several precise predictive models (each model predicts one thing separately) in order to predict across the total prediction items - trees and random forest is an ensemble method, as each leaf is a separate model.
 
# Gradient Boosting : a method that goes through cycles to iteratively add models into an ensemble - the process is (for each model) : train model, test predictions, calculate loss (mean squared error, accuracy), improve model (train new model - "use the loss function to fit a new model"), add new model to ensemble

# The "gradient" in "gradient boosting" refers to the fact that we'll use gradient descent on the loss function to determine the parameters in this new model.)

# XGBoost stands for extreme gradient boosting, which is an implementation of gradient boosting with several additional features focused on performance and speed
# ----------------------------------------------

# Multi-class XGBoost RandomForest 
# written in an more efficient manner, so it is faster and more accurate than RandomForest.
def multiclass_XGBClassifier(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    from xgboost import XGBClassifier
    
    model = XGBClassifier(num_iterations=1000, eval_metric='mlogloss', boosting='gbdt')
    # can not say num_class=2, gives error
    #num_class=2, learning_rate=0.1,  max_depth=10, feature_fraction=0.7, 
    #scale_pos_weight=1.5, boosting='gbdt', metric='multiclass')
    
    model.fit(X_train, Y_train_1D)

    # -------
    # Y_predict : size [n_samples, 1]
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    # -------
    
    # -------
    # The prediction probability of each class : size is [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------
    
    # There is NO decision_function
    # ------------------------------
    Y_train_bin_score = Y_train_bin_pp
    Y_test_bin_score = Y_test_bin_pp

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    
    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score
    
# ----------------------------------------------

# Multi-class Gradient Boosting Classifier (gradient descent w/ logistic regression cost function)
def multiclass_GradientBoostingClassifier(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0, n_iter_no_change=10)
    
    # loss{‘deviance’, ‘exponential’}, default=’deviance’ : The loss function to be optimized. ‘deviance’ refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss ‘exponential’ gradient boosting recovers the AdaBoost algorithm.

    # max_depth=1, 
    
    model.fit(X_train, Y_train_1D)

    # -------
    # Y_predict : size [n_samples, 1]
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    # -------
    
    # -------
    # The prediction probability of each class : size is [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------
    
    # There is NO decision_function
    # ------------------------------
    Y_train_bin_score = Y_train_bin_pp
    Y_test_bin_score = Y_test_bin_pp

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    
    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score

# ----------------------------------------------

# Multi-class Decision Tree Classifier
def multiclass_Decision_Tree_Classifier(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    
    model.fit(X_train, Y_train_1D)

    # -------
    # Y_predict : size [n_samples, 1]
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    # -------

    # -------
    # The prediction probability of each class : size is [1, n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    unique_classes = np.unique(Y_train_1D)
    Y_train_bin_pp = np.reshape(Y_train_bin_pp, (len(Y_train_1D), len(unique_classes)))
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.reshape(Y_test_bin_pp, (len(Y_test_1D), len(unique_classes)))
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------

    # There is NO decision_function
    # ------------------------------
    Y_train_bin_score = Y_train_bin_pp
    Y_test_bin_score = Y_test_bin_pp
    
    #tree.plot_tree(model)
    
    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score

# ----------------------------------------------

# NearestNeighbors
def Nearest_Neighbors(X_train, Y_train_1D, X_test):
    weights = 'uniform' # 'distance'
    n_neighbors = 15
    model = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    model.fit(X_train, Y_train_1D)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    
    h = 0.02  # step size in the mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])  #decision boundary

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], palette=cmap_bold, alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("n-Class classification: training (k = %i, weights = '%s')" % (n_neighbors, weights))

    plt.show()
    
    return model, Z

# ----------------------------------------------

# Multilayer perceptron (MLP)/neural network (Deep Learning) : logistic regression NN
def multiclass_multilayer_perceptron_bininput(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    # -------
    # Need to binarize Y into size [n_samples, n_classes]
    Y_train_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_train_1D)
    Y_test_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_test_1D)

    Y_train_bin = np.array(Y_train_bin)
    print('shape of Y_train_bin : ', Y_train_bin.shape)

    Y_test_bin = np.array(Y_test_bin)
    print('shape of Y_test_bin : ', Y_test_bin.shape)
    # -------
    
    # You can standarize the data : Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data
    from sklearn.preprocessing import StandardScaler  
    scaler = StandardScaler()  
    # Don't cheat - fit only on training data
    scaler.fit(X_train)  
    X_train = scaler.transform(X_train)  
    # apply same transformation to test data
    X_test = scaler.transform(X_test) 


    from sklearn.neural_network import MLPClassifier

    # If your Y_train has multiple classes (Y_train_1D), it will use softmax to predict the output classes!
    # Else, it will use sigmoid at the last layer.

    # Default: L2 regularization, adam, 
    # If you want a one-layer hL0=1, hL1 =empty (Feedforward perceptron neural network  (FFNN) - (1 layer=shalow learning))
    hL0 = 5
    hL1 = 2
    # Problem : you don't know if it is doing He, or Xavier initialization. Also you do not not how the 
    # scalar random number you give for W is being used to randomly initialize W (a matrix the size of 
    #  the layers, not a scalar)
    w_int = np.random.permutation(hL0)[0]     # Random initialization
    model = MLPClassifier(solver='lbfgs', learning_rate_init=0.0075, hidden_layer_sizes=(hL0, hL1), random_state=w_int, max_iter=100)
    
    model.fit(X_train, Y_train_bin)

    Y_train_bin_predict = model.predict(X_train)
    Y_test_bin_predict = model.predict(X_test)
    #print('Y_test_bin_predict : ', Y_test_bin_predict)

    # The prediction probability of each class : is size [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train) # size is [n_samples, n_classes]
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    #Y_train_bin_pp = np.reshape(Y_train_bin_pp, (len(Y_train_1D_predict), len(unique_classes)))
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    #Y_test_bin_pp = np.reshape(Y_test_bin_pp, (len(Y_test_1D_predict), len(unique_classes)))
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    
    # There is NO decision_function
    # ------------------------------
    Y_train_bin_score = Y_train_bin_pp
    Y_test_bin_score = Y_test_bin_pp
    # OR
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    #Y_train_bin_score = model.decision_function(X_train)  # size is n_samples, 1
    #Y_test_bin_score = model.decision_function(X_test)

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    
    return model, Y_train_bin, Y_test_bin, Y_train_bin_predict, Y_test_bin_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score

# ----------------------------------------------

# Multilayer perceptron (MLP)/neural network (Deep Learning) : logistic regression NN
def multiclass_multilayer_perceptron_1Dinput(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    # You can standarize the data : Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data
    from sklearn.preprocessing import StandardScaler  
    scaler = StandardScaler()  
    # Don't cheat - fit only on training data
    scaler.fit(X_train)  
    X_train = scaler.transform(X_train)  
    # apply same transformation to test data
    X_test = scaler.transform(X_test) 


    from sklearn.neural_network import MLPClassifier

    # If your Y_train has multiple classes (Y_train_1D), it will use softmax to predict the output classes!
    # Else, it will use sigmoid at the last layer.

    # Default: L2 regularization, adam, 
    # If you want a one-layer hL0=1, hL1 =empty (Feedforward perceptron neural network  (FFNN) - (1 layer=shalow learning))
    hL0 = 5
    hL1 = 2
    # Problem : you don't know if it is doing He, or Xavier initialization. Also you do not not how the 
    # scalar random number you give for W is being used to randomly initialize W (a matrix the size of 
    #  the layers, not a scalar)
    w_int = np.random.permutation(hL0)[0]     # Random initialization
    model = MLPClassifier(solver='lbfgs', learning_rate_init=0.0075, hidden_layer_sizes=(hL0, hL1), random_state=w_int, max_iter=100)
    
    model.fit(X_train, Y_train_1D)

    # -------
    # Y_predict : size [n_samples, 1]
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    # -------
    
    # -------
    # The prediction probability of each class : size is [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------
    
    # There is NO decision_function
    # ------------------------------
    Y_train_bin_score = Y_train_bin_pp
    Y_test_bin_score = Y_test_bin_pp

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    
    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score
    
# ----------------------------------------------

# Multi-class Gaussian Naive Bayes
def multiclass_gaussian_naive_bayes(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    from sklearn.naive_bayes import GaussianNB
    
    model = GaussianNB()
    
    model.fit(X_train, Y_train_1D)
    # OR
    # model.partial_fit(X_train, Y_train_bin, np.unique(Y))

    # -------
    # Y_predict : size [n_samples, 1]
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    # -------
    
    # -------
    # The prediction probability of each class : size is [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------
    
    # There is NO decision_function
    # ------------------------------
    Y_train_bin_score = Y_train_bin_pp
    Y_test_bin_score = Y_test_bin_pp

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    
    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score
    
# ----------------------------------------------

 
# ----------------------------------------------
def pipeline_permutation_importance(model, X_test, Y_test, feature_names):
    
    from sklearn.inspection import permutation_importance
    
    # How to do we know which features are important? 
    # 1) test different configuration of X and see which gives the best prediction (what I did before)
    # 2) permutation method (permute the rows of each feature to see if it changes the prediction value)

    # ----------------
    # Permutation importance of features : probe which features are most predictive
    # ----------------
    # The permutation feature importance is the decrease in a model score 
    # when a single feature value is randomly shuffled. 

    # it is the difference between the mean accuracy using all the features and the mean 
    # accuracy of each feature shuffled

    # The difference that is positive and largest, means that the feature is important 
    # because without the feature in proper order the model can not predict well on the 
    # validation data.

    # Example for feature_names :
    # X_test.columns.tolist()
    # feature_names = ['joy', 'joy1derv', 'joy2derv', 'fres', 'freq_t', 'freq_fres']

    r = permutation_importance(model, X_test, Y_test, n_repeats=10, random_state=0, scoring='accuracy')

    ovtot = []
    for i in r.importances_mean.argsort()[::-1]:
        outvals = feature_names[i], r.importances_mean[i], r.importances_std[i]
        ovtot.append(outvals)

    print('ovtot : ', ovtot)

    return ovtot
# ----------------------------------------------


# Regression evaluation : 

# ----------------------------------------------
def R_squared(y, yhat):
    SSres = np.square(np.sum(y - np.mean(y)))
    SStot = np.square(np.sum(y - yhat))
    
    R_squared = 1 - SSres/SStot
    
    return R_squared
# ----------------------------------------------


# Classification evaluation : 

# ----------------------------------------------

def handmade_accuracy(X, Y_1D, Y_1D_predict):
    # 0) Accuracy percentage : classification
    cor = 0
    n_samples = X.shape[0]
    for i in range(0,n_samples):
        if np.sum(Y_1D_predict[i] - Y_1D[i]) == 0:
            cor = cor + 1
    
    Accuracy = (cor/n_samples)*100
    print("Accuracy: " + str(Accuracy) + "%")
    
    return Accuracy

# ----------------------------------------------


# ----------------------------------------------
def evaluation_methods_multi_class_bin(model, X, Y_bin, Y_bin_predict, Y_bin_pp, Y_bin_score):
    
    # 1) cross_val_score with scoring : 
    from sklearn.model_selection import cross_val_score
    cv_num = 5
    acc_crossval = cross_val_score(model, X, Y_bin, cv=cv_num, scoring="accuracy")
    # print('acc_crossval : ', acc_crossval)

    prec_crossval = cross_val_score(model, X, Y_bin, cv=cv_num, scoring="precision")
    # print('prec_crossval : ', prec_crossval)

    recall_crossval = cross_val_score(model, X, Y_bin, cv=cv_num, scoring="recall")
    # print('recall_crossval : ', recall_crossval)

    # Multiclass case :
    rocauc_crossval = cross_val_score(model, X, Y_bin, cv=cv_num, scoring="roc_auc_ovo")
    # print('rocauc_crossval : ', rocauc_crossval)

    roc_auc_ovo_weighted_crossval = cross_val_score(model, X, Y_bin, cv=cv_num, scoring="roc_auc_ovo_weighted")
    # print('roc_auc_ovo_weighted_crossval : ', roc_auc_ovo_weighted_crossval)
    
    # ----------------------------
    
    # Collapse the Y_bin into a Y_1D
    Y_1D = debinarize_Ybin_2_Y1Dvec(Y_bin)
    # print('shape of Y_1D : ', Y_1D.shape)
    
    # Collapse the Y_bin_predict into a Y_1D_predict
    Y_1D_predict = debinarize_Ybin_2_Y1Dvec(Y_bin_predict)
    # print('shape of Y_1D_predict : ', Y_1D_predict.shape)
    
    import numpy as np
    
    # Ensure vector is of integers
    Y_1D = [int(i) for i in Y_1D]
    Y_1D_predict = [int(i) for i in Y_1D_predict]
    
    # Number of samples
    m_examples = len(Y_1D)

    # Number of classes
    temp = np.unique(Y_1D)
    unique_classes = [int(i) for i in temp]
    
    # ----------------------------
    
    # 2) Confusion matrix
    from sklearn.metrics import confusion_matrix
    matrix_of_counts = confusion_matrix(Y_1D, Y_1D_predict)
    # print('matrix_of_counts : ', matrix_of_counts)
    # OR
    matrix_normalized = confusion_matrix(Y_1D, Y_1D_predict, normalize='all')
    # print('matrix_normalized : ', matrix_normalized)

    # Display the confusion matrix
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # disp = ConfusionMatrixDisplay(confusion_matrix=matrix_of_counts, display_labels=clf.classes_)
    # disp.plot()
    # plt.show()
    
    # ----------------------------
    
    # 3) Classification report : builds a text report showing the main classification metrics
    # Not interested

    # ----------------------------

    # 4) Direct calculation of metrics
    from sklearn import metrics

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html

    # labels = array-like, default=None

    # pos_label = str or int, default=1 The class to report if average='binary' and the data is binary. 
    # If the data are multiclass or multilabel, this will be ignored

    # average = ['binary', 'micro', 'macro', 'weighted', 'samples', None]
    # This parameter is required for multiclass/multilabel targets. 
    # None : the scores for each class are returned
    # 'binary'  : Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) are binary.
    # 'micro' : Calculate metrics globally by counting the total true positives, false negatives and false positives.
    # 'macro' : Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    # 'weighted' : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
    # 'samples' : Calculate metrics for each instance, and find their average (only meaningful for multilabel classification where this differs from accuracy_score).
    average = 'micro'

    # sample_weight : array-like of shape (n_samples,), default=None
    acc_dircalc = metrics.accuracy_score(Y_1D, Y_1D_predict)
    # print('acc_dircalc : ', acc_dircalc)

    prec_dircalc = metrics.precision_score(Y_1D, Y_1D_predict, average=average)
    # print('prec_dircalc : ', prec_dircalc)

    recall_dircalc = metrics.recall_score(Y_1D, Y_1D_predict, average=average)
    # print('recall_dircalc : ', recall_dircalc)

    f1_dircalc = metrics.f1_score(Y_1D, Y_1D_predict, average=average)
    # print('f1_dircalc : ', f1_dircalc)

    # beta=0.5, 1, 2
    fbeta_dircalc = metrics.fbeta_score(Y_1D, Y_1D_predict, beta=0.5, average=average)
    # print('fbeta_dircalc : ', fbeta_dircalc)

    # y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
    # y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
    rocauc_pp_dircalc = metrics.roc_auc_score(Y_bin, Y_bin_pp, average=average) # prediction probability
    # print('rocauc_pp_dircalc : ', rocauc_pp_dircalc)
    # OR
    rocauc_df_dircalc = metrics.roc_auc_score(Y_bin, Y_bin_score, average=average) # decision function 
    # print('rocauc_df_dircalc : ', rocauc_df_dircalc)

    # ----------------------------
    
    # # 5) Direct calculation of metrics : micro-average ROC curve and ROC area
    # # True Positive Rate (TPR) = TP / (TP + FN) = efficiency (εₛ) to identify the signal (also known as Recall or Sensitivity)
    
    # # False Positive Rate (FPR) = FP / (FP + TN) = inefficiency (ε_B) to reject background
    
    # #https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    # from sklearn.metrics import roc_curve, auc
    # # Compute ROC curve and ROC area for each class
    # # fpr = dict()
    # # tpr = dict()
    # # roc_auc = dict()
    
    # # # [1] Compute ROC curve and ROC area for each class
    # # for i in range(len(unique_classes)):
        # # fpr[i], tpr[i], _ = roc_curve(Y_bin[:, i], Y_bin_score[:, i]) # decision function
        # # # OR
        # # fpr[i], tpr[i], _ = roc_curve(Y_bin[:, i], Y_bin_pp[:, i]) # prediction probability
        
        # # roc_auc[i] = auc(fpr[i], tpr[i])
        
    # # [0] Micro-average ROC curve and ROC area : all the classes together!
    # fpr0, tpr0, thresh = roc_curve(Y_bin.ravel(), Y_bin_score.ravel()) # decision function
    # # OR
    # fpr0, tpr0, thresh = roc_curve(Y_bin.ravel(), Y_bin_pp.ravel()) # prediction probability
    
    # roc_auc0 = auc(fpr0, tpr0)

    # # Plot of a ROC curve for a specific class
    # import matplotlib.pyplot as plt
    # plt.figure()
    # lw = 2
    # plt.plot(fpr0, tpr0, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc0)
    # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver operating characteristic example")
    # plt.legend(loc="lower right")
    # plt.show()
    
    value_pack = {}
    
    var_list = ['acc_crossval', 'prec_crossval', 'recall_crossval', 'rocauc_ovo_crossval', 'roc_auc_ovo_weighted_crossval', 'acc_dircalc', 'prec_dircalc', 'recall_dircalc', 'f1_dircalc', 'fbeta_dircalc', 'rocauc_pp_dircalc', 'rocauc_df_dircalc']
    var_list_num = [acc_crossval, prec_crossval, recall_crossval, rocauc_ovo_crossval, roc_auc_ovo_weighted_crossval, acc_dircalc, prec_dircalc, recall_dircalc, f1_dircalc, fbeta_dircalc, rocauc_pp_dircalc, rocauc_df_dircalc]
    
    for q in range(len(var_list)):
        value_pack['%s' % (var_list[q])] = var_list_num[q]
    
    return value_pack
    
# ----------------------------------------------
def evaluation_methods_multi_class_1D(model, X, Y_1D, Y_1D_predict, Y_bin_pp, Y_bin_score):
    
    import numpy as np
    
    # -------
    # Need to binarize Y into size [n_samples, n_classes]
    Y_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_1D)
    Y_bin_predict, unique_classes = binarize_Y1Dvec_2_Ybin(Y_1D_predict)

    Y_bin = np.array(Y_bin)
    print('shape of Y_bin : ', Y_bin.shape)

    Y_bin_predict = np.array(Y_bin_predict)
    print('shape of Y_bin_predict : ', Y_bin_predict.shape)
    # -------
    
    # 1) cross_val_score with scoring : 
    from sklearn.model_selection import cross_val_score
    cv_num = 5
    acc_crossval = cross_val_score(model, X, Y_1D, cv=cv_num, scoring="accuracy")
    # print('acc_crossval : ', acc_crossval)
    
    prec_crossval = cross_val_score(model, X, Y_1D, cv=cv_num, scoring="precision")
    # print('prec_crossval : ', prec_crossval)

    recall_crossval = cross_val_score(model, X, Y_1D, cv=cv_num, scoring="recall")
    # print('recall_crossval : ', recall_crossval)
    
    # Multiclass case :  ** check if Y_bin works
    rocauc_crossval = cross_val_score(model, X, Y_1D, cv=cv_num, scoring="roc_auc_ovo")
    # print('rocauc_crossval : ', rocauc_crossval)

    roc_auc_ovo_weighted_crossval = cross_val_score(model, X, Y_1D, cv=cv_num, scoring="roc_auc_ovo_weighted")
    # print('roc_auc_ovo_weighted_crossval : ', roc_auc_ovo_weighted_crossval)
    
    # ----------------------------

    # Number of classes
    temp = np.unique(Y_1D)
    unique_classes = [int(i) for i in temp]
    
    # ----------------------------
    
    # 2) Confusion matrix
    from sklearn.metrics import confusion_matrix
    matrix_of_counts = confusion_matrix(Y_1D, Y_1D_predict)
    # print('matrix_of_counts : ', matrix_of_counts)
    # OR
    matrix_normalized = confusion_matrix(Y_1D, Y_1D_predict, normalize='all')
    # print('matrix_normalized : ', matrix_normalized)

    # Display the confusion matrix
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # disp = ConfusionMatrixDisplay(confusion_matrix=matrix_of_counts, display_labels=clf.classes_)
    # disp.plot()
    # plt.show()
    
    # ----------------------------
    
    # 3) Classification report : builds a text report showing the main classification metrics
    # Not interested

    # ----------------------------

    # 4) Direct calculation of metrics
    from sklearn import metrics

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html

    # labels = array-like, default=None

    # pos_label = str or int, default=1 The class to report if average='binary' and the data is binary. 
    # If the data are multiclass or multilabel, this will be ignored

    # average = ['binary', 'micro', 'macro', 'weighted', 'samples', None]
    # This parameter is required for multiclass/multilabel targets. 
    # None : the scores for each class are returned
    # 'binary'  : Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) are binary.
    # 'micro' : Calculate metrics globally by counting the total true positives, false negatives and false positives.
    # 'macro' : Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    # 'weighted' : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
    # 'samples' : Calculate metrics for each instance, and find their average (only meaningful for multilabel classification where this differs from accuracy_score).
    average = 'micro'

    # sample_weight : array-like of shape (n_samples,), default=None
    acc_dircalc = metrics.accuracy_score(Y_1D, Y_1D_predict)
    # print('acc_dircalc : ', acc_dircalc)

    prec_dircalc = metrics.precision_score(Y_1D, Y_1D_predict, average=average)
    # print('prec_dircalc : ', prec_dircalc)

    recall_dircalc = metrics.recall_score(Y_1D, Y_1D_predict, average=average)
    # print('recall_dircalc : ', recall_dircalc)

    f1_dircalc = metrics.f1_score(Y_1D, Y_1D_predict, average=average)
    # print('f1_dircalc : ', f1_dircalc)

    # beta=0.5, 1, 2
    fbeta_dircalc = metrics.fbeta_score(Y_1D, Y_1D_predict, beta=0.5, average=average)
    # print('fbeta_dircalc : ', fbeta_dircalc)
    
    prec_recall_f_dircalc = metrics.precision_recall_fscore_support(Y_1D, Y_1D_predict, beta=0.5, average=average)
    # print('prec_recall_f_dircalc : ', prec_recall_f_dircalc)
    
    # y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
    # y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
    # rocauc_pp_dircalc = metrics.roc_auc_score(Y_bin, Y_bin_pp, average=average) # prediction probability
    # print('rocauc_pp_dircalc : ', rocauc_pp_dircalc)
    # OR
    rocauc_df_dircalc = metrics.roc_auc_score(Y_bin, Y_bin_score, average=average) # decision function 
    # print('rocauc_df_dircalc : ', rocauc_df_dircalc)
    
    rocauc_pp_dircalc = rocauc_df_dircalc

    # ----------------------------
    
    # # 5) Direct calculation of metrics : micro-average ROC curve and ROC area
    # # True Positive Rate (TPR) = TP / (TP + FN) = efficiency (εₛ) to identify the signal (also known as Recall or Sensitivity)
    
    # # False Positive Rate (FPR) = FP / (FP + TN) = inefficiency (ε_B) to reject background
    
    # #https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    # from sklearn.metrics import roc_curve, auc
    # # Compute ROC curve and ROC area for each class
    # # fpr = dict()
    # # tpr = dict()
    # # roc_auc = dict()
    
    # # # [1] Compute ROC curve and ROC area for each class
    # # for i in range(len(unique_classes)):
        # # fpr[i], tpr[i], _ = roc_curve(Y_bin[:, i], Y_bin_score[:, i]) # decision function
        # # # OR
        # # fpr[i], tpr[i], _ = roc_curve(Y_bin[:, i], Y_bin_pp[:, i]) # prediction probability
        
        # # roc_auc[i] = auc(fpr[i], tpr[i])
        
    # # [0] Micro-average ROC curve and ROC area : all the classes together!
    # fpr0, tpr0, thresh = roc_curve(Y_bin.ravel(), Y_bin_score.ravel()) # decision function
    # # OR
    # fpr0, tpr0, thresh = roc_curve(Y_bin.ravel(), Y_bin_pp.ravel()) # prediction probability
    
    # roc_auc0 = auc(fpr0, tpr0)

    # # Plot of a ROC curve for a specific class
    # import matplotlib.pyplot as plt
    # plt.figure()
    # lw = 2
    # plt.plot(fpr0, tpr0, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc0)
    # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver operating characteristic example")
    # plt.legend(loc="lower right")
    # plt.show()
    
    
    value_pack = {}
    
    var_list = ['acc_crossval', 'prec_crossval', 'recall_crossval', 'rocauc_crossval', 'roc_auc_ovo_weighted_crossval', 'acc_dircalc', 'prec_dircalc', 'recall_dircalc', 'f1_dircalc', 'fbeta_dircalc', 'prec_recall_f_dircalc', 'rocauc_pp_dircalc', 'rocauc_df_dircalc']
    var_list_num = [acc_crossval, prec_crossval, recall_crossval, rocauc_crossval, roc_auc_ovo_weighted_crossval, acc_dircalc, prec_dircalc, recall_dircalc, f1_dircalc, fbeta_dircalc, prec_recall_f_dircalc, rocauc_pp_dircalc, rocauc_df_dircalc]
    
    for q in range(len(var_list)):
        value_pack['%s' % (var_list[q])] = var_list_num[q]
    
    return value_pack