# Created by Jamilah Foucher, Novembre 19, 2021 

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



# Machine Learning models :

# ----------------------------------------------
# 2-class Stochastic Gradient Descent
def binary_stochastic_gradient_descent(X_train, X_test, Y_train_1D, Y_test_1D):
    # X_train is size (n_samples, n_features)
    # X_test is size (n_samples, n_features)
    # Y_train_1D is size (n_samples, 1) where the n_classes are represented by 1,2,3,..., etc
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    from sklearn.calibration import CalibratedClassifierCV
    
    # Can ONLY calculate the probability estimates (predict_proba), but NOT the decision_function
    lr = SGDClassifier(loss='hinge', alpha=0.001, class_weight='balanced')
    clf =lr.fit(X_train, Y_train_1D)
    calibrator = CalibratedClassifierCV(clf, cv='prefit')
    model = calibrator.fit(X_train, Y_train_1D)
    
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    
    # The prediction probability of each class : is size [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train) 
    Y_test_bin_pp = model.predict_proba(X_test)
    
    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    
    
    # probability estimates (predict_proba) are not available for loss='hinge', but can calculate the decision_function
    model = SGDClassifier(alpha=0.001, loss='hinge') 
    model.fit(X_train, Y_train_1D)
    
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    Y_train_1D_score = model.decision_function(X_train)  # size is [n_samples, 1]
    Y_test_1D_score = model.decision_function(X_test)
    
    Y_train_1D_score = np.array(Y_train_1D_score)
    print('shape of Y_train_1D_score : ', Y_train_1D_score.shape)
    Y_test_1D_score = np.array(Y_test_1D_score)
    print('shape of Y_test_1D_score : ', Y_test_1D_score.shape)

    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_1D_score, Y_test_1D_score

# ----------------------------------------------

# 2-class Linear Discriminant Analysis (LDA)
def binary_LDA_classifier(X_train, X_test, Y_train_1D, Y_test_1D):
    import numpy as np
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    model = LinearDiscriminantAnalysis()

    model.fit(X_train, Y_train_1D)

    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)

    # The prediction probability of each class : is size [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train) 
    Y_test_bin_pp = model.predict_proba(X_test)
    
    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    Y_train_1D_score = model.decision_function(X_train)  # size is [n_samples, 1]
    Y_test_1D_score = model.decision_function(X_test)
    
    Y_train_1D_score = np.array(Y_train_1D_score)
    print('shape of Y_train_1D_score : ', Y_train_1D_score.shape)
    Y_test_1D_score = np.array(Y_test_1D_score)
    print('shape of Y_test_1D_score : ', Y_test_1D_score.shape)

    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_1D_score, Y_test_1D_score
    
# ----------------------------------------------

# 2-class Support Vector Machine (SVC) using the pipeline function
def binary_svm_pipeline(X_train, X_test, Y_train_1D, Y_test_1D):
    # X_train is size (n_samples, n_features)
    # X_test is size (n_samples, n_features)
    # Y_train_1D is size (n_samples, 1) where the n_classes are represented by 1,2,3,..., etc
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline

    model = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    model.fit(X_train, Y_train_1D)
    Pipeline(steps=[('standardscaler', StandardScaler()), ('svc', SVC(gamma='auto', probability=True))])

    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)

    # The prediction probability of each class : is size [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train) 
    Y_test_bin_pp = model.predict_proba(X_test)
    
    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    Y_train_1D_score = model.decision_function(X_train)  # size is [n_samples, 1]
    Y_test_1D_score = model.decision_function(X_test)
    
    Y_train_1D_score = np.array(Y_train_1D_score)
    print('shape of Y_train_1D_score : ', Y_train_1D_score.shape)
    Y_test_1D_score = np.array(Y_test_1D_score)
    print('shape of Y_test_1D_score : ', Y_test_1D_score.shape)

    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_1D_score, Y_test_1D_score
    
# ----------------------------------------------

# 2-class Support Vector Machine (SVC)
def binary_svm(X_train, X_test, Y_train_1D, Y_test_1D):

    # X_train is size [n_samples, n_features]
    # Y_train_1D is size [n_samples, 1]  where each class is a unique value

    from sklearn import svm
    import numpy as np
    
    model = svm.SVC(decision_function_shape='ovo', probability=True, max_iter=100)  # “one-versus-one” : binary ONLY, Y_train_1Darra, uses C in cost function
    
    # Available options :
    # C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False), cv=3, method='sigmoid'

    model.fit(X_train, Y_train_1D)

    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)

    # The prediction probability of each class : is size [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train) 
    Y_test_bin_pp = model.predict_proba(X_test)
    
    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    Y_train_1D_score = model.decision_function(X_train)  # size is [n_samples, 1]
    Y_test_1D_score = model.decision_function(X_test)
    
    Y_train_1D_score = np.array(Y_train_1D_score)
    print('shape of Y_train_1D_score : ', Y_train_1D_score.shape)
    Y_test_1D_score = np.array(Y_test_1D_score)
    print('shape of Y_test_1D_score : ', Y_test_1D_score.shape)

    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_1D_score, Y_test_1D_score
    
# ----------------------------------------------

# 2-class Support Vector Machine (NuSVC)
def binary_svm_NuSVC(X_train, X_test, Y_train_1D, Y_test_1D):

    # X_train is size [n_samples, n_features]
    # Y_train_1D is size [n_samples, 1]  where each class is a unique value

    from sklearn import svm
    import numpy as np
    
    model = svm.NuSVC(decision_function_shape='ovo', probability=True, max_iter=1)  # “one-versus-one” : binary ONLY, Y_train_1D, same implementation as libsvm (uses 1/lambda instead of C in cost function)
    
    # Available options :
    # C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf', max_iter=-1 (runs until convergence), probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False), cv=3, method='sigmoid'

    model.fit(X_train, Y_train_1D)

    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)

    # The prediction probability of each class : is size [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train) 
    Y_test_bin_pp = model.predict_proba(X_test)
    
    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    Y_train_1D_score = model.decision_function(X_train)  # size is [n_samples, 1]
    Y_test_1D_score = model.decision_function(X_test)
    
    Y_train_1D_score = np.array(Y_train_1D_score)
    print('shape of Y_train_1D_score : ', Y_train_1D_score.shape)
    Y_test_1D_score = np.array(Y_test_1D_score)
    print('shape of Y_test_1D_score : ', Y_test_1D_score.shape)

    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_1D_score, Y_test_1D_score
    
# ----------------------------------------------

# 2-class Support Vector Machine (SVC) Bagging
# Slower than normal svm, but accuracy is better
def binary_svm_bagging(X_train, X_test, Y_train_1D, Y_test_1D):

    # X_train is size [n_samples, n_features]
    # Y_train_1D is size [n_samples, 1]  where each class is a unique value
    
    import numpy as np
    from sklearn import svm
    from sklearn.ensemble import BaggingClassifier
    from sklearn.multiclass import OneVsOneClassifier
    
    n_estimators = 10
    model = OneVsOneClassifier(BaggingClassifier(svm.SVC(kernel='linear', probability=True, class_weight=None, max_iter=100), n_estimators=n_estimators))
    
    # Available options :
    # C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False), cv=3, method='sigmoid'
    
    # OR
    # model = svm.NuSVC(decision_function_shape='ovo', probability=True)  # “one-versus-one” : binary ONLY, Y_train_1D, same implementation as libsvm (uses 1/lambda instead of C in cost function)

    model.fit(X_train, Y_train_1D)

    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)

    # The prediction probability of each class : is size [n_samples, n_classes]
    #Y_train_bin_pp = model.predict_proba(X_train) 
    #Y_test_bin_pp = model.predict_proba(X_test)
    
    Y_train_bin_pp = Y_train_1D_predict, Y_train_1D_predict
    Y_test_bin_pp = Y_test_1D_predict, Y_test_1D_predict
    
    Y_train_bin_pp = np.array(Y_train_bin_pp)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    
    Y_train_bin_pp = np.reshape(Y_train_bin_pp, (len(Y_train_1D_predict),2))
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    
    Y_test_bin_pp = np.reshape(Y_test_bin_pp, (len(Y_test_1D_predict),2))
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    Y_train_1D_score = model.decision_function(X_train)  # size is [n_samples, 1]
    Y_test_1D_score = model.decision_function(X_test)
    
    Y_train_1D_score = np.array(Y_train_1D_score)
    print('shape of Y_train_1D_score : ', Y_train_1D_score.shape)
    Y_test_1D_score = np.array(Y_test_1D_score)
    print('shape of Y_test_1D_score : ', Y_test_1D_score.shape)

    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_1D_score, Y_test_1D_score

# ----------------------------------------------

# 2-class RandomForest
def binary_RandomForest(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(random_state=1, min_samples_leaf=50)  # min_samples_leaf is 100 by default
    model = MultiOutputClassifier(forest, n_jobs=-1) #n_jobs=-1 means apply parallel processing
    
    Y_train_1D = np.reshape(Y_train_1D, (len(Y_train_1D), 1))  # Y needs to have a defined shape ***
    model.fit(X_train, Y_train_1D)

    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)

    # The prediction probability of each class : is size [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train) 
    Y_test_bin_pp = model.predict_proba(X_test)
    
    Y_train_bin_pp = np.reshape(Y_train_bin_pp, (len(Y_train_1D_predict), 2))
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.reshape(Y_test_bin_pp, (len(Y_test_1D_predict), 2))
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    
    # There is NO decision_function
    # ------------------------------
    Y_train_1D_score = transform_Y_bin_pp_2_Y_1D_pp(Y_train_1D, Y_train_bin_pp)
    Y_test_1D_score = transform_Y_bin_pp_2_Y_1D_pp(Y_test_1D, Y_test_bin_pp)
    # OR
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    #Y_train_1D_score = model.decision_function(X_train)  # size is [n_samples, 1]
    #Y_test_1D_score = model.decision_function(X_test)
    
    Y_train_1D_score = np.array(Y_train_1D_score)
    print('shape of Y_train_1D_score : ', Y_train_1D_score.shape)
    Y_test_1D_score = np.array(Y_test_1D_score)
    print('shape of Y_test_1D_score : ', Y_test_1D_score.shape)

    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_1D_score, Y_test_1D_score
# ----------------------------------------------


# ----------------------------------------------
# 2-class XGBoost RandomForest 
# written in an more efficient manner, so it is faster and more accurate than RandomForest.

def binary_XGBClassifier(X_train, X_test, Y_train_1D, Y_test_1D):

    import numpy as np
    from xgboost import XGBClassifier
    
    model = XGBClassifier(num_iterations=1000, eval_metric='mlogloss', boosting='gbdt')
    # can not say num_class=2, gives error
    #num_class=2, learning_rate=0.1,  max_depth=10, feature_fraction=0.7, 
    #scale_pos_weight=1.5, boosting='gbdt', metric='multiclass')
    
    Y_train_1D = np.reshape(Y_train_1D, (len(Y_train_1D), 1))
    model.fit(X_train, Y_train_1D)
    
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)

    # The prediction probability of each class : is size [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train) 
    Y_test_bin_pp = model.predict_proba(X_test)
    
    Y_train_bin_pp = np.reshape(Y_train_bin_pp, (len(Y_train_1D_predict), 2))
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.reshape(Y_test_bin_pp, (len(Y_test_1D_predict), 2))
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    
    # There is NO decision_function
    # ------------------------------
    Y_train_1D_score = transform_Y_bin_pp_2_Y_1D_pp(Y_train_1D, Y_train_bin_pp)
    Y_test_1D_score = transform_Y_bin_pp_2_Y_1D_pp(Y_test_1D, Y_test_bin_pp)
    # OR
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    #Y_train_1D_score = model.decision_function(X_train)  # size is [n_samples, 1]
    #Y_test_1D_score = model.decision_function(X_test)
    
    Y_train_1D_score = np.array(Y_train_1D_score)
    print('shape of Y_train_1D_score : ', Y_train_1D_score.shape)
    Y_test_1D_score = np.array(Y_test_1D_score)
    print('shape of Y_test_1D_score : ', Y_test_1D_score.shape)

    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_1D_score, Y_test_1D_score
# ----------------------------------------------

# 2-class Gradient Boosting Classifier (gradient descent w/ logistic regression cost function)
def binary_GradientBoostingClassifier(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0, n_iter_no_change=10)
    
    # loss{‘deviance’, ‘exponential’}, default=’deviance’ : The loss function to be optimized. ‘deviance’ refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss ‘exponential’ gradient boosting recovers the AdaBoost algorithm.

    # max_depth=1, 
    
    model = model.fit(X_train, Y_train_1D)
    
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)

    # The prediction probability of each class : is size [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train) 
    Y_test_bin_pp = model.predict_proba(X_test)
    
    Y_train_bin_pp = np.reshape(Y_train_bin_pp, (len(Y_train_1D_predict), 2))
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.reshape(Y_test_bin_pp, (len(Y_test_1D_predict), 2))
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    
    # There is NO decision_function
    # ------------------------------
    Y_train_1D_score = transform_Y_bin_pp_2_Y_1D_pp(Y_train_1D, Y_train_bin_pp)
    Y_test_1D_score = transform_Y_bin_pp_2_Y_1D_pp(Y_test_1D, Y_test_bin_pp)
    # OR
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    #Y_train_1D_score = model.decision_function(X_train)  # size is [n_samples, 1]
    #Y_test_1D_score = model.decision_function(X_test)
    
    Y_train_1D_score = np.array(Y_train_1D_score)
    print('shape of Y_train_1D_score : ', Y_train_1D_score.shape)
    Y_test_1D_score = np.array(Y_test_1D_score)
    print('shape of Y_test_1D_score : ', Y_test_1D_score.shape)
    
    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_1D_score, Y_test_1D_score

# ----------------------------------------------

# 2-class Decision Tree Classifier
def binary_Decision_Tree_Classifier(X_train, X_test, Y_train_1D, Y_test_1D):

    import numpy as np
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model = model.fit(X_train, Y_train_1D)
    
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)

    # The prediction probability of each class : is size [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train) 
    Y_test_bin_pp = model.predict_proba(X_test)
    
    Y_train_bin_pp = np.reshape(Y_train_bin_pp, (len(Y_train_1D_predict), 2))
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.reshape(Y_test_bin_pp, (len(Y_test_1D_predict), 2))
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    
    # There is NO decision_function
    # ------------------------------
    Y_train_1D_score = transform_Y_bin_pp_2_Y_1D_pp(Y_train_1D, Y_train_bin_pp)
    Y_test_1D_score = transform_Y_bin_pp_2_Y_1D_pp(Y_test_1D, Y_test_bin_pp)
    # OR
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    #Y_train_1D_score = model.decision_function(X_train)  # size is [n_samples, 1]
    #Y_test_1D_score = model.decision_function(X_test)
    
    Y_train_1D_score = np.array(Y_train_1D_score)
    print('shape of Y_train_1D_score : ', Y_train_1D_score.shape)
    Y_test_1D_score = np.array(Y_test_1D_score)
    print('shape of Y_test_1D_score : ', Y_test_1D_score.shape)
    
    #tree.plot_tree(model)
    
    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_1D_score, Y_test_1D_score

# ----------------------------------------------

# 2-class Multilayer perceptron (MLP)/neural network (Deep Learning) : logistic regression NN
def binary_multilayer_perceptron(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
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

    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)

    # The prediction probability of each class : is size [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train) 
    Y_test_bin_pp = model.predict_proba(X_test)
    
    Y_train_bin_pp = np.reshape(Y_train_bin_pp, (len(Y_train_1D_predict), 2))
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.reshape(Y_test_bin_pp, (len(Y_test_1D_predict), 2))
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    
    # There is NO decision_function
    # ------------------------------
    Y_train_1D_score = transform_Y_bin_pp_2_Y_1D_pp(Y_train_1D, Y_train_bin_pp)
    Y_test_1D_score = transform_Y_bin_pp_2_Y_1D_pp(Y_test_1D, Y_test_bin_pp)
    # OR
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    #Y_train_1D_score = model.decision_function(X_train)  # size is [n_samples, 1]
    #Y_test_1D_score = model.decision_function(X_test)
    
    Y_train_1D_score = np.array(Y_train_1D_score)
    print('shape of Y_train_1D_score : ', Y_train_1D_score.shape)
    Y_test_1D_score = np.array(Y_test_1D_score)
    print('shape of Y_test_1D_score : ', Y_test_1D_score.shape)

    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_1D_score, Y_test_1D_score
    
# ----------------------------------------------

# 2-class Gaussian Naive Bayes
def binary_gaussian_naive_bayes(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    from sklearn.naive_bayes import GaussianNB
    
    model = GaussianNB()
    
    # priors : array-like of shape (n_classes,) : Prior probabilities of the classes. If specified the priors are not adjusted according to the data.
    
    # var_smoothing : float, default=1e-9 : Portion of the largest variance of all features that is added to variances for calculation stability.
    
    model.fit(X_train, Y_train_1D)
    # OR
    # model.partial_fit(X_train, Y_train_1D, np.unique(Y))

    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)

    # The prediction probability of each class : is size [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train) 
    Y_test_bin_pp = model.predict_proba(X_test)
    
    Y_train_bin_pp = np.reshape(Y_train_bin_pp, (len(Y_train_1D_predict), 2))
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.reshape(Y_test_bin_pp, (len(Y_test_1D_predict), 2))
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    
    # There is NO decision_function
    # ------------------------------
    Y_train_1D_score = transform_Y_bin_pp_2_Y_1D_pp(Y_train_1D, Y_train_bin_pp)
    Y_test_1D_score = transform_Y_bin_pp_2_Y_1D_pp(Y_test_1D, Y_test_bin_pp)
    # OR
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    #Y_train_1D_score = model.decision_function(X_train)  # size is [n_samples, 1]
    #Y_test_1D_score = model.decision_function(X_test)
    
    Y_train_1D_score = np.array(Y_train_1D_score)
    print('shape of Y_train_1D_score : ', Y_train_1D_score.shape)
    Y_test_1D_score = np.array(Y_test_1D_score)
    print('shape of Y_test_1D_score : ', Y_test_1D_score.shape)

    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_1D_score, Y_test_1D_score
# ----------------------------------------------


# ----------------------------------------------
def permutation_importance_scikit(model, X_test, Y_test):
    
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


# ----------------------------------------------
def evaluation_methods_binary_class(model, X, Y_1D, Y_1D_predict, Y_bin_pp, Y_1D_score):
    
    # 1) cross_val_score with scoring : 
    from sklearn.model_selection import cross_val_score
    cv_num = 5
    acc_crossval = cross_val_score(model, X, Y_1D, cv=cv_num, scoring="accuracy")
    print('acc_crossval : ', acc_crossval)

    prec_crossval = cross_val_score(model, X, Y_1D, cv=cv_num, scoring="precision")
    print('prec_crossval : ', prec_crossval)

    recall_crossval = cross_val_score(model, X, Y_1D, cv=cv_num, scoring="recall")
    print('recall_crossval : ', recall_crossval)

    # Binary class case :
    rocauc_crossval = cross_val_score(model, X, Y_1D, cv=cv_num, scoring="roc_auc")
    print('rocauc_crossval : ', rocauc_crossval)
    
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

    # For binary case ONLY:
    tn, fp, fn, tp = confusion_matrix(Y_1D, Y_1D_predict, normalize='all').ravel()
    
    # ----------------------------
    
    # 3) Classification report : builds a text report showing the main classification metrics
    # Not interested
    # from sklearn.metrics import classification_report
    
    # target_names = ['class 0', 'class 1', 'class 2']
    # print(classification_report(Y_1D, Y_1D_predict, target_names=target_names))

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
    print('acc_dircalc : ', acc_dircalc)

    prec_dircalc = metrics.precision_score(Y_1D, Y_1D_predict, average=average)
    print('prec_dircalc : ', prec_dircalc)

    recall_dircalc = metrics.recall_score(Y_1D, Y_1D_predict, average=average)
    print('recall_dircalc : ', recall_dircalc)

    f1_dircalc = metrics.f1_score(Y_1D, Y_1D_predict, average=average)
    print('f1_dircalc : ', f1_dircalc)
    
    # beta=0.5, 1, 2
    fbeta_dircalc = metrics.fbeta_score(Y_1D, Y_1D_predict, beta=0.5, average=average)
    print('fbeta_dircalc : ', fbeta_dircalc)
    
    prec_recall_f_dircalc = metrics.precision_recall_fscore_support(Y_1D, Y_1D_predict, beta=0.5, average=average)
    print('prec_recall_f_dircalc : ', prec_recall_f_dircalc)

    # y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
    # y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
    
    # Y_bin_pp is size [n_samples, n_classes=2]
    # Take the column of Y_bin_pp for the class of Y_1D, because both vectors need to be [n_samples, 1]
    Y_1D_pp = transform_Y_bin_pp_2_Y_1D_pp(Y_1D, Y_bin_pp)
    import numpy as np
    Y_1D = np.array(Y_1D)
    print('shape of Y_1D : ', Y_1D.shape)
    
    print('shape of Y_1D_pp : ', Y_1D_pp.shape)
    
    # prediction probability  (***does not seem to be correct***)
    rocauc_pp_dircalc = metrics.roc_auc_score(Y_1D, Y_1D_pp, average=average) # prediction probability
    print('rocauc_pp_dircalc : ', rocauc_pp_dircalc)
    
    # Y_1D_score is size [n_samples, 1]
    rocauc_df_dircalc = metrics.roc_auc_score(Y_1D, Y_1D_score, average=average) # decision function
    print('rocauc_df_dircalc : ', rocauc_df_dircalc)

    # ----------------------------
    
    # # 5) Direct calculation of metrics : micro-average ROC curve and ROC area
    # #https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    # from sklearn.metrics import roc_curve, auc
    
    # # Compute ROC curve and ROC area for each class
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    
    # # [0] Micro-average ROC curve and ROC area : all the classes together!
    # fpr0, tpr0, thresh = roc_curve(Y_1D, Y_1D_pp) # prediction probability  (*does not seem to be correct)
    # # OR
    # fpr0, tpr0, thresh = roc_curve(Y_1D, Y_1D_score) # decision function
    # roc_auc0 = auc(fpr0, tpr0)
    
    # # Assign to dictionary
    # fpr["micro_avg"] = fpr0
    # tpr["micro_avg"] = tpr0
    # roc_auc["micro_avg"] = roc_auc0
    
    # # [1] Compute ROC curve and ROC area for each class
    # # Micro : class 0 and 1
    # Y_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_1D)
    # #print('unique_classes : ', unique_classes)
    # fpr1 = []
    # tpr1 = []
    # roc_auc1 = []
    # for i in range(len(unique_classes)):
        # fpr1_temp, tpr1_temp, thresh = roc_curve(Y_bin[:, i], Y_bin_pp[:, i])  # prediction probability
        # roc_auc1_temp = auc(fpr1_temp, tpr1_temp)
        # fpr1.append(fpr1_temp)
        # tpr1.append(tpr1_temp)
        # roc_auc1.append(roc_auc1_temp)
    # print('roc_auc1 : ', roc_auc1)
    
    # # Assign to dictionary
    # fpr["micro"] = fpr1
    # tpr["micro"] = tpr1
    # roc_auc["micro"] = roc_auc1
    
    # # For a binary case the Micro-average ROC is the same as each Micro class 0 and class 1
    
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
    
    var_list = ['acc_crossval', 'prec_crossval', 'recall_crossval', 'rocauc_crossval', 'tn', 'fp', 'fn', 'tp', 'acc_dircalc', 'prec_dircalc', 'recall_dircalc', 'f1_dircalc', 'fbeta_dircalc', 'prec_recall_f_dircalc', 'rocauc_pp_dircalc', 'rocauc_df_dircalc']
    var_list_num = [acc_crossval, prec_crossval, recall_crossval, rocauc_crossval, tn, fp, fn, tp, acc_dircalc, prec_dircalc, recall_dircalc, f1_dircalc, fbeta_dircalc, prec_recall_f_dircalc, rocauc_pp_dircalc, rocauc_df_dircalc]
    
    for q in range(len(var_list)):
        value_pack['%s' % (var_list[q])] = var_list_num[q]
    
    return value_pack
    
# ----------------------------------------------
