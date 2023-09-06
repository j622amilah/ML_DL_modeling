import numpy as np
import pandas as pd
from sklearn import cluster

def unsupervised_lab_kmeans_clustering(*arg):
    
    n_clusters = arg[0]
    X = arg[1]
    
    X = np.array(X)
    
    if len(X.shape) == 1 or X.shape[1] == 1:
        X = np.ravel(X)
        out = pd.Series(X)
        X = pd.concat([out, out], axis=1).to_numpy()
    
    kmeans = cluster.KMeans(n_clusters=n_clusters, init='k-means++',algorithm='elkan', random_state=2)
    # n_clusters : The number of clusters to form as well as the number of centroids to generate. (int, default=8)
    
    # init : Method for initialization : (default=’k-means++’)
    # init='k-means++' : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. 
    # init='random': choose n_clusters observations (rows) at random from data for the initial centroids.
    
    # n_init : Number of time the k-means algorithm will be run with different centroid seeds (int, default=10)
    
    # max_iter : Maximum number of iterations of the k-means algorithm for a single run. (int, default=300)
    
    # tol : Relative tolerance with regards to Frobenius norm of the difference in the cluster centers 
    # of two consecutive iterations to declare convergence. (float, default=1e-4)
    
    # (extremly important!) random_state : Determines random number generation for centroid initialization
    #(int, RandomState instance or None, default=None)
    
    # algorithm{“auto”, “full”, “elkan”}, default=”auto”
    # K-means algorithm to use. The classical EM-style algorithm is “full”. The “elkan” variation is more 
    # efficient on data with well-defined clusters, by using the triangle inequality. However it’s more 
    # memory intensive due to the allocation of an extra array of shape (n_samples, n_clusters).
    
    # ------------------------------
    
    # print('shape of X : ', X.shape)
    kmeans.fit(X)

    # ------------------------------

    # Get the prediction of each category : predicted label
    label = kmeans.labels_
    # print('clusters_out : ' + str(clusters_out))
    # OR
    label = kmeans.predict(X)
    # print('clusters_out : ' + str(clusters_out))
    # print('length of clusters_out', len(clusters_out))
    
    # ------------------------------
    
    # Centroid values for feature space : this is the center cluster value per feature in X
    centroids = kmeans.cluster_centers_
    # print('centroids org : ' + str(centroids))

    # ------------------------------

    return kmeans, label, centroids
