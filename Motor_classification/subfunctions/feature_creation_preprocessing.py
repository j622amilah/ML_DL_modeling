import pandas as pd
import numpy as np

# Signal processing
from scipy import signal

from subfunctions.my_dropna_python import *
from subfunctions.numderiv import *
from subfunctions.scale_feature_data import *
from subfunctions.force_vec_taille_row import *
from subfunctions.make_a_properlist import *

from subfunctions.normal_distribution_feature_data import *
from subfunctions.tsig_2_discrete_wavelet_transform import *
from subfunctions.tsig_2_spectrogram import *
from subfunctions.tsig_2_continuous_wavelet_transform import *

from subfunctions.unsupervised_lab_kmeans_clustering import *


def feature_creation_preprocessing(feat0, t_feat0):


    # ----------------
    # Make your features
    # ----------------
    df_feat = pd.DataFrame()

    n = 4   # filter order
    fs = 250 # data sampling frequency (Hz)
    fcc = 10  # Cut-off frequency of the filter
    w = fcc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(n, w, 'low')  # 3rd order

    scales = np.arange(1, 128)

    print('num de samples avant dropna : ', len(feat0))

    # ----------------

    # Drop nan values from feat0
    temp = pd.DataFrame(feat0)
    temp0 = my_dropna_python(temp)
    feat0 = temp0.to_numpy()
    print('num de samples apres dropna : ', len(feat0))
    
    # ----------------
    
    # Need to find when one trial starts and end - take derivative from start-stop periods
    for i in range(len(feat0)):

        if i == 0:
            plotORnot = 0 #1
        else:
            plotORnot = 0
            
        # ----------------------------
        # Causale ordre
        # ----------------------------
        # (0) position
        col0 = scale_feature_data(feat0[i], plotORnot)
        col0 = np.ravel(col0)
        col0 = force_vec_taille_row(col0)

        # (1) velocity
        vel = numderiv(feat0[i], t_feat0[i])
        col1 = scale_feature_data(vel, plotORnot)
        col1 = make_a_properlist(col1)
        col1 = force_vec_taille_row(col1)

        # (2) acceleration
        acc = numderiv(vel, t_feat0[i])
        filtacc = signal.filtfilt(b, a, acc) # the signal is noisy
        col2 = scale_feature_data(filtacc, plotORnot)
        col2 = make_a_properlist(col2)
        col2 = force_vec_taille_row(col2)
        # ----------------------------


        # ----------------------------
        # Compter
        # ----------------------------
        num = i*np.ones((len(col0),))
        num = np.ravel(num)
        num = force_vec_taille_row(num)
        # print('num : ', num.shape)   # num :  (471,)
        # ----------------------------


        part1 = pd.DataFrame(data=[num, col0, col1, col2]).T
        # print('part1 : ', part1.shape)

        # ----------------------------
        # Non-causale ordre
        # ----------------------------
        # (4) position
        col3 = normal_distribution_feature_data(col0, plotORnot)
        col3 = make_a_properlist(col3)
        col3 = force_vec_taille_row(col3)

        # (5) velocity
        col4 = normal_distribution_feature_data(col1, plotORnot)
        col4 = np.ravel(col4)
        col4 = force_vec_taille_row(col4)

        # (6) acceleration
        col5 = normal_distribution_feature_data(col2, plotORnot)
        col5 = np.ravel(col5)
        col5 = force_vec_taille_row(col5)
        # ----------------------------

        part2 = pd.DataFrame(data=[col3, col4, col5]).T
        # print('part2 : ', part2.shape)

        # ----------------------------
        # Frequence marquers : sublevels of frequency pattern
        # ----------------------------
        # (7-22) une transformation de fréquence (ondelettes)
        coeff = tsig_2_discrete_wavelet_transform(col0, waveletname='sym5', level=5, plotORnot=0)
        cols6 = pd.DataFrame(coeff).T
        # print('cols6 : ', cols6.shape)# col6 :  (471, 5)

        coeff = tsig_2_discrete_wavelet_transform(col1, waveletname='sym5', level=5, plotORnot=0)
        cols7 = pd.DataFrame(coeff).T
        # print('cols7 : ', cols7.shape)  # col7 :  (471, 5)

        coeff = tsig_2_discrete_wavelet_transform(col2, waveletname='sym5', level=5, plotORnot=0)
        cols8 = pd.DataFrame(coeff).T
        # print('cols8 : ', cols8.shape)   # col8 :  (471, 5)
        # ----------------------------

        # ----------------------------
        # Hybrid marquers : temporalle et frequence information
        # ----------------------------
        # (8) spectrogram flatten - periodogram (fft)
        col9 = tsig_2_spectrogram(col0, fs=10, nfft=20, noverlap=0, plotORnot=0)
        col9 = np.ravel(col9)
        col9 = force_vec_taille_row(col9)

        col10 = tsig_2_spectrogram(col1, fs=10, nfft=20, noverlap=0, plotORnot=0)
        col10 = np.ravel(col10)
        col10 = force_vec_taille_row(col10)

        col11 = tsig_2_spectrogram(col2, fs=10, nfft=20, noverlap=0, plotORnot=0)
        col11 = np.ravel(col11)
        col11 = force_vec_taille_row(col11)
        # ----------------------------

        part3 = pd.DataFrame(data=[col9, col10, col11]).T
        # print('part3 : ', part3.shape)

        # ----------------------------
        # 2D continuous wavelet transform flattened
        # ----------------------------
        # continuous_wavelets = ['mexh', 'morl', 'cgau5', 'gaus5']
        col12 = tsig_2_continuous_wavelet_transform(t_feat0[i], col0, scales, waveletname='mexh', plotORnot=0)
        col12 = np.ravel(col12)
        col12 = force_vec_taille_row(col12)
        # print('col12 : ', col12.shape)   # col12 :  (471,)

        col13 = tsig_2_continuous_wavelet_transform(t_feat0[i], col1, scales, waveletname='mexh', plotORnot=0)
        col13 = np.ravel(col13)
        col13 = force_vec_taille_row(col13)
        # print('col13 : ', col13.shape)   # col13 :  (471,)

        col14 = tsig_2_continuous_wavelet_transform(t_feat0[i], col2, scales, waveletname='mexh', plotORnot=0)
        col14 = np.ravel(col14)
        col14 = force_vec_taille_row(col14)
        # print('col14 : ', col14.shape)   # col14 :  (471,)
        # ----------------------------

        part4 = pd.DataFrame(data=[col12, col13, col14]).T
        # print('part4 : ', part4.shape)


        # ----------------------------
        # kmeans
        # ----------------------------
        n_clusters = 2  # Note: c'est change par rapport ynum!! ynum=0 ou 1
        X = part1.iloc[:,1:4].to_numpy()
        kmeans, col15, centroids = unsupervised_lab_kmeans_clustering(n_clusters, X)
        col15 = np.ravel(col15)
        col15 = force_vec_taille_row(col15)
        # print('col15 : ', col15.shape)   # col15 :  (471,)

        n_clusters = 3  # Note: c'est change par rapport ynum!!  ynum=2
        kmeans, col16, centroids = unsupervised_lab_kmeans_clustering(n_clusters, X)
        col16 = np.ravel(col16)
        col16 = force_vec_taille_row(col16)
        # print('col16 : ', col16.shape)   # col16 :  (471,)
        # ----------------------------

        part5 = pd.DataFrame(data=[col15, col16]).T
        # print('part5 : ', part5.shape)

        # ----------------------------

        del num, col0, col1, col2, col3, col4, col5, coeff, col9, col10, col11, col12, col13, col14, col15, col16, kmeans, centroids, X

        temp = pd.concat([part1, part2, cols6, cols7, cols8, part3, part4, part5], axis=1)

        del part1, part2, cols6, cols7, cols8, part3, part4, part5

        df_feat = pd.concat([df_feat, temp], axis=0)

        del temp
    
    # ----------------


    return df_feat
