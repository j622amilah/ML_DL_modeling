# File manipulation
import os
from os.path import exists

# Loading and saving data
import pickle

import numpy as np
import pandas as pd
import time

from collections import Counter

# Scaling signal
from mlxtend.preprocessing import minmax_scaling

# Signal processing
from scipy import signal
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import periodogram

# Wavelets
import pywt
from pywt import wavedec

# Sci-kit learn
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn import svm
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import cluster

# Tensorflow
import tensorflow as tf
from tensorflow import keras
tf.compat.v1.enable_eager_execution()  # This allows you to use placeholder in version 2.0 or higher
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, LeakyReLU, GlobalAveragePooling2D, BatchNormalization, Embedding, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers


# Visualization
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns

# Image processing
from PIL import Image

varr = {}
varr['main_path'] = "/home/oem2/Documents/9_Motor_classification_2018-22/Coding_version3_python_FINAL"  
varr['main_path1'] = "%s/a_data_standardization" % (varr['main_path'])
varr['main_path2'] = "%s/b_data_preprocessing" % (varr['main_path'])
varr['main_path3'] = "%s/c_calculate_metrics" % (varr['main_path'])

import sys
sys.path.insert(1, '%s' % (varr['main_path']))


from c_calculate_metrics.put_timeseries_trialdata_into_pandas import *
from subfunctions.isnan import *
from subfunctions.my_dropna_python import *
from subfunctions.is_empty import *
from subfunctions.make_a_properlist import *
from subfunctions.numderiv import *
from subfunctions.sort_dict_by_value import *
from subfunctions.interpretation_of_kstest import *
from subfunctions.normal_distribution_feature_data import *
from subfunctions.linear_intercurrentpt_makeshortSIGlong_interp1d import *
from subfunctions.tsig_2_discrete_wavelet_transform import *
from subfunctions.tsig_2_spectrogram import *
from subfunctions.tsig_2_continuous_wavelet_transform import *
from subfunctions.get_df import *
from subfunctions.indexit import *
from subfunctions.unsupervised_lab_kmeans_clustering import *
from subfunctions.create_labels_and_initial_feature import *
from subfunctions.scale_feature_data import *
from subfunctions.force_vec_taille_row import *
from subfunctions.confirm_joystick_sig import *
from subfunctions.feature_creation_preprocessing import *
from subfunctions.pandas_rename_columns import *
from subfunctions.pad_data_2makeclasses_equivalent import *
from subfunctions.detect_nonconsecutive_values_debut_fin_pt import *
from subfunctions.count_classes import *
from subfunctions.save_dat_pickle import *
from subfunctions.load_dat_pickle import *



# ----------------

# Load data
varr = {}
varr['main_path'] = "/home/oem2/Documents/9_Motor_classification_2018-22/Coding_version3_python_FINAL"
varr['main_path1'] = "%s/a_data_standardization" % (varr['main_path'])
varr['main_path2'] = "%s/b_data_preprocessing" % (varr['main_path'])
varr['main_path3'] = "%s/c_calculate_metrics" % (varr['main_path'])

df_timeseries_exp = put_timeseries_trialdata_into_pandas(varr)

# ----------------

exp = 'rot'
ax_val = 'all'
ss_val = 'all'

# ----------------


df = get_df(ax_val, ss_val, exp, df_timeseries_exp)

feat0, t_feat0, y1_feat0, y2_feat0, y3_feat0 = create_labels_and_initial_feature(df) 
# Elasped time for feature processing :  715.9615476131439

del df

# ----------------

# Confirm AGAIN that the joystick signal is correct.
# Look at the joystick features
fig = make_subplots(rows=1, cols=1)
for i in range(len(feat0)):
    fig.append_trace(go.Scatter(x=t_feat0[i], y=feat0[i],), row=1, col=1)

fig.update_layout(height=600, width=600, title_text="Avant")
fig.show()
# fig.write_image("Avant.png")

# We check the signal for Classification because we create the position feature in 
# the function above by selecting the joystick movement on the axis in which there
# was the stimulus.

# Explination for why there is not a joystick signal in each field of feat0:
# In previous steps, we time-locked all "start and stop" indexing with the time
# vector in the experiment. 
# And we confirmed using the movement of the simulator cabin using the cabin 
# position and the joystick, using both cabin and joystick direction and amplitude.

# At the moment no good reason why the data is not correct - we dropped all avnormal 
# data in step B (s1_removeBADtrials_savedata)

# ----------------

if exists("df_feat.pkl") == False:
    print('Creation des marquers')
    df_feat, rm_ind = feature_creation_preprocessing(feat0, t_feat0)
    del feat0, t_feat0
    save_dat_pickle(df_feat, file_name="df_feat.pkl")
    save_dat_pickle(rm_ind, file_name="rm_ind.pkl")
else:
    print('Load des marquers')
    df_feat = load_dat_pickle(file_name="df_feat.pkl")
    rm_ind = load_dat_pickle(file_name="rm_ind.pkl")

df_org = df_feat

del df_feat

# ----------------

mat = df_org.to_numpy()

t_feat0 = [t_feat0[i] for i in rm_ind]

fig = make_subplots(rows=1, cols=1)
for i in range(len(mat)):
    fig.append_trace(go.Scatter(x=t_feat0[i], y=mat[i],), row=1, col=1)

fig.update_layout(height=600, width=600, title_text="Apres")
fig.show()
# fig.write_image("Apres.png")



