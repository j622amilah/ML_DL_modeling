# Created by Jamilah Foucher, 15/02/2022

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')

from string_text_processing.get_cossine_similarity import *



def calc_cossim_mat_of_2sen_arr(sen1_ar, sen2_ar, plotORnot):
    cossim_sen_mat = np.zeros((len(sen1_ar), len(sen2_ar)))
    for ind_s1, s1 in enumerate(sen1_ar):
        for ind_s2, s2 in enumerate(sen2_ar):
            cossim_sen_mat[ind_s1,ind_s2] = get_cossine_similarity(s1, ind_s2, sen2_ar)

    # -------------------------

    if plotORnot == 1:
        plt.figure(figsize=(14,7))

        # Heatmap showing average arrival delay for each airline by month
        # annot=True ensures that the values for each cell appear on the chart
        sns.heatmap(data=cossim_sen_mat)  

        # Add label for horizontal axis
        plt.title("Similarity of Question 1 and 2")
        plt.ylabel("Question 1")
        plt.xlabel("Question 2")

    # -------------------------

    return cossim_sen_mat