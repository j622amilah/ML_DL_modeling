# Created by Jamilah Foucher, 23/02/2022

import numpy as np

# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')

from make_a_properlist_str import *
from num_vec2category_vec import *

from string_text_processing.get_unique_sen_grpsWITHmax import *
from string_text_processing.pandas_visualize_senarr import *
from string_text_processing.minimally_regroup_senarr2 import *

def get_sentences_from_grp(df_grp1):

    senarr = df_grp1.iloc[:,0].to_numpy()
    senarr = make_a_properlist_str(senarr)

    # -------------------------

    # Make two sentence lists
    r = int(np.floor(len(senarr)/2))  # half of all sentences
    print('r: ', r)

    sen1_ar = senarr[0:r]
    sen2_ar = senarr[r::]

    return sen1_ar, sen2_ar


def get_unique_sen_grpsWITHmax_MAIN(sen1_ar, sen2_ar, sim_thresh, plotORnot):

    # ----------REPEATS---------------
    # step 1
    grp_new = get_unique_sen_grpsWITHmax(sen1_ar, sen2_ar, sim_thresh, plotORnot)

    # Check
    v_out = make_a_properlist_str(grp_new)
    v_out = make_a_properlist_str(v_out)
    orglen = len(v_out)
    print('Total sentences: ', orglen)
    print('Total sentences grouped : ', len(grp_new))

    # step 2
    df_grp1 = pandas_visualize_senarr(grp_new)
    # df_grp1

    # ----------REPEATS---------------
    while len(v_out) == orglen:
        # step 3
        sen1_ar, sen2_ar = get_sentences_from_grp(df_grp1)

        # step 4
        grp_new2 = get_unique_sen_grpsWITHmax(sen1_ar, sen2_ar, sim_thresh, plotORnot)

        # step 5
        # combine grp_new1 and grp_new2
        prev = len(grp_new)
        grp_new = grp_new, grp_new2
        grp_new = make_a_properlist_str(grp_new)

        # step 6
        # Condense sentence groups
        grp_new = minimally_regroup_senarr2(grp_new)
        
        # Check
        v_out = make_a_properlist_str(grp_new)
        v_out = make_a_properlist_str(v_out)
        print('Total sentences: ', len(v_out))
        print('Total sentences grouped : ', len(grp_new))

        # step 7
        df_grp1 = pandas_visualize_senarr(grp_new)

    # --------------------
    # Take the largest groups : we will have some groups that have significantly more sentences than other groups.
    # The groups need to be roughly balanced
    # --------------------
    
    # Determine which sentence groups are significant
    lens_of_grps = [len(i) for i in grp_new]
    print('The number of sentences per sentence group : ', lens_of_grps)

    # Use kmeans to find two groups based on the number of sentences per group
    df_vec = pd.Series(lens_of_grps)
    num_of_groups = 2
    cat_names = ['0', '1']
    centroids_sorted, kmean_cats, stat_cats, stats_cats_mean = num_vec2category_vec(df_vec, num_of_groups, cat_names)
    # print('centroids_sorted : ', centroids_sorted)
    # print('kmean_cats : ', kmean_cats)

    # Identify the groups with the most number of sentences 
    kmean_num = [int(i) for i in kmean_cats.to_numpy()]
    stat_num = [int(i) for i in stat_cats.to_numpy()]
    tot = [kmean_num[i]+stat_num[i] for i in range(len(kmean_num))]
    vec = make_a_properlist(tot)
    newvec, ind = findall(vec, '>=', 1)
    print('Main sentence groups : ', ind)

    grp_maxlen = [grp_new[i] for i in ind]
    print('Total sentences grouped : ', len(grp_maxlen))
    
    df_grp1 = pandas_visualize_senarr(grp_maxlen)
    df_grp1

    return grp_maxlen, df_grp1