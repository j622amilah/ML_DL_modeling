import numpy as np

from collections import Counter


from subfunctions.detect_nonconsecutive_values_debut_fin_pt import *



def count_classes(df_test_noNan):

    # Get start and end index values for each sample
    num = list(map(int, df_test_noNan.num.to_numpy()))
    st, endd = detect_nonconsecutive_values_debut_fin_pt(num)

    # ----------------

    yy = list(map(int, df_test_noNan.y.to_numpy()))
    y_short = []
    for i in range(len(st)):
        y_short.append(yy[st[i]:st[i]+1][0])

    # ----------------

    liste = Counter(y_short).most_common()
    count_index, counted_value = list(map(list, zip(*liste)))

    print('Before sorting counted_value : ', counted_value)
    print('Before sorting count_index : ', count_index)

    # ----------------

    # Sort counted_value by count_index; in ascending order
    sind = np.argsort(count_index)
    count_index = [count_index[i] for i in sind]
    counted_value = [counted_value[i] for i in sind]
    print('After sorting counted_value : ', counted_value)
    print('After sorting count_index : ', count_index)

    # ----------------

    # Determine how much to pad each class label
    needed_samps_class = np.max(counted_value) - counted_value
    print('needed_samps_class : ', needed_samps_class)

    # ----------------
    
    return needed_samps_class, counted_value, count_index, st, endd
