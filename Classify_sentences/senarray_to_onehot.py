# Created by Jamilah Foucher, 26/02/2022


# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')

from make_a_properlist import *
from findall import *
from is_empty import *

from string_text_processing.make_a_properlist_str import *
from string_text_processing.preprocessing import *
from string_text_processing.get_word_count_uniquewords import *
from string_text_processing.detect_numbers_from_wordtokens import *



def senarray_to_onehot(grp_new):
    
    list_of_sens = make_a_properlist_str(grp_new)
    list_of_sens = make_a_properlist_str(list_of_sens)
    tot_sens = len(list_of_sens)
    
    word_tokens = make_a_properlist(grp_new)
    word_tokens = make_a_properlist(word_tokens)
    # print('word_tokens: ', word_tokens)

    # Get the theme of the knowledge base
    word_tokens2 = preprocessing(word_tokens)

    list_to_remove = ['https']
    wc, keywords, mat_sort = get_word_count_uniquewords(word_tokens2, list_to_remove)

    # Remove all numbers from the keyword list
    wt_nums, wt_nums_str, keywords2 = detect_numbers_from_wordtokens(keywords)
    
    X = np.zeros((len(keywords2), tot_sens))
    c = 0
    for senarr in grp_new:
        for sen in senarr:
            for word in sen:
                newvec, ind_newvec = findall(keywords2, '==', word)

                if is_empty(ind_newvec) == False:
                    for i in ind_newvec:
                        # print('i : ', i)
                        # print('c : ', c)
                        X[i,c] = 1
            c = c + 1
    
    X = X.T
    print('size of X (sentences, keywords): ', X.shape)
    
    return X, keywords2