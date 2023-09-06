import nltk
from nltk.corpus import stopwords

import numpy as np

# Checks to see if the entire word token matches the list_to_remove

# Created by Jamilah Foucher, 31/05/2021



def remove_stopwords(wordtokens):
    
    # Put words that are 4 characters long or more, like a name, location, etc that you do not want to process
    list_to_remove = ["gmail", "gmail.com", "https"]
    
    # first let's do a marker method
    marker_vec = np.zeros((len(wordtokens), 1))

    # search for the remove tokens in tok, an put a 1 in the marker_vec
    for i in range(len(wordtokens)):
        for j in range(len(list_to_remove)):
            if wordtokens[i] == list_to_remove[j]:
                marker_vec[i] = 1

    word_tokens0 = []
    for i in range(len(marker_vec)):
        if (marker_vec[i] == 0) & (len(wordtokens[i]) > 4): # this will remove tokens that are 3 characters or less 
            word_tokens0.append(wordtokens[i])
            
    # 4. Removing stopwords using sklearn              
    stop_words = set(stopwords.words('english')) # does not remove "and" or "or"
    word_tokens1 = []
    for w in word_tokens0: 
        if w not in stop_words: 
            word_tokens1.append(w)
    
    return word_tokens1