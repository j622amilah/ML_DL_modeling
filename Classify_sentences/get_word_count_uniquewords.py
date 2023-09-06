# Created by Jamilah Foucher, 31/05/2021

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

def get_word_count_uniquewords(word_tokens, list_to_remove):
    
    # -------------------------------------
    # Process word tokens
    # -------------------------------------
    vectorizer = CountVectorizer()

    # -------------------------------------
    # 1. Count word tokens and get a unique list of words : count how many times a word appears
    # Get the document-term frequency array: you have to do this first because it links cvtext to vectorizer
    X = vectorizer.fit_transform(word_tokens)
    word_count0 = np.ravel(np.sum(X, axis=0)) # sum vertically
    
    # Get document-term frequency list : returns unique words in the document that are mentioned at least once
    unique_words0 = np.ravel(vectorizer.get_feature_names())
    # -------------------------------------
    # 3. Remove undesireable words AGAIN and adjust the unique_words and word_count vectors
	
    # first let's do a marker method
    marker_vec = np.zeros((len(unique_words0), 1))

    # search for the remove tokens in tok, an put a 1 in the marker_vec
    for i in range(len(unique_words0)):
        for j in range(len(list_to_remove)):
            if unique_words0[i] == list_to_remove[j]:
                marker_vec[i] = 1
    
    unique_words = []
    word_count = []
    for i in range(len(marker_vec)):
        if (marker_vec[i] == 0) & (len(unique_words0[i]) > 4):
            unique_words.append(unique_words0[i])
            word_count.append(word_count0[i])
    
    m = len(np.ravel(word_count))
    # -------------------------------------
    
    # Matrix of unique words and how many times they appear
    mat = np.concatenate([np.reshape(np.ravel(word_count), (m,1)), np.reshape(unique_words, (m,1))], axis=1)

    print('There are ' + str(len(word_tokens)) + ' word tokens, but ' + str(len(unique_words)) + ' words are unique.')

    # 2. (Option) sort the unique_words by the word_count such that most frequent words are 1st
    # Gives the index of unique_word_count sorted from min to max
    sort_index = np.argsort(word_count)
    
    # Convert from matrix to array, so we can manipulate the entries
    # Puts the response vector in an proper array vector
    A = np.array(sort_index.T)

    # But we want the index of unique_word_count sorted max to min
    Ainvert = A[::-1]
    
    # Convert the array to a list : this is a list where each entry is a list
    Ainv_list = []
    for i in range(len(Ainvert)):
        Ainv_list.append(Ainvert[i])
        
    # Top num_of_words counted words in document : cvkeywords
    keywords = []
    wc = []
    p = np.ravel(word_count)
    
    #print('Ainv_list' + str(Ainv_list))
    
    top_words = len(Ainv_list)  # 20
    for i in range(top_words):
        keywords.append(unique_words[Ainv_list[i]])
        wc.append(p[Ainv_list[i]])
    
    # Matrix of unique words and how many times they appear
    mat_sort = np.concatenate([np.reshape(np.ravel(wc), (top_words,1)), np.reshape(np.ravel(keywords), (top_words,1))], axis=1)
    print(mat_sort)
    # -------------------------------------
    
    return wc, keywords, mat_sort