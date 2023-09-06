# Created by Jamilah Foucher, 31/05/2021


import nltk
from nltk.stem import PorterStemmer

# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')

from string_text_processing.remove_stopwords import *


def preprocessing(text):
    # -------------------------------------
    # Creating word tokens (recognizing each word separately)
    # -------------------------------------
    # 1. Put the text into string format
    # Content = ""
    # for t in text:
        # Content = Content + t.lower().replace("'",'')

    # # 2. Tokenize first to get each character separate
    # tok = nltk.word_tokenize(Content)
    # print('length of tok: ' + str(len(tok)))
    
    # 3. Remove undesireable words from MY OWN stopword list
    word_tokens1 = remove_stopwords(text)
    
    # 5. Combining word stems 
    ps = PorterStemmer()
    word_tokens2 = []
    for w in word_tokens1:
        word_tokens2.append(ps.stem(w))
    
    # It does true stemming and actually deforms the word to a root word
    
    return word_tokens2