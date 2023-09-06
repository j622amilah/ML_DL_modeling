# Created by Jamilah Foucher, 15/02/2022

import numpy as np

import nltk
from nltk.corpus import stopwords

import requests
from bs4 import BeautifulSoup


# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')


from make_a_properlist import *
from string_text_processing.parse_soup_split import *
from string_text_processing.remove_tokens_within_stopwords import *


def text_url_2_senANDwordtokens(inputurl, name_txtfile):

    # -----------------------------------------
    
    # 1) Get URL page
    page = requests.get(inputurl)

    # Scrape webpage : soup is the html document
    soup = BeautifulSoup(page.content, 'html.parser')

    # display status code
    # print(page.status_code)
    
    # Get the first paragraph of text
    # page = soup.find('p').getText()
    
    # Get all the paragraphs of text
    # page = soup.find_all('p')
    
    # Get the all the hyperlinks in the text
    # get_hyperlinks = soup.find_all("a")
    
    # Get all the paragraphs of text in terms of b'<html> tags and tokenized
    text = page.content.split()

    # -----------------------------------------
    
    # Get start-stop points to get text in-between html paragraphs: <p>  </p>
    st = []
    ender = []
    for ind, x in enumerate(text):
        i = str(x)
        out = i.partition('<')
        out2 = out[2].partition('>')
        
        if out2[0] == 'p': # word inbetween brackets
            st.append(ind)
        elif out2[0] == '/p': # word inbetween brackets
            ender.append(ind)
        
        # Check if there is a </p><p> combination
        if out2[2][0:3] == '<p>':
            st.append(ind)
            
    # print('st: ', st)
    # print('length of st: ', len(st))
    # print('ender: ', ender)
    # print('length of ender: ', len(ender))
    
    # -----------------------------------------
    
    # -----------------------------------------
    # Get paragraph text between <p>  </p>
    min_len = np.min([len(st), len(ender)])
    text_out = []
    for i in range(min_len):
        text_out.append(text[st[i]:ender[i]])
    # -----------------------------------------
    
    # Removes characters from a word token
    dw_token = [] # desired word tokens
    for h in range(len(text_out)):
        p_w_token = []  # paragraph word tokens
        for ind, word in enumerate(text_out[h]):
            #print('ind: ', ind)

            token = str(word)
            #print('token: ', token)

            # We know that the 1st value in array text_out has <p>
            if ind == 0:
                # parse by <p>
                partition_str = '<p>'
            else:
                partition_str = "'"
                

            desired_token = parse_soup_split(token, partition_str)
            #print('desired_token: ', desired_token)
            
            p_w_token.append(desired_token)
            
        dw_token.append(p_w_token)
    
    # -----------------------------------------
    
    vecout = make_a_properlist(dw_token)

    all_val = []
    for i in range(len(vecout)):
        for j in range(len(vecout[i])):
            all_val.append(vecout[i][j])
    
    # -----------------------------------------

    # Hyperlinks denoted by <a> </a> remain in word tokens
    # For now, just remove the arrays completely until I figure out how to parse it

    list_to_remove = ['<a', 'id=', "href=", 'title=', 'class=', '</a>', '\\']
    word_tok0 = remove_tokens_within_stopwords(all_val, list_to_remove)
    
    word_tok = []
    for i in word_tok0:
        if i != '':
            word_tok.append(i)
    
    # -----------------------------------------
    
    # Separate word tokens into sentences
    eofs = []
    for ind, word in enumerate(word_tok):
        
        # look for '.' End Of Sentence
        j = "."
        gp_char_token = word.partition(j)
        
        # Check if the list_to_remove char-word is in the word token
        # If True, do not include word and go to the next work token
        if j in gp_char_token:
            #print('j: ', j)
            #print('gp_char_token: ', gp_char_token)
            eofs.append(ind)
    print('There are %d sentences' % (len(eofs)))
    
    sen = []
    for i in range(len(eofs)):
        if i == 0:
            st = 0
            ender = eofs[i]
        else:
            st = eofs[i-1]+1
            ender = eofs[i]
        #print('word_tok[st:ender]', word_tok[st:ender])
        if len(word_tok[st:ender]) > 2:
            sen.append(make_a_properlist(word_tok[st:ender]))
        else:
            sen.append(word_tok[st:ender])
    
    # -----------------------------------------

    # Save text to file
    filename = '%s.txt' % (name_txtfile)
    with open(filename, 'w') as f:
        for line in word_tok:
            f.write(line)
            f.write('\n')
    # -----------------------------------------

    return sen, word_tok