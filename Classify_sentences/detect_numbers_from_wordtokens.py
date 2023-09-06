# Created by Jamilah Foucher, 19/02/2022


# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')

from string_text_processing.remove_chars_from_wordtoken import *



def detect_numbers_from_wordtokens(wordtokens):

    wt_nums = []
    wt_letters = []
    for i in wordtokens:
        # Search to see if there are '.' at the end of the word
        i = remove_chars_from_wordtoken(i, '.', '')
        i = remove_chars_from_wordtoken(i, '%', '')
        i = remove_chars_from_wordtoken(i, '$', '')

        # First determine if a number is a ratio
        i = remove_chars_from_wordtoken(i, '/', ' divided by ')

        # Second determine if a number is present
        if i.upper() == i.lower() and i != '':
            # is a number
            wt_nums.append(int(i))  # save numbers as integers
        else:
            # is NOT a number
            wt_letters.append(i)
    print('The following words were present: ', wt_letters)


    # Save numbers separately from words, as strings
    imp_nums = np.unique(wt_nums)
    wt_nums_str = [str(i) for i in imp_nums]
    print('The following numbers were present: ', wt_nums)  

    return wt_nums, wt_nums_str, wt_letters

