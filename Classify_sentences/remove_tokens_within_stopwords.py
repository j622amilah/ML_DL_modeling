# Created by Jamilah Foucher, 15/02/2022

# Checks to see if a part of the word token matches the list_to_remove

def remove_tokens_within_stopwords(all_val, list_to_remove):
    word_tok = []
    
    for i in all_val:
        flag = 0
        cnt = 0

        while flag == 0:
            j = list_to_remove[cnt]
            gp_char_token = i.partition(j)

            # Check if the list_to_remove char-word is in the word token
            # If True, do not include word and go to the next work token
            if j in gp_char_token:
                #print('j: ', j)
                #print('gp_char_token: ', gp_char_token)
                flag = 1
            else:
                cnt = cnt + 1

            # You reached the end of list_to_remove 
            if cnt == len(list_to_remove):
                # You searched the entire list, so just save the word
                flag = 1

                # save word once 
                word_tok.append(i)
            
    return word_tok