# Created by Jamilah Foucher, 15/02/2022

def separate_token(token, partition_str):
    
    # Separate word using single quotes
    #print('token:', token)
    #gp_char_token = token.partition("'")  # group of char per token
    gp_char_token = token.partition(partition_str)  # group of char per token
    #print('gp_char_token: ', gp_char_token)

    return gp_char_token

def get_desired_token_from_gpchartoken(gp_char_token):
    # Remove b and single quotes
    token = []
    for i in gp_char_token:
        if i not in ['b', "'", "b'<blockquote>", '<p>']:
            #print('i:', i)
            #print('response:', i not in ['b', "'", 'b<blockquote>', '<p>'])
            i = i.replace("'",'')
            i = i.replace("]",'')
            i = i.replace("[",'')
            i = i.replace("<b>",'')
            i = i.replace("</b>",'')
            i = i.replace('"','')
            i = i.replace('<>','')
            i = i.replace('</>','')
            i = i.replace('<i>','')
            i = i.replace('</p><p>','')
            i = i.replace('</p>','')
            i = i.replace(';','')
            i = i.replace('&','')
            i = i.replace('#','')
            i = i.replace('<sup','')
            i = i.replace('<sup','')
            i = i.replace('<sup','')
            i = i.replace(',','')
            i = i.replace('...','')
            i = i.replace(')','')
            i = i.replace('(','')
            token.append(i)
    #print('token: ', token)
    
    return token
	
	
	
	
def parse_soup_split(token, partition_str):
    gp_char_token = separate_token(token, partition_str)
    token = get_desired_token_from_gpchartoken(gp_char_token)
        
    return token