# Created by Jamilah Foucher, 26/02/2022


# Personal python functions
import sys
sys.path.insert(1, 'C:\\Users\\jamilah\\Documents\\Subfunctions_python')

from is_empty import *
from make_a_properlist import *

from string_text_processing.remove_chars_from_wordtoken import *
from string_text_processing.is_sen_in_senarray import *
from string_text_processing.detect_numbers_from_wordtokens import *
from string_text_processing.preprocessing import *
from string_text_processing.get_word_count_uniquewords import *


def findall_nonempty(vec):
    indy = []
    for ind, i in enumerate(vec):
        if is_empty(i) == False:
            indy.append(ind)

    return indy


def reshuffle_repeating_keywords_2A_single_grp(topkeywords, topwc, grp_new):
	# Try 1

	# Generate a set of combinations: n must be bigger than 2 
	n = len(topkeywords)
	a = []
	b = []
	cou = 1
	q = list(range(n))
	for i in q[::-1]:
		for j in range(cou, n):
			if 2 % n == 0:
				a.append(n-i-2)
			else:
				a.append(n-i-1)
			b.append(j)
		cou = cou + 1
	print('a: ', a)
	print('b: ', b)


	# Using the total possible sets: find repeating words
	keep_across_sens = []
	for i in range(len(a)):
		
		grp_cur = topkeywords[a[i]]
		grp_next = topkeywords[b[i]]
		grpwc_cur = topwc[a[i]]
		grpwc_next = topwc[b[i]]
		keeper = []
		for word in grp_cur:
			indy, result = is_sen_in_senarray(word, grp_next)
			
			# we only want common words: 
			if result == True:
				indy_cur, result = is_sen_in_senarray(word, grp_cur)
				
				wc_next = [grpwc_next[q] for q in indy]
				wc_cur = [grpwc_cur[q] for q in indy_cur]
				keeper.append([wc_cur, wc_next, word])
			   
		keep_across_sens.append(keeper)
		
		
		
	i_val = findall_nonempty(keep_across_sens)
	i_val

	which_grps = []
	for i in i_val:
		grp1 = a[i] # scalar
		grp2 = b[i] # scalar
		repeaters = keep_across_sens[i] # is a list
		for q in range(len(repeaters)):
			repword = repeaters[q][2]
			wc_cur = repeaters[q][0][0]
			wc_next = repeaters[q][1][0]
			
			print('repword: ', repword)
			
			print('wc_cur: ', wc_cur)
			print('wc_next: ', wc_next)
			
			if wc_cur > wc_next:
				# move all sentences from grp2 with repword to grp1
				grp_new_2mod = []
				for z in grp_new[grp2]:
					x = [remove_chars_from_wordtoken(r, '?', '').lower() for r in z]
					x = [remove_chars_from_wordtoken(r, '!', '').lower() for r in x]
					x = [remove_chars_from_wordtoken(r, '.', '').lower() for r in x]
					grp_new_2mod.append(x)
					
				ind_grp2 = []
				for ii, sen in enumerate(grp_new_2mod):
					indy, out = is_sen_in_senarray(repword, sen)
					if out == True:
						ind_grp2.append(ii)
				print('ind_grp2: ', ind_grp2)
				
				# add sentences to grp1
				get_sen = [grp_new[grp2][x] for x in ind_grp2]
				grp_new[grp1] = grp_new[grp1] + get_sen
				
				# remove sentences from grp2
				tokeep = np.setdiff1d(list(range(len(grp_new[grp2]))), ind_grp2)
				grp2_new = [grp_new[grp2][x] for x in tokeep]
				# print('grp2_new: ', grp2_new)
				grp_new[grp2] = grp2_new
			else:
				# move all sentences from grp1 with repword to grp2
				
				# need to lowercase and remove characters
				grp_new_1mod = []
				for z in grp_new[grp1]:
					x = [remove_chars_from_wordtoken(r, '?', '').lower() for r in z]
					x = [remove_chars_from_wordtoken(r, '!', '').lower() for r in x]
					x = [remove_chars_from_wordtoken(r, '.', '').lower() for r in x]
					grp_new_1mod.append(x)
				
				ind_grp1 = []
				for ii, sen in enumerate(grp_new_1mod):
					indy, out = is_sen_in_senarray(repword, sen)
					if out == True:
						ind_grp1.append(ii)
				print('ind_grp1: ', ind_grp1)
				
				# add sentences to grp2
				get_sen = [grp_new[grp1][x] for x in ind_grp1]
				grp_new[grp2] = grp_new[grp2] + get_sen
				
				# remove sentences from grp2
				tokeep = np.setdiff1d(list(range(len(grp_new[grp1]))), ind_grp1)
				grp1_new = [grp_new[grp1][x] for x in tokeep]
				# print('grp1_new: ', grp1_new)
				grp_new[grp1] = grp1_new
				
				
	# Rerun topkeywords to confirm
	
	# Find keywords per sentence group
	topwc = []
	topkeywords = []
	for sgrp in grp_new:
		word_tokens = make_a_properlist(sgrp)
		word_tokens = make_a_properlist(word_tokens)
		# print('word_tokens: ', word_tokens)

		# Get the theme of the knowledge base
		word_tokens2 = preprocessing(word_tokens)

		list_to_remove = ['https']
		wc, keywords, mat_sort = get_word_count_uniquewords(word_tokens2, list_to_remove)

		# Remove all numbers from the keyword list
		wt_nums, wt_nums_str, keywords2 = detect_numbers_from_wordtokens(keywords)

		topwc.append(wc[0:10])
		topkeywords.append(keywords2[0:10])
		
	return topkeywords, keywords2, grp_new