def minimally_regroup_senarr():

	grp_str = []
	grp_ind = []

	for i in range(len(grp_red)):
		for j in range(len(grp_red[i])):
			grp_str.append(grp_red[i][j])
			grp_ind.append(i)

	# find unique string value
	ind_new, uq_senarray = unique_str_arrays(grp_str)
	print('uq_senarray: ', uq_senarray)


	# get unique strings per grp
	grps = []
	for i in uq_senarray:
		# print('i: ', i)
		val,indy = findall(grp_str, '==', i)
		# print('indy: ', indy)
		out = [grp_ind[q] for q in indy]
		# print('out: ', out)
		grps.append(out)
	print('grps: ', grps)
	  

	# each number in each set of grps is the sentence group number
	# each set is a unique sentence = uq_senarray

    # -------------------------
    
	# could get unique sets first 
	ind, uq_grps = unique_str_arrays(grps)
	print('uq_grps: ', uq_grps)
	print('ind: ', ind)

	# then check if smaller sets are in larger sets
	len_of_sets = [len(i) for i in uq_grps]
	print('len_of_sets: ', len_of_sets)
    
    # -------------------------
    
    # sort len_of_sets from small to big
    sort_index = np.argsort(len_of_sets)
    print('sort_index: ', sort_index)

    s_len_of_sets = [uq_grps[i] for i in sort_index]
    print('s_len_of_sets: ', s_len_of_sets)
    # s_len_of_sets is a list of most reduced cross-referenced groups 
    
    # -------------------------
    
    arr = []
    for i in range(len(s_len_of_sets)-1):
        
        print('i: ', i)
        longer = s_len_of_sets[i+1]
        shorter = s_len_of_sets[i]
        print('longer: ', longer)
        print('shorter: ', shorter)
        out = np.setdiff1d(longer, shorter) 
        print('out: ', out)
        
        if is_empty(out) == True: # shorter == longer ---> combine
            keep_val = shorter, longer
            keep_val = make_a_properlist_str(keep_val)
            
        elif len(out) == len(longer): # same size set but ALL different elements
            # check if shorter is in arr, if YES - only keep longer, if NO - keep shorter and longer
            indy = []
            for q in vec:
                print('q: ', q)
                
                for ae_ind, arr_element in enumerate(arr):
                    arr_element = make_a_properlist(arr_element)
                    print('arr_element: ', arr_element)
                    for arr_entry in arr_element:
                        if arr_entry == q:
                            indy.append(ae_ind)
                print('indy: ', indy)
            if is_empty(indy) == False: # NOT empty
                # case: vec is already in arr, so only save longer
                keep_val = longer
            else:
                keep_val = shorter, longer
            keep_val = make_a_properlist_str(keep_val)
            
        elif len(out) < len(longer):  # some elements in longer are in shorter
            
            # we know that shorter is the last entry in arr - remove last entry
            arr = arr[0:-1]
            # then, replace shorter with longer
            keep_val = longer # keep longer
            
            for q in longer:
                print('q: ', q)
                val, indy = findall(arr, '==', [q])
                print('indy: ', indy)
                if is_empty(indy) == False:
                    #remove indy from array
                    if indy[0] == 0:
                        arr = arr[1::]
                    elif indy[0] == len(arr)-1:
                        arr = arr[0:-1]
                    else:
                        arr = arr[0:indy[0]-1], arr[indy[0]+1::]
                    print('arr: ', arr)
                    
        print('keep_val: ', keep_val)
        arr.append(keep_val)
        
        print('arr loop fin: ', arr)
    arr