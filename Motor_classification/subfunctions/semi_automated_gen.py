# Indexes the matrix vec, containing the vectors of specific conditions
# To prevent errors

def semi_automated_gen(inner_name, outer_name, vec, inner_name_list, outer_name_list):
	
	# Rename specific names to general names
	cou = 0
	for i in inner_name_list:
		if inner_name == i:
			inner_name_gen = 'inner%d' % (cou)
		cou = cou + 1
		
	# Rename specific names to general names
	cou = 0
	for i in outer_name_list:
		if outer_name == i:
			outer_name_gen = 'outer%d' % (cou)
		cou = cou + 1
	
	num_cat_outer = vec.shape[0]  # outer condition (sub/sup)
	num_cat_inner = vec.shape[1]  # inner condition (IC, EC, NC, NR)
	
	# Just make it really long : there is a limit to the number of inner categories
	if inner_name_gen == 'inner0':
		if outer_name_gen == 'outer0':
			condi_vec = vec[0][0]
		elif outer_name_gen == 'outer1':
			condi_vec = vec[1][0]
		elif outer_name_gen == 'outer2':
			condi_vec = vec[2][0]
		elif outer_name_gen == 'outer3':
			condi_vec = vec[3][0]
		elif outer_name_gen == 'outer4':
			condi_vec = vec[4][0]
	
	elif inner_name_gen == 'inner1':
		if outer_name_gen == 'outer0':
			condi_vec = vec[0][1]
		elif outer_name_gen == 'outer1':
			condi_vec = vec[1][1]
		elif outer_name_gen == 'outer2':
			condi_vec = vec[2][1]
		elif outer_name_gen == 'outer3':
			condi_vec = vec[3][1]
		elif outer_name_gen == 'outer4':
			condi_vec = vec[4][1]
	
	elif inner_name_gen == 'inner2':
		if outer_name_gen == 'outer0':
			condi_vec = vec[0][2]
		elif outer_name_gen == 'outer1':
			condi_vec = vec[1][2]
		elif outer_name_gen == 'outer2':
			condi_vec = vec[2][2]
		elif outer_name_gen == 'outer3':
			condi_vec = vec[3][2]
		elif outer_name_gen == 'outer4':
			condi_vec = vec[4][2]
			
	elif inner_name_gen == 'inner3':
		if outer_name_gen == 'outer0':
			condi_vec = vec[0][3]
		elif outer_name_gen == 'outer1':
			condi_vec = vec[1][3]
		elif outer_name_gen == 'outer2':
			condi_vec = vec[2][3]
		elif outer_name_gen == 'outer3':
			condi_vec = vec[3][3]
		elif outer_name_gen == 'outer4':
			condi_vec = vec[4][3]
			
	elif inner_name_gen == 'inner4':
		if outer_name_gen == 'outer0':
			condi_vec = vec[0][4]
		elif outer_name_gen == 'outer1':
			condi_vec = vec[1][4]
		elif outer_name_gen == 'outer2':
			condi_vec = vec[2][4]
		elif outer_name_gen == 'outer3':
			condi_vec = vec[3][4]
		elif outer_name_gen == 'outer4':
			condi_vec = vec[4][4]
	
	
	return condi_vec
