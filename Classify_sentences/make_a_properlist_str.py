# Created by Jamilah Foucher, 20/02/2022


# Note: If you have multiple nested array inside an array, specifically if the nested
# array is not first in the list, you will need to run it two times

def decend_into_array(vec):
    vec = vec[0]
    return vec



def make_a_properlist_str(a):
    v_out = []
    for i in range(len(a)):
        vec_sub = a[i]

        # determine how many to descend
        vec = vec_sub
        flag = 0
        c = 0
        while flag == 0:
            if isinstance(vec, list) == True:
                vec = decend_into_array(vec)
                c = c + 1
            else:
                flag = 1
        #print('c: ', c)

        # unravel c-1 times
        if c > 1:  # the if statement keeps square brackets for c=1 case
            for r in range(len(vec_sub)):
                v_out.append(vec_sub[r])
        else:
            v_out.append(vec_sub)
    return v_out