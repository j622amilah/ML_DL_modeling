# Created by Jamilah Foucher, Novembre 17, 2020 

# Purpose: The purpose of this function is to find the size (row, column) of a list named vec. 

# Input VARIABLES:
# (1) vec is the list
# 
# Output VARIABLES:
# (1) row and column dimension as a tuple
# 

# Example:
# vec1 = [[1, 1, 1, 1, 2, 2, 3, 2, 2, 1, 1, 4]]
# vec1 = [[1, 1, 1, 1], [2, 2, 3, 2], [2, 1, 1, 4]]
# print (vec1)
# [row, col] = size(vec1)
# print ('Overall= row: %s, col:%s' % (row, col))


# RETURNS
# [[1, 1, 1, 1, 2, 2, 3, 2, 2, 1, 1, 4]]
# Overall= row: 1, col:12

# [[1, 1, 1, 1], [2, 2, 3, 2], [2, 1, 1, 4]]
# Overall= row: 3, col:4

from collections import namedtuple
import numpy as np

def size(vec):
    rvec = []
    cvec = []
    coutr = 0
    
    for r in vec:
        #print(r)
        rvec = rvec + [int(1)]

        coutr += 1
        coutc = 0

        # print('length of r : ' + str(len(r)))
        # r = [int(x) for x in r]
        # OR
        temp = []
        for x in r:
            if np.isnan(x).any() == False:
                temp.append(int(x))
            else:
                temp.append(x)

        for col in r:
            #print(col)
            coutc += 1
        
        cvec = cvec + [coutc]

    # print ('row vec: %s, col:%s' % (rvec, cvec))

    # Overall ROW and COLUMN : some columns could be shorter than others take the largest column
    row = coutr
    col = max(cvec)
    #print ('Overall= row: %s, col:%s' % (row, col))

    #return (row, col)
    # OR
    # Use as namedtuple
    result = namedtuple('result', 'row00 col00')
    return result(row00=row, col00=col)
