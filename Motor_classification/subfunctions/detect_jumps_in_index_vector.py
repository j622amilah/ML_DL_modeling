# Created by Jamilah Foucher, October 16, 2020

# Purpose: Search for non-consecutive index points.
# This was useful when I used detect_sig_change_wrt_baseline, this showed how many times the signal went inside and outside of a desired y-axis/baseline/setpoint zone.  I used this function on the signal index outside or inside the zone, this gave me the non-consecutive index points (or spaces/jumps in the index) which correspond to when the signal was inside or outside the zone.

# Input VARIABLES:
# (1) vec is the index of a signal 

# Output VARIABLES:
# (1) index_st is the index of the break_st/value_st vector
#
# (2) break_st/value_st is the vec value of the first point in each group of consecutive data points in vec.
# break_st is a vector containing the of index values where a break starts
#
# (3) index_end is the index of the break_end/value_end vector
#
# (4) break_end/value_end is the vec value of the last point in each group of consecutive data points in vec
# break_end is a vector containing the index values where a break ends and starts

# ------------
# Example:
# ------------
# start_val = 0
# stop_val = 5
# ts = 1
# f = 1/ts  # sampling frequency
# N = int(f*stop_val)
# t = np.multiply(range(start_val, N), ts) 
# t2 = np.array(range(9,12))
# vec = np.concatenate((t, t2), axis=0)
# print('vec:' + str(vec))

# index_st, value_st, index_end, value_end = detect_jumps_in_index_vector(vec)

# print('value_st :' + str(value_st))
# print('value_end :' + str(value_end))

def detect_jumps_in_index_vector(vec):

    index_st = [0]
    value_st = [vec[0]]

    index_end = []
    value_end = []

    if len(vec) > 1:
        for i in range(len(vec)-1):
            # Check for end
            if i == len(vec)-1:
                index_end = index_end + [i+1]
                value_end = value_end + [vec[i+1]]
            elif vec[i] != vec[i+1]-1:
                # not consecutive data point
                index_end = index_end + [i]
                value_end = value_end + [vec[i]]

                index_st = index_st + [i+1]
                value_st = value_st + [vec[i+1]]
            #else:
                # Consecutive : continue searching

    return index_st, value_st, index_end, value_end
