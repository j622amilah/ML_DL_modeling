# Created by Jamilah Foucher, October 16, 2020

# Purpose: Search for non-consecutive index points.
# This was useful when I used detect_sig_change_wrt_baseline, this showed how many times the signal went inside and outside of a desired y-axis/baseline/setpoint zone.  I used this function on the signal index outside or inside the zone, this gave me the non-consecutive index points (or spaces/jumps in the index) which correspond to when the signal was inside or outside the zone.

# Input VARIABLES:
# (1) vec is the index of a signal 

# Output VARIABLES:
# (1) break_st is a vector containing the of index values where a break starts
#
# (2) break_end is a vector containing the index values where a break ends and starts

# [break_st : break_end]

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

# break_st, break_end = detect_jumps_in_index_vector_simple(vec)

# print('break_st :' + str(break_st))
# print('break_end :' + str(break_end))


def detect_jumps_in_index_vector_simple(vec):

    # Initialize outputs
    break_st = [vec[0]]
    break_end = []      # if empty the vector does not pass again through zero
    
    if len(vec) > 1:
        for i in range(len(vec)-1):
            # Check for end
            if i == len(vec)-1:
                break_end = break_end + [vec[i+1]]
            elif vec[i] != vec[i+1]-1:
                # not consecutive data point
                break_end = break_end + [vec[i]]
                break_st = break_st + [vec[i+1]]
            # else:
                # Consecutive : continue searching (just for thinking)
    
    return break_st, break_end