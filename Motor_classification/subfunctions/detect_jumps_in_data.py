def detect_jumps_in_data(*arg):

    y = arg[0]
    dp_jump = arg[1]

    if len(arg) > 2:
        desired_break_num = arg[2]
    else:
        desired_break_num = len(y)  # default, select a large number (ie: the length of the dataset)

    # Optional input, only use if you need to index subsections of a larger vector
    if len(arg) > 3:
        y_index_holder = arg[3]  # the starting index of each vector subsection
    else:
        y_index_holder = 0  # default

    c = 1
    ind_jumpPT = []
    # 1) Check the difference between consecutive points
    for j in range(len(y)-1):
        if abs(y[j] - y[j+1]) > dp_jump:
            if len(arg) > 3:
                ind_jumpPT = ind_jumpPT + [y_index_holder + j - 1]   # where trial should start, to remove jump portion before
            else:
                ind_jumpPT = ind_jumpPT + [j] # x-axis point during trial when jumped

            # If you want to stop searching the data after a certain number of breaks - ie: for when the robot stopped put 1 (the data for that trial is not good), if you want to search the entire length of the data put the length of y
            if c == desired_break_num:
                break
            c = c + 1


    return ind_jumpPT