import numpy as np

from subfunctions.findall import *


def standarization_fill_in_matrix(ind_joy_ax_moved, cab_index_viaSS, joy_ax_index, dirC, cab_dir, joy_ax_dir):

    # -------------------------------------------------
    # Fill in matrix 
    # Need to determine and classify the order of movements
    # -------------------------------------------------

    # bin_m, sca_ind, and d_m stay zero if : 1) cabin did not move but joy moved
    bin_m = np.zeros((3,1))
    sca_ind = np.zeros((3,1))
    d_m = np.zeros((3,1))

    # We search over the axes that were initially moved (ax) for each trial (i)
    for ax in ind_joy_ax_moved:
        # This is for the joy-cabin response timing
        if cab_index_viaSS[ax] != 0:
            # There is cabin detection! This value should always be positive
            cabjoy_diff_index = (cab_index_viaSS[ax] - joy_ax_index[ax] + 1)
        else:
            # No cabin movement detected - this measure is irrelavant so leave at zeros
            cabjoy_diff_index = 0

        # ***********************
        # a] binary_marker : Did the cabin move after joystick movement? 1=yes,cab follow ok, 0=no
        # ***********************
        if cabjoy_diff_index > 0:
            bin_m[ax] = 1
            
            # ***********************
            # b] scalar_marker : What is the time difference between joystick and cabin movement?: t_delay 
            # ***********************
            sca_ind[ax] = cabjoy_diff_index*0.004  # when this equals 0 when cab_index=0 (cabin did not move)
            
            # ***********************
            # c] direction_marker : Was the direction of joy movement with respect to cabin movement ok, 
            # according to the defined conventions (case 0 and 1)?: 1=correct, 0=not correct
            # ***********************
            # print('dirC :' + str(dirC))
            if dirC == 0:
                # Convention 1 : (cab = cab_com - joystick), thus (cab = -joy)
                if cab_dir[ax] == -joy_ax_dir[ax]:
                    d_m[ax] = 1  # 1=correct
                else:
                    d_m[ax] = 0  # 0=not correct
            elif dirC == 1:
                # Convention 2 : (cab = joystick - cab_com), thus (cab = joy)
                if cab_dir[ax] == joy_ax_dir[ax]:
                    d_m[ax] = 1  # 1=correct
                else:
                    d_m[ax] = 0  # 0=not correct
            

    bin_m = [int(x) for x in bin_m]
    sca_ind = np.ravel(sca_ind)
    d_m = [int(x) for x in d_m]

    # print('bin_m :' + str(bin_m))
    # print('sca_ind :' + str(sca_ind))
    # print('d_m :' + str(d_m))

    return bin_m, sca_ind, d_m                      