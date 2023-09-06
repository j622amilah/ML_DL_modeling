# Get the joystick stimulus axis data and put it in a pandas column
def indexit(row):
    joy_mat = [row.JOY_ax0, row.JOY_ax1, row.JOY_ax2]
    return joy_mat[int(row.ax)]
