import math

# np.nan can not process strings,  use this.
def isnan(value):
    try:
        return math.isnan(float(value))
    except:
        return False
