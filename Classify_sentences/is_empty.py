def is_empty(vec):
    
    if not any(vec) and len(vec) < 1:
        # print('yes, the array is empty')
        out = True
    else:
        # print('no, the array is not empty')
        out = False
        
    return out