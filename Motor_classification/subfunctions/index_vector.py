# Index certain entries in a vector using an index vector, return the remaining vector.

def index_vector(ind, vec):

    # Ensure index is an integer
    ind = [int(x) for x in ind] 
    indexedvec = []
    for i in ind:
        indexedvec = indexedvec + [vec[i]]
        
    return indexedvec