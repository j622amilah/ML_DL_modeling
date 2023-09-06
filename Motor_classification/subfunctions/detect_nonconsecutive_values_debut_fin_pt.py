def detect_nonconsecutive_values_debut_fin_pt(vec):
  
    st = [0]
    endd = []
    
    for i in range(len(vec)-1):
        if vec[i] != vec[i+1]:
            st.append(i+1)
            endd.append(i)
    
    endd.append(len(vec)-1)
    
    return st, endd
