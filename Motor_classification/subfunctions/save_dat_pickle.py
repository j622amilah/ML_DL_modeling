def save_dat_pickle(outSIG, file_name="outSIG.pkl"):
    # Save data matrices to file
    open_file = open(file_name, "wb")
    pickle.dump(outSIG, open_file)
    open_file.close()
