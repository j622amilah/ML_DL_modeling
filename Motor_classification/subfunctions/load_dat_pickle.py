# Loading and saving data
import pickle

def load_dat_pickle(file_name="outSIG.pkl"):
    open_file = open(file_name, "rb")
    dataout = pickle.load(open_file)
    open_file.close()
    return dataout
