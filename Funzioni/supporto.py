import pickle
import numpy as np

def save_data_py(dati,filename):
    datifile = open(filename,'wb')
    pickle.dump(dati,datifile)
    datifile.close()
    return

def load_data_py(filename):
    datifile = open(filename,'rb')
    x = pickle.load(datifile)
    datifile.close()
    return x