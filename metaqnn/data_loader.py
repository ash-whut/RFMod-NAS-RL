import os
import sys
from sklearn.model_selection import train_test_split

import h5py as h5
import numpy as np

def load_data(path: str):
    try:
        file_handle = h5.File(path,'r+')
    except ValueError:
        print(f"Error: can't open HDF5 file '{path}' for reading (it might be malformed) ...")
        sys.exit(-1)
    x = file_handle['X'][:]
    y = file_handle['Y'][:]
    z = file_handle['Z'][:]
    return train_test_split(x, y, z, test_size=0.2, random_state=0)

def validation_data(x_tst: tuple, y_tst: tuple):
    return train_test_split(x_tst, y_tst, test_size=0.5, random_state=0)