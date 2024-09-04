import numpy as np
import pandas as pd
import pathlib
from numba import jit

def read_trj_data(path):
    # p_temp = list( pathlib.Path("traindata"))
    exp_data = pd.read_csv(path,header=0, index_col=0 )
    return exp_data
