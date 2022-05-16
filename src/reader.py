import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.interpolate import interp1d

from .shraman import SHRaman, params

def read_summary(branch):
    shr = SHRaman(branch=branch, **params)
    return pd.read_csv(shr.branchfolder + 's.csv')

def read_state(branch, filename, N=None):
    shr = SHRaman(branch=branch, **params)
    if N is None: N = params['N']
    return shr.loadX(filename)[:N]

def readX(branch, filename):
    shr = SHRaman(branch=branch, **params)
    return shr.loadX(filename)

def write_state(file, branch, filename):
    shr = SHRaman(branch=branch, **params)
    shr.saveX(file, filename=filename)

def read_belgium(file):
    mat = loadmat(file)
    return mat

def interpolate(x, y, Nnew):
    xnew = np.linspace(x[0], x[-1], Nnew, endpoint=False)

    print(x.shape, y.shape)

    f = interp1d(x, y)
    
    return f(xnew)