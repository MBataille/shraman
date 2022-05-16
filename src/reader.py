import numpy as np
import pandas as pd

from .shraman import SHRaman, params

def read_summary(branch):
    shr = SHRaman(branch=branch, **params)
    return pd.read_csv(shr.branchfolder + 's.csv')

def read_state(branch, filename):
    shr = SHRaman(branch=branch, **params)
    return shr.loadX(filename)[:params['N']]

def readX(branch, filename):
    shr = SHRaman(branch=branch, **params)
    return shr.loadX(filename)

def write_state(file, branch, filename):
    shr = SHRaman(branch=branch, **params)
    shr.saveX(file, filename=filename)