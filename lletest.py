#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import (advancePALC, SHRaman, animateBifDiag, 
                plotBifDiags, parameterSweep, getPrange,
                readParameterSweep, read_state, read_summary, 
                readX, read_belgium, interpolate, write_state,
                LugiatoLefeverEquation)
from src.reader import write_state
import sys

# %%
params = {
    'Delta': 1.7,
    'S': 1.2,
    'f_R': 0.05,
    'tau0': 1,
    'tau1': 3,
    'tau2': 10,
    'beta': 1.0,
    'dx': 0.25,
    'N' : 512,
    'd4' : -1.0
}

lle = LugiatoLefeverEquation(branch='lle_dns_test', **params)

X = lle.loadX()

N = lle.getParam('N')
A0 = X[:N] + 1j * X[N:2*N]
c0, S0 = X[2*N:2*N+2]

lle.setInitialCondition(A0)
lle.solve_dns()
Af = lle.getState(-1)
lle.saveState()
plt.plot(np.abs(Af) ** 2)
plt.show()

#lle.init_cont(X, 0.1)
#%%

plotBifDiags('lle_palc_test_f_R=0.05')
#animateBifDiag('lle_palc_test_', **params)

#%%

plt.plot(X)
plt.show()
t0 = np.zeros_like(X)
t0[-1] = 1
with threadpool_limits(limits=1):
    advancePALC(X, 1e-2, t0=t0, branch='lle_palc_d4', equation=LugiatoLefeverEquation, **params)