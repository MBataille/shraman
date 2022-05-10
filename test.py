#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import advancePALC, SHRaman, params, animateBifDiag, plotBifDiags, parameterSweep, getPrange, readParameterSweep


params['gamma'] = 0.24
params['eta'] = 0.0

#animateBifDiag('periodic_gamma=0.24', branches_ref=('hss_gamma=0.24', ), colors=['tab:blue', 'tab:green'], **params)
#plotBifDiags(('periodic_gamma=0.24','periodic_back_gamma=0.24' ),'bs1_gamma=0.24', 'hss_gamma=0.24')


shr = SHRaman(branch = 'bs1_dns', **params)

X = np.zeros(params['N']) +  shr.getHSS()

X = shr.loadX('bs1_gamma=0.24')

u = X[:-1]

#readParameterSweep('bs1_dns', 'gamma')
#parameterSweep('gamma', getPrange(0.7, 0.1, 0.01), initcond=u, branch='bs1_dns', **params)

#params['eta'] = X[-1]

#plt.plot(u)
#plt.plot(-u)
#plt.show()

#%%


X = np.append(-u, X[-1])
X0 = np.append(X, params['eta']) # append eta
#X0 = X
t0 = np.zeros_like(X0)
t0[-1] = 1

#advanceParam(0.2, 0.0001, X, branch='b1', auto_switch=True,  **params)
with threadpool_limits(limits=1):
    advancePALC(X0, 5e-3, t0=t0, branch='ds1_gamma=0.24', motionless=False, **params)

# etas = np.linspace(-2.0, 2.0, 1001)


# GAMMA = 0.12
# branches = (f'hss_gamma={GAMMA}', f'bs1_gamma={GAMMA}', f'pattern_gamma={GAMMA}')

# animateBifDiag(branches[2])

# params['mu'] = -0.1
# params['gamma'] = 0.12
# shr = SHRaman(**params)
# plt.plot(etas, shr.getHSS(etas=etas))
# plt.show()

# shr = SHRaman(branch=branches[0], **params)
# x = shr.loadX('x0')

# plt.plot(x[:-1], label='u')
# plt.plot(shr.coupling(np.fft.fft(x[:-1])), label='coupling')
# plt.legend()
# plt.show()

# params['mu'] = -0.1
# params['gamma'] = 0.12
# params['eta'] = x[-1]

# plotBifDiags(*branches, **params)

