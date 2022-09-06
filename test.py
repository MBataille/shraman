#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import (advancePALC, SHRaman, params, animateBifDiag, 
                plotBifDiags, parameterSweep, getPrange,
                readParameterSweep, read_state, read_summary, 
                readX, read_belgium, interpolate, write_state)
from src.reader import write_state
import sys

def branch_name(branch, gamma=0.24, way=''):
    if way != '':
        way = '_' + way
    return f'{branch}{way}_gamma={gamma}'

def branches_name(branches, ways, gamma=0.24):
    return [branch_name(branch, gamma=gamma, way=way)\
                 for branch, way in zip(branches, ways)]

#%%

params['eta'] = -.022
params['tau0'] = 1
params['tau1'] = 3
params['tau2'] = 10
fr = 0.18
params['gamma'] = 0.25 # np.sqrt(3) / 2 * fr
params['mu'] = -0.1  # - params['gamma']
params['alpha'] = -1.8
params['beta'] = - 4 / 3
params['dx'] = 0.5
params['N'] = 256

#animateBifDiag('bs4_large___', **params)


i = int(sys.argv[1])
direction = sys.argv[2]
if not direction.startswith(('f', 'b')):
    raise ValueError("Direction must start with 'f' or 'b'")

#animateBifDiag('bs1_large_mu', param_cont='mu', **params)
bs_branches = ([f'bs{i+1}_large_mu', f'bs{i+1}_large_mu_back'] for i in range(4))
ds_branches = ([f'ds{i+1}_large_mu', f'ds{i+1}_large_mu_back'] for i in range(4))

plotBifDiags(*bs_branches, *ds_branches, 'hss_large_mu', ['pattern_large_mu', 'pattern_large_mu_back'] , param_cont='mu', **params)

# u = read_state(f'pttrn_large_dns', f'bs1', N=256)
# v = 0.7
# #u = np.roll(u, 60)[:256]

# params['N'] = 256

# plt.plot(u)
# plt.show()

#parameterSweep('eta', getPrange(params['eta'], params['eta'] + 0.01, 0.01), initcond=u, branch=f'pttrn_large_dns', **params)

#plotBifDiags(['pttrn_belg', 'pttrn_belg_back'], 'hss_belg', *(f'bs{i+1}_belg_' for i in range(17)))

#points = [{'branch': f'bs{i+1}_large___', 'file_idx': 0} for i in range(3)]
points = [
    {'branch': 'bs1_large___', 'file_idx': 1, 'tag': 'BS1_'},
    {'branch': 'bs2_large___', 'file_idx': 30, 'tag': 'BS2_'},
    {'branch': 'bs3_large___', 'file_idx': 19, 'tag': 'BS3_'},
    {'branch': 'bs4_large___', 'file_idx': 27, 'tag': 'BS4_'},
]


bs_branches = (f'bs{i+1}_large_cropped' for i in range(4))
ds_branches = (f'ds{i+1}_large_cropped' for i in range(4))
plotBifDiags(['pttrn_large_cropped', 'pttrn_large_back'], 'hss_large___', *bs_branches, *ds_branches)

# params['mu'] = -1.0
# shr = SHRaman(**params)
# X_hss = shr.getHSS() + np.zeros(params['N'])

# X0 = np.append(X_hss, params['mu'])

# X = np.append(u, v)

#X = readX('pttrn_large_back', 'x0')
X = readX(f'bs{i}_large___', f'x0')
X[:256] = - X[:256]
X[-1] = - X[-1]
params['eta'] = X[-1]
X[-1] = params['mu']
#X0 = np.append(X, params['eta']) # append eta
X0 = X

u = X0[:-2]
plt.plot(u)
plt.show()


t0 = np.zeros_like(X0)
suffix = ''
if direction.startswith('f'):
    t0[-1] = 1
else:
    t0[-1] = -1
    suffix = '_back'


#advanceParam(0.2, 0.0001, X, branch='b1', auto_switch=True,  **params)

with threadpool_limits(limits=1):
    advancePALC(X0, 5e-3, t0=t0, branch=f'ds{i}_large_mu{suffix}', motionless=False, param_cont='mu',**params)


#%%
fname = 'data/belgium/1p/solution-one-pic.mat'
mat = read_belgium(fname)
u = mat['u'].real.flatten()
tau = np.linspace(0, 1028 * 0.55, 1028, endpoint=False)

print(len(tau), len(u))

dtau_new = 0.5

N_new = int(1028 * 0.55 / dtau_new)
tau_new = np.linspace(0, N_new * dtau_new, N_new, endpoint=False)

print(N_new)

u_new = interpolate(tau, u, N_new)
u_new = np.roll(u_new, -200)

u_new = u_new[:512]
tau_new = tau_new[:512]

plt.plot(tau_new,u_new)
plt.show()

write_state(u_new, 'subcritical', 'bs1')

#%%
I = 7

params = {'N': 512, 'dx': 0.5, 'tau0': 1, 'tau1': 3, 'tau2': 10, # 18, 12, 32
          'eta': 0, 'mu': -0.1, 'alpha': -1.0, 'beta': -1.0,
          'gamma': 0.7}

# parameterSweep('gamma', getPrange(0.7, 0.24, 0.01), initcond=read_state(f'bs{I}_tmp', 'bs7'),
#                branch=f'bs{I}_dns', **params)

_branches = ['periodic'] * 2 
_ways = ['', 'back'] 

periodic_branches = branches_name(_branches, _ways)

_branches =  ['hss'] + [f'bs{i+1}' for i in range(6)]
_ways = [''] * len(_branches)

branches = [periodic_branches] + branches_name(_branches, _ways)
print(branches[2])

# #%%

points = [
    {'branch': branches[2], 'file_idx': 0, 'tag': 'B_1'},
    {'branch': branches[3], 'file_idx': 5, 'tag': 'B_2'},
    {'branch': branches[4], 'file_idx': 28, 'tag': 'B_3', 'shift': -50},
    {'branch': branches[5], 'file_idx': 5, 'tag': 'B_4', 'shift': 200},
    {'branch': branches[6], 'file_idx': 8, 'tag': 'B_5', 'shift': 150},
    {'branch': branches[7], 'file_idx': 3, 'tag': 'B_6', 'shift': 100}
]

#animateBifDiag(branches[7], branches_ref=(*branches[0], branches[1]), colors=['tab:red', 'tab:blue', 'tab:blue', 'tab:orange'], **params)
plotBifDiags(*branches, legend=False)

#%%

params['gamma'] = 0.24
params['eta'] = 0.0

#animateBifDiag(f'bs{I}_gamma=0.24', branches_ref=('hss_gamma=0.24', 'periodic_gamma=0.24'), colors=['tab:blue', 'tab:green', 'tab:orange'], **params)
#plotBifDiags(('periodic_gamma=0.24','periodic_back_gamma=0.24' ),'bs1_gamma=0.24', 'hss_gamma=0.24')



# X = readX(f'bs{I}_dns', 'bs1')
# X0 = np.append(X, params['eta']) # append eta



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


# %%
