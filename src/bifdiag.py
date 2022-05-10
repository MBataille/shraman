import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from alive_progress import alive_bar

from .shraman import SHRaman, DATADIR

import sys

def parameterSweep(param_name, param_range, initcond=None, **other_params):

    T_transient = 200
    Tf = 400

    params = {**other_params}

    params[param_name] = param_range[0]
    branch = other_params['branch']

    shr = SHRaman(**params)

    if initcond is None:
        shr.setInitialConditionGaussian()
        shr.solve()

        last_state = shr.getState(-1)
    else:
        last_state = initcond

    plt.plot(last_state)
    plt.show()

    vs = np.zeros_like(param_range)
    verrs = np.zeros_like(param_range)
    
    with alive_bar(len(param_range)) as bar:
        for i, param_val in enumerate(param_range):
            params = {**other_params}
            params[param_name] = param_val
            shr = SHRaman(**params)

            shr.setInitialCondition(last_state)
            shr.solve(T_transient=T_transient, T_f=Tf)
            shr.saveState()

            last_state = shr.getState(-1)

            vs[i], verrs[i] = shr.getVelocity()

            if i == 0:
                plt.plot(last_state)
                plt.show()
                shr.saveX(np.append(last_state, vs[i]), 'bs1')
            
            # print(f'Velocity for {param_name} = {param_val} is {vs[i]}')
            bar()

    data = {param_name: param_range, 'v': vs, 'verr': verrs}
    df = pd.DataFrame(data)
    df.to_csv(DATADIR + branch + '.csv')

######
def getPrange(p0, pf, dp):
    Np = int(abs(pf - p0) / dp) + 1
    if p0 > pf:
        return np.linspace(pf, p0, Np)[::-1]
    return np.linspace(p0, pf, Np)

if __name__ == '__main__':

    branch = 'gsimple2'
    pname = 'gamma'
    p0 = 0.2
    pf =  0.01
    dp = 0.001
    # branch = sys.argv[1]
    # pname = sys.argv[2]
    # p0 = float(sys.argv[3])
    # pf = float(sys.argv[4])
    # dp = float(sys.argv[5])

    params = {'N': 512, 'dx': 0.5, 'tau1': 3, 'tau2': 10,
          'eta': 0, 'mu': -0.1, 'alpha': -1.0, 'beta': -1.0,
          'gamma': 0.7}

    prange = getPrange(p0, pf, dp)

    params['gamma'] = p0
    sh = SHRaman(branch='gsimple', **params)
    u0 = sh.loadState(sh.getFilename(ext='.npy'))

    print(f'Parameter sweep of {pname} from {prange[0]} to {prange[-1]} with {len(prange)} points.')
    parameterSweep(pname, prange, initcond=u0, branch=branch, **params)

