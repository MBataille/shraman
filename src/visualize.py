import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shraman import SHRaman, params, DATADIR

def readParameterSweep(branch):
    df = pd.read_csv(DATADIR + branch + '.csv')

    prange = df['gamma']
    vs = df['v']
    verrs = df['verr']

    for pval in prange[::2]:
        params['gamma'] = pval
        sh = SHRaman(branch=branch, **params)
        u = sh.loadState(sh.getFilename(ext='.npy'))
        u = sh.center(u)

        plt.plot(u, label=f'Gamma = {pval}')
    # plt.title()
    plt.legend()
    plt.show()

    plt.semilogy(prange, vs)
    plt.show()
    print(verrs)

readParameterSweep('gtest')