import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shraman import SHRaman, params, DATADIR


def readParameterSweep(branch, pname):
    df = pd.read_csv(DATADIR + branch + '.csv')

    prange = df[pname]
    vs = df['v']
    verrs = df['verr']

    L2 = []
    for pval in prange:
        params[pname] = pval
        sh = SHRaman(branch=branch, **params)
        u = sh.loadState(sh.getFilename(ext='.npy'))
        L2.append(np.sum(u ** 2) / len(u))

    for pval in prange[-20:-10]:
        params[pname] = pval
        sh = SHRaman(branch=branch, **params)
        u = sh.loadState(sh.getFilename(ext='.npy'))
        u = sh.center(u)

        plt.plot(u, label=f'{pname} = {round(pval, 3)}')
    # plt.title()
    plt.legend()
    plt.show()

    plt.plot(prange, L2)
    plt.show()
    print(verrs)

    # plt.semilogy(prange, vs)
    # plt.show()
    # print(verrs)

readParameterSweep('gsimple', 'gamma')