import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shraman import SHRaman, params, DATADIR


def readParameterSweep(branch, pname):
    df = pd.read_csv(DATADIR + branch + '.csv')

    prange = df[pname]
    vs = df['v']
    verrs = df['verr']

    for pval in prange[::10]:
        params[pname] = pval
        sh = SHRaman(branch=branch, **params)
        u = sh.loadState(sh.getFilename(ext='.npy'))
        u = sh.center(u)

        plt.plot(u, label=f'{pname} = {pval}')
    # plt.title()
    plt.legend()
    plt.show()

    plt.errorbar(prange, vs, verrs)
    plt.show()
    print(verrs)

    plt.semilogy(prange, vs)
    plt.show()
    print(verrs)

readParameterSweep('gtest', 'gamma')