import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import pandas as pd

from shraman import SHRaman, params

def animateBifDiag(branch):
    shr = SHRaman(branch=branch, **params)
    
    df = pd.read_csv(shr.branchfolder + 's.csv')

    etas = df['eta']
    vs = df['v']
    fnames = df['fname']
    us = [np.load(fname)[:-2] for fname in fnames]

    N = len(etas)

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(etas, vs)
    p, = ax1.plot(etas[0:1], vs[0:1], 'ok')
    lus, = ax2.plot(us[0])

    fact = 10

    def animate(j):
        i = N - fact * j - 1
        p.set_data([etas[i]], [vs[i]])
        lus.set_ydata(us[i])
        return p, lus

    ani = anim.FuncAnimation(fig, animate, frames=int(len(etas) / fact), blit=True)
    plt.show()



if __name__ == '__main__':
    animateBifDiag('ds3_palc')
