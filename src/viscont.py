import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import pandas as pd

from shraman import SHRaman, params

def animateBifDiag(branch):
    shr = SHRaman(branch=branch, **params)
    
    df = pd.read_csv(shr.branchfolder + 's.csv').iloc[:100000]

    etas = df['eta']
    vs = df['v']
    fnames = df['fname']
    us = [np.load(fname)[:-2] for fname in fnames]
    Ls = [np.sum(u**2) / len(u) for u in us]

    N = len(etas)

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    ax1.plot(etas, vs)
    ax2.plot(etas, Ls)
    p1, = ax1.plot(etas[0:1], vs[0:1], 'ok')
    p2, = ax2.plot(etas[0:1], Ls[0:1], 'ok')
    lus, = ax3.plot(us[0])

    fact = 500

    def animate(j):
        i = N - fact * j - 1
        p1.set_data([etas[i]], [vs[i]])
        p2.set_data([etas[i]], [Ls[i]])
        lus.set_ydata(us[i])
        return p1, p2, lus

    ani = anim.FuncAnimation(fig, animate, frames=int(len(etas) / fact), blit=True)
    writervideo = anim.FFMpegWriter(fps=60) 
    ani.save('ds3.mp4', writer=writervideo)
    plt.show()



if __name__ == '__main__':
    animateBifDiag('ds3_palc_forw')
