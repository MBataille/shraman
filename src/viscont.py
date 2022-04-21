import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import pandas as pd
import os

from shraman import SHRaman, params

def animateBifDiag(branch):
    shr = SHRaman(branch=branch, **params)
    
    df = pd.read_csv(shr.branchfolder + 's.csv')#.iloc[:100000]

    #print(np.asarray(df.iloc[0]['u']) )

    etas = df['eta']
    vs = df['v']
    L2 = df['L2']
    us = []
    i = 0
    while True:
        fname = shr.branchfolder + f'x{i}.npy'
        if not os.path.exists(fname): break
        us.append(np.load(fname))
        i += 1
    # fnames = df['fname']
    #us = np.array([np.load(fname)[:-2] for fname in fnames])
    #np.save(shr.branchfolder + 's.npy', us)
    #return
    # Ls = 

    N = len(etas)

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    # ax1.plot(etas, vs)
    # ax2.plot(etas, L2)

    ax1.plot(etas, vs, linewidth=0.5)
    ax2.plot(etas, L2, linewidth=0.5)
    p1, = ax1.plot(etas[0:1], vs[0:1], 'ok')
    p2, = ax2.plot(etas[0:1], L2[0:1], 'ok')
    lus, = ax3.plot(us[0][:-2])
    txt = ax3.text(0.1, 0.1, '', transform=ax3.transAxes)

    fact = 100

    def animate(j):
        v, eta = us[j][-2:]
        p1.set_data([eta], [v])
        p2.set_data([eta], [L2[j * fact]])
        lus.set_ydata(us[j][:-2])
        txt.set_text(f'i = {j * fact}')
        return p1, p2, lus, txt

    ani = anim.FuncAnimation(fig, animate, frames=int(len(etas) / fact), blit=True)
    # writervideo = anim.FFMpegWriter(fps=20) 
    # ani.save(f'{branch}.mp4', writer=writervideo)
    #plt.clf()
    plt.show()



if __name__ == '__main__':
    #for branch in ['bs1_palc_back', 'bs1_palc_forw', 'ds3_palc_back', 'ds3_palc_forw']:
    animateBifDiag('ds3_palc_back_')
