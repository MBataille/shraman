import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as anim
import pandas as pd
import os

from shraman import SHRaman, params

GAMMA = 0.12

def separateBranches(df):
    etas = df['eta']
    L2 = df['L2']
    tau = df['tau']

    cs_pts = df[tau * np.roll(tau, 1) < 0].index

    print(cs_pts)

    if len(cs_pts) == 0:
        line = '-' if tau.iloc[1] < 0 else '--'
        return [df], [line]


    branches = []
    stability = []
    line = ''

    for i in range(len(cs_pts) + 1):
        start = 0 if i == 0 else cs_pts[i-1]
        end = -1 if i == len(cs_pts) else cs_pts[i]
        branches.append(df.iloc[start:end])
        if tau.iloc[start+1] < 0: # stable
            line = '-'
        else:
            line = '--'
        stability.append(line)

    print(stability)
    return branches, stability


def getHSS(etas):
    us = np.zeros_like(etas)

    mu = -0.1
    gamma = GAMMA

    for i, eta in enumerate(etas):
        roots = np.roots([-1, 0, mu, eta + gamma])
        for root in roots:
            if root.imag == 0:
                us[i] = root.real
                break
    
    return us

def animateBifDiag(branch):
    shr = SHRaman(branch=branch, **params)
    
    df = pd.read_csv(shr.branchfolder + 's.csv').iloc[:100000]


    #print(np.asarray(df.iloc[0]['u']) )

    etas = df['eta']
    vs = df['v']
    #df['L2'] = np.sqrt(df['L2']) 

    L2 = df['L2']
    us = []
    i = 0
    while True:
        fname = shr.branchfolder + f'x{i}.npy'
        if not os.path.exists(fname): break
        u = np.load(fname)
        #us.append(np.append(shr.center(u[:-2]), u[-2:]))
        us.append(u)
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

    branches, stability = separateBranches(df)

    for i in range(len(branches)):
        b = branches[i]
        line = stability[i]

        ax1.plot(b['eta'], b['v'], line, color='tab:blue')
        ax2.plot(b['eta'], b['L2'], line, color='tab:blue')

    p1, = ax1.plot(etas[0:1], vs[0:1], 'ok')
    p2, = ax2.plot(etas[0:1], L2[0:1], 'ok')
    lus, = ax3.plot(us[0][:-2])
    txt = ax3.text(0.1, 0.1, '', transform=ax3.transAxes)
    ax3.set_ylim(-1.3, 1.3)

    fact = 100


    def animate(j):
        v, eta = us[j][-2:]
        p1.set_data([eta], [v])
        p2.set_data([eta], [L2[j * fact]])
        lus.set_ydata(us[j][:-2])
        txt.set_text(f'i = {j * fact}')
        return p1, p2, lus, txt

    ax1.set_ylabel('v')
    ax1.set_xlabel('eta')
    ax2.set_ylabel('L2')
    ax2.set_xlabel('eta')
    ax3.set_ylabel('u')
    ax3.set_xlabel('x')
    #frames = 1000
    frames = int(len(etas) / fact)
    
    etashss = np.linspace(-1., 1., 1001)
    ax2.plot(etashss, getHSS(etashss) ** 2, color='tab:orange')

    
    ani = anim.FuncAnimation(fig, animate, frames=frames, blit=True)
    #writervideo = anim.FFMpegFileWriter(fps=15) 
    #ani.save(f'{branch}a.mp4', writer=writervideo)
    #plt.clf()
    plt.show()

def plotBifDiags(*branches):

    fig, (ax1, ax2) = plt.subplots(2)

    COLORS = list(mcolors.TABLEAU_COLORS)

    for k, branch in enumerate(branches):
        shr = SHRaman(branch=branch, **params)
        df = pd.read_csv(shr.branchfolder + 's.csv')
        # df['L2'] = np.sqrt(df['L2'])

        curves, stab = separateBranches(df)

        for i in range(len(curves)):
            c = curves[i]
            line = stab[i]

            if i == 0:
                label = branch.split('_')[0]
            else:
                label = None

            ax1.plot(c['eta'], c['v'], line, color=COLORS[k], label=label)
            ax2.plot(c['eta'], c['L2'], line, color=COLORS[k], label=label)

    ax1.set_ylabel('v')
    ax1.set_xlabel('eta')
    ax2.set_ylabel('L2')
    ax2.set_xlabel('eta')

    plt.legend()

    #etashss = np.linspace(-1., 1., 1001)
    #ax2.plot(etashss, getHSS(etashss) ** 2, color='tab:orange')   

    plt.show()
     
GAMMA = 0.12

if __name__ == '__main__':
    #for branch in ['bs1_palc_back', 'bs1_palc_forw', 'ds3_palc_back', 'ds3_palc_forw']:

    branches = (f'hss_gamma={GAMMA}', f'bs1_gamma={GAMMA}', f'pattern_gamma={GAMMA}')

    #animateBifDiag('hss_gamma=0.12')
    #plotBifDiags('hss_gamma=0.12')
    plotBifDiags(*branches)
