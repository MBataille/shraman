import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as anim
import pandas as pd
import time
import os

from .shraman import SHRaman, params
from .reader import read_summary, read_state

GAMMA = 0.12
FACT = 100

def separateBranches_(df):
    k = 0
    branches = []
    stability = []

    while True:

        branch = df[df['branch'] == k]

        if len(branch) == 0:
            break
        
        branches.append(branch)
        
        line = '-' if branch['stability'].iloc[0] == 'stable' else '--'
        stability.append(line)

        k += 1

    return branches, stability


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

def animateBifDiag(branch, branches_ref=None, colors=None, **other_params):
    shr = SHRaman(branch=branch, **other_params)
    
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

    branches, stability = separateBranches_(df)

    for i in range(len(branches)):
        b = branches[i]
        line = stability[i]

        if colors is None:
            color = 'tab:blue'
        else:
            color = colors[0]
        ax1.plot(b['eta'], b['v'], line, color=color)
        ax2.plot(b['eta'], b['L2'], line, color=color)

    p1, = ax1.plot(etas[0:1], vs[0:1], 'ok')
    p2, = ax2.plot(etas[0:1], L2[0:1], 'ok')
    lus, = ax3.plot(us[0][:-2])
    txt = ax3.text(0.1, 0.1, '', transform=ax3.transAxes)
    ax3.set_ylim(-1.3, 1.3)

    fact = FACT


    def animate(j):
        v, eta = us[j][-2:]
        p1.set_data([eta], [v])
        p2.set_data([eta], [L2[j * fact]])
        lus.set_ydata(us[j][:-2])
        txt.set_text(f'i = {j * fact}')
        time.sleep(0.2)
        return p1, p2, lus, txt

    ax1.set_ylabel('v')
    ax1.set_xlabel('eta')
    ax2.set_ylabel('L2')
    ax2.set_xlabel('eta')
    ax3.set_ylabel('u')
    ax3.set_xlabel('x')
    #frames = 1000
    frames = int(len(etas) / fact)
    
    if branches_ref is None:
        etashss = np.linspace(-1., 1., 1001)
        u_hss = shr.getHSS(etashss)
        L2_hss = [shr.L2norm(np.array([u])) for u in u_hss]
        ax2.plot(etashss, L2_hss, color='tab:orange')
    else:
        for i, br in enumerate(branches_ref):
            if colors is None:
                color = 'tab:orange'
            else:
                color = colors[i+1]
            shr_ref = SHRaman(branch=br, **params)
            df_ref = pd.read_csv(shr_ref.branchfolder + 's.csv')
            ax2.plot(df_ref['eta'], df_ref['L2'], color=color)

    
    ani = anim.FuncAnimation(fig, animate, frames=frames, blit=True)
    #writervideo = anim.FFMpegFileWriter(fps=15) 
    #ani.save(f'{branch}a.mp4', writer=writervideo)
    #plt.clf()
    plt.show()

def plotBifDiags(*branches, points=[], legend=True, **other_params):

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    COLORS = list(mcolors.TABLEAU_COLORS)

    _params = {**params}
    for p in other_params:
        _params[p] = other_params[p]

    for k, branch in enumerate(branches):

        color = COLORS[k % len(COLORS)]

        if type(branch) != list:
            branch = (branch, )

        for j, br in enumerate(branch):

            df = read_summary(br)
            # df['L2'] = np.sqrt(df['L2'])

            curves, stab = separateBranches_(df)

            for i in range(len(curves)):
                c = curves[i]
                line = stab[i]

                if i == 0 and j == 0:
                    label = br.split('_')[0]
                else:
                    label = None

                if br[:3] != 'hss': # homogeneous solutions do not have speed
                    ax1.plot(c['eta'], c['v'], line, color=color, label=label)
                ax2.plot(c['eta'], c['L2'], line, color=color, label=label)


    # plot pts in bif diag
    for point in points:

        branch = point['branch']
        file_idx = point['file_idx']

        df = read_summary(branch).iloc[FACT * file_idx]

        ax1.plot(df['eta'], df['v'], 'ok')
        ax2.plot(df['eta'], df['L2'], 'ok')


    ax1.set_ylabel('v')
    # ax1.set_xlabel('eta')
    ax2.set_ylabel('L2*')
    ax2.set_xlabel('eta')

    if legend:
        plt.legend()

    #etashss = np.linspace(-1., 1., 1001)
    #ax2.plot(etashss, getHSS(etashss) ** 2, color='tab:orange')   

    plt.show()


    # save profile for every pt
    for point in points: 

        branch = point['branch']
        file_idx = point['file_idx']

        u = read_state(branch, f'x{file_idx}')

        if 'shift' in point:
            u = np.roll(u, point['shift'])

        if 'tag' in point: # save profile and fft
            fig_tmp, ax_tmp = plt.subplots(1)
            
            shr_tmp = SHRaman(**params)
            ax_tmp.plot(shr_tmp.tau, u)
            ax_tmp.set_ylim(-0.6, 0.6)
            
            tag = point['tag']
            fig_tmp.savefig(f'figs/{tag}.svg')
            fig_tmp.clear()

            fig_tmp, ax_tmp = plt.subplots(1)

            # Fourier transform
            N, dx = shr_tmp.getParams('N dx')
            freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
            u_ft = np.fft.fftshift(np.fft.fft(u)) * 2 / N

            ax_tmp.semilogy(freqs, np.abs(u_ft), '.')

            ax_tmp.set_ylim(1e-14, 1e0)
            
            fig_tmp.savefig(f'figs/{tag}_fft.svg')
            fig_tmp.clear()
     
GAMMA = 0.12

if __name__ == '__main__':
    #for branch in ['bs1_palc_back', 'bs1_palc_forw', 'ds3_palc_back', 'ds3_palc_forw']:

    branches = (f'hss_gamma={GAMMA}', f'bs1_gamma={GAMMA}', f'pattern_gamma={GAMMA}')

    #animateBifDiag('hss_gamma=0.12')
    #plotBifDiags('hss_gamma=0.12')
    plotBifDiags(*branches)
