import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as anim
import seaborn as sns
import pandas as pd
import time
import os

from .shraman import SHRaman, params
from .reader import read_summary, read_state

GAMMA = 0.12
FACT = 100

## plot params ##

sns.set_palette('viridis')
# Configuración de los gráficos
fsize = 15
tsize = 16
tdir = 'in'
major = 5.0
minor = 3.0
lwidth = 0.7
lhandle = 2.0
plt.style.use('seaborn-deep')
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = fsize
plt.rcParams['legend.fontsize'] = tsize-4
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 3.0
plt.rcParams['axes.linewidth'] = lwidth
plt.rcParams['legend.handlelength'] = lhandle
plt.rcParams['axes.labelsize'] = tsize + 4



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

def animateBifDiag(branch, branches_ref=None, colors=None, param_cont='eta', **other_params):
    shr = SHRaman(branch=branch, **other_params)
    
    df = pd.read_csv(shr.branchfolder + 's.csv').iloc[:100000]


    #print(np.asarray(df.iloc[0]['u']) )

    pconts = df[param_cont]
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

    N = len(pconts)

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
        ax1.plot(b[param_cont], b['v'], line, color=color)
        ax2.plot(b[param_cont], b['L2'], line, color=color)

    p1, = ax1.plot(pconts[0:1], vs[0:1], 'ok')
    p2, = ax2.plot(pconts[0:1], L2[0:1], 'ok')
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
    ax1.set_xlabel(param_cont)
    ax2.set_ylabel('L2')
    ax2.set_xlabel(param_cont)
    ax3.set_ylabel('u')
    ax3.set_xlabel('x')
    #frames = 1000
    frames = int(len(pconts) / fact)
    
    if branches_ref is not None:
        for i, br in enumerate(branches_ref):
            if colors is None:
                color = 'tab:orange'
            else:
                color = colors[i+1]
            shr_ref = SHRaman(branch=br, **params)
            df_ref = pd.read_csv(shr_ref.branchfolder + 's.csv')
            ax2.plot(df_ref[param_cont], df_ref['L2'], color=color)

    
    ani = anim.FuncAnimation(fig, animate, frames=frames, blit=True)
    #writervideo = anim.FFMpegFileWriter(fps=15) 
    #ani.save(f'{branch}a.mp4', writer=writervideo)
    #plt.clf()
    plt.show()

def plotBifDiags(*branches, points=[], legend=True, param_cont='eta', COLORS=None, 
    figsize=None, vertical_lines=None , lims=None, ticks=None, 
    plot_points=True, save_points=True, **other_params):

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=figsize)
    
    if lims is not None:
        xlims, ylims = lims
        ax1.set_xlim(xlims)
        ax1.set_ylim(ylims)

    if ticks is not None:
        xticks, yticks = ticks
        ax1.set_xticks(xticks)
        ax1.set_yticks(yticks)

    if COLORS is None:
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
                    ax2.plot(c[param_cont], c['v'], line, color=color, label=label)
                ax1.plot(c[param_cont], c['L2'], line, color=color, label=label)
                if legend:
                    plt.legend()

    if vertical_lines is not None:
        ax1.vlines(vertical_lines['x'], vertical_lines['ymin'], vertical_lines['ymax'],
                    colors=vertical_lines['colors'], linestyles=vertical_lines['linestyles'])

    # plot pts in bif diag
    if plot_points:
        for point in points:

            branch = point['branch']
            file_idx = point['file_idx']

            df = read_summary(branch).iloc[FACT * file_idx]

            ax2.plot(df[param_cont], df['v'], 'ok')
            ax1.plot(df[param_cont], df['L2'], 'ok')


    ax2.set_ylabel(r'$v$')
    # ax1.set_xlabel(param_cont)
    ax1.set_ylabel(r'$L^{2*}$')
    if param_cont == 'eta':
        ax2.set_xlabel(r'$\eta$')
    else:
        ax2.set_xlabel(param_cont)

    #etashss = np.linspace(-1., 1., 1001)
    #ax2.plot(etashss, getHSS(etashss) ** 2, color='tab:orange')   

    plt.show()


    # save profile for every pt
    if save_points:
        for point in points: 

            branch = point['branch']
            file_idx = point['file_idx']
            color = point['color']

            u = read_state(branch, f'x{file_idx}')

            if 'shift' in point:
                u = np.roll(u, point['shift'])

            if 'tag' in point: # save profile and fft
                fig_tmp, ax_tmp = plt.subplots(1)
                
                shr_tmp = SHRaman(**params)
                u = shr_tmp.center(u)
                ax_tmp.plot(shr_tmp.tau, u, linewidth=3, color=color)

                ax_tmp.tick_params(axis='y', length=8, width=2)

                ax_tmp.set_ylim(-0.7, 0.7)
                ax_tmp.set_yticks([-0.6, 0.6])

                xlims = [0, 128]
                ax_tmp.set_xlim(*xlims)
                ax_tmp.set_xticks(xlims)
                #ax_tmp.set_xticklabels([0, 128])

                ax_tmp.set_xlabel(r'$\tau$')
                ax_tmp.set_ylabel(r'$u$')

                tag = point['tag']
                fig_tmp.savefig(f'figs/{tag}.svg', bbox_inches='tight')
                fig_tmp.clear()

                _u = u
                #for f in range(10):
                fig_tmp, ax_tmp = plt.subplots(1)

                # Fourier transform
                N, dx = shr_tmp.getParams('N dx')
                freqs = np.fft.fftshift(np.fft.fftfreq(len(_u), d=dx)) * 2 * np.pi
                u_ft = np.fft.fftshift(np.fft.fft(_u)) * 2 / N

                ax_tmp.semilogy(freqs, np.abs(u_ft), '.')

                ax_tmp.set_ylim(1e-14, 1e0)
                
                fig_tmp.savefig(f'figs/{tag}_fft.svg')
                fig_tmp.clear()
                plt.close(fig_tmp)

                #_u = np.append(_u, u)
     
GAMMA = 0.12

if __name__ == '__main__':
    #for branch in ['bs1_palc_back', 'bs1_palc_forw', 'ds3_palc_back', 'ds3_palc_forw']:

    branches = (f'hss_gamma={GAMMA}', f'bs1_gamma={GAMMA}', f'pattern_gamma={GAMMA}')

    #animateBifDiag('hss_gamma=0.12')
    #plotBifDiags('hss_gamma=0.12')
    plotBifDiags(*branches)
