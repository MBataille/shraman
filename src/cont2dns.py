import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from threadpoolctl import threadpool_limits

def cont2dns(Equation, params, param_cont, branch, file_idxs, FACT=100, verbose=False):
    """ Runs DNS for each state in filename at file_idxs
    in order to check the stability. Then saves results 
    in filenamev2.csv

    Args:
        Equation (Equation): Can be either SHraman or LugiatoLefever
        params (dict): Dict with equation parameters
        param_cont (string): name of the parameter for the continuation
        filename (string or path): path to csv file with results from cont.
        file_idxs (array_like): list with indexes of states to simulate. 
    """
    eq = Equation(branch=branch, **params)

    summary_filename = eq.branchfolder + '/s.csv'

    df = pd.read_csv(summary_filename)

    pvals = []
    stability = []
    L2 = []
    v = []

    with alive_bar(total=len(file_idxs)) as bar:
        for file_idx in file_idxs:
            k = file_idx * FACT

            try:
                pval = df[param_cont].iloc[k]
            except IndexError:
                break
            
            eq.setParam(param_cont, pval)

            filename = eq.branchfolder + f'/x{file_idx}.npy'
            state = np.load(filename)[:params['N']]

            l2_cont = eq.L2norm(state)

            eq.setInitialCondition(state)
            with threadpool_limits(limits=1):
                eq.solve(T_f=500)

            l2_dns = eq.L2norm(eq.getState(-1))

            if abs(l2_cont - l2_dns) / l2_cont < 0.005: # stable
                stability.append('stable')
                color = 'tab:blue'
            else:
                stability.append('unstable')
                color = 'tab:orange'

            pvals.append(pval)
            L2.append(df['L2'].iloc[k])
            v.append(df['v'].iloc[k])

            # TODO  delete this later
            plt.plot([pval], [l2_cont], 'o', color=color)

            bar.text(f'For {param_cont}={pval} state is {stability[-1]}')
            bar()

    df_dns = pd.DataFrame({param_cont: pvals, 'stability': stability,
                            'L2': L2, 'v': v})
    df_dns.to_csv(eq.branchfolder + '/s_v2.csv')

    plt.show()


