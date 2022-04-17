from enum import auto
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from alive_progress import alive_bar

from shraman import SHRaman, params
from bifdiag import getPrange


def newton(X0, func, jac, max_steps=100, tol=1e-6):

    success = False
    for step in range(max_steps):

        if step == 0:
            Y = func(X0)
            err = np.abs(Y).sum()
            # print(f'. Initial error is {err}')

        J = jac(X0)

        with threadpool_limits(limits=1):
            dX = np.linalg.solve(J, -Y)

        X0 += dX

        Y = func(X0)
        err = np.abs(Y).sum()
        # print(f'.. Step {step}: error = {err}, v = {X0[-1]}')

        if err < tol:
            # print(f'. Error is smaller than threshold. Finishing Newtons method')
            success = True
            break

    return X0, success

def advanceParam(p0, dp, X0, switch=False, auto_switch=False, stopAt=None, **other_params):

    params = {**other_params}

    fnames = []
    vs = []
    etas = []

    if switch:
        params['v'] = p0
    else:
        params['eta'] = p0

    shr = SHRaman(**params)
            
    if switch:
        eta = shr.switch_on(p0)

    with alive_bar() as bar:

        while True:

            if len(etas) > 0:
                bar.text(f'eta = {etas[-1]}, v = {vs[-1]}')
            shr.init_cont(X0[:-1])
            X0, success = newton(X0, shr.cont_rhs, shr.jacobian)

            if stopAt is not None and len(etas) > 0:
                if eval(str(etas[-1]) + stopAt): break

            if not success: # change switch

                if not auto_switch:
                    break

                ####
                if len(vs) >= 2:
                    if shr.switch:
                        slope = etas[-1] - etas[-2]
                    else:
                        slope = vs[-1] - vs[-2]
                else:
                    print('SOSOSOSOS, dont know slope!!!')
                
                p0 = etas[-1] if shr.switch else vs[-1]  
                dp = slope / 10
                X0 = shr.change_switch(X0)
                
                old_pname = 'eta' if shr.switch else 'v'
                old_pval = etas[-1] if shr.switch else vs[-1]
                pname = 'v' if shr.switch else 'eta'
                print(f'Switch changed! \nOcurred at {old_pname}={old_pval}.\nMoving now along {pname}.\np0 = {p0}, dp = {dp}')

                continue

            fname = shr.saveX(X0, filename=f'x{len(etas)}')

            if shr.switch:
                v = p0
                eta = X0[-1]
            else:
                v = X0[-1]
                eta = p0

            vs.append(v)
            etas.append(eta)
            fnames.append(fname)

            p0 += dp

            if shr.switch:
                shr.setParam('v', p0)
            else:
                shr.setParam('eta', p0)

            df = pd.DataFrame({'eta': etas, 'v': vs, 'fname': fnames})
            df.to_csv(shr.branchfolder + f's.csv')

            bar()
        
    plt.plot(etas, vs)
    plt.show()

def parameterSweep(pname, prange, X0, **other_params):
    
    params = {**other_params}

    fnames = []
    vs = np.zeros_like(prange)

    # print(f'Sweeping {pname} from {prange[0]} to {prange[-1]}.')

    plt.plot(X0[:-1], label=f'{pname} = {prange[0]}')

    with alive_bar(len(prange)) as bar:
        for i, pval in enumerate(prange):
            params[pname] = pval
            shr = SHRaman(**params)

            shr.init_cont(X0[:-1])
            X0, success = newton(X0, shr.cont_rhs, shr.jacobian)



            fname = shr.getFilename(X='X_', ext='.npy')
            shr.saveX(X0, '')

            vs[i] = X0[-1]
            fnames.append(fname)

            plt.plot(X0[:-1], label=f'{pname} = {pval}')

            bar()

        # print(f' Iteration {i}, {pname} = {pval}, v = {vs[i]}.')

    plt.legend()
    plt.show()

    data = {pname: prange, 'v': vs, 'fname': fnames}
    df = pd.DataFrame(data)
    df.to_csv(shr.branchfolder + f'cont_{pname}.csv')

    return vs

if __name__ == '__main__':
    params['eta'] = 0.2001
    shr = SHRaman(branch = 'actest', **params)

    #u0 = shr.loadState(shr.getFilename(ext='.npy'))
    #v = 3.626771296520384

    X = shr.loadX()
    print(X[-1])
    
    advanceParam(0.2, 0.0001, X, branch='b1', auto_switch=True,  **params)
    # etas = getPrange(0, 0.3, 0.02)
    # vs = parameterSweep('eta', etas, X, branch='ptest', **params)



    # plt.plot(etas, vs)
    # plt.show()