import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla
import pandas as pd
from threadpoolctl import threadpool_limits
from alive_progress import alive_bar

import numdifftools as nd

from .shraman import SHRaman, params
from .bifdiag import getPrange


def newton(X0, func, jac, max_steps=50, tol=1e-6, test_function=None):

    success = False
    tau = 0

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
            
            if test_function is not None:
                tau = test_function(J)
            break

    return X0, success, tau

def test_func_stability(jac):
    # assuming PALC for SH-raman, one
    # needs to discard last 2 rows and cols
    #return spla.eigs(jac[:-2, :-2], k=1, which='LR')[0]
    w = np.linalg.eigvals(jac[:-2, :-2])
    return np.amax(w.real)

def advancePALC(X0, ds, t0=None, motionless=False, **other_params):

    _params = {**other_params}
    N = _params['N']

    # fnames = []
    # tnames = []

    vs = []
    etas = []
    L2 = []
    taus = []
    branches = []
    stability = []

    shr = SHRaman(**_params)
    if motionless:
        shr.motionless = True

    # estimate tangent?

    SAVE_EVERY = 1000
    SAVE_FILE_EVERY = 100

    output_file = shr.branchfolder  + 's.csv'
    iter_count = 0
    last_save = 0
    file_count = 0
    branch_count = 0

    with alive_bar() as bar:
        while True:

            if len(etas) > 0:
                bar.text(f'eta = {etas[-1]}, L2 = {L2[-1]}, tau = {taus[-1]}')
            
            if motionless:
                shr.init_cont(X0[:-1])
            else:
                shr.init_cont(X0[:-2])

            t0 = shr.get_tangent(X0, prev_tangent=t0)
            shr.init_palc(ds, t0, X0)

            X0, success, tau = newton(X0 + ds * t0, shr.palc_rhs, shr.jacobian_palc, test_function=test_func_stability)

            if iter_count == 0:
                plt.plot(X0[:-2])
                plt.show()

            if not success:
                break

            if iter_count % SAVE_FILE_EVERY == 0:
                shr.saveX(X0, filename=f'x{file_count}')
                file_count += 1

            L2.append(shr.L2norm(X0[:N]))
            vs.append(X0[-2])
            etas.append(X0[-1])
            taus.append(tau)
            stability.append('stable' if tau < 0 else 'unstable')
            branches.append(branch_count)

            if len(taus) > 1 and taus[-1] * taus[-2] < 0:
                print(f'Bifurcation point (stability change) between eta = {etas[-1]} and {etas[-2]}')
                branch_count += 1

            iter_count += 1

            bar()

            if iter_count % SAVE_EVERY == 0:
                df = pd.DataFrame({'eta': etas[last_save:], 'v': vs[last_save:], 'L2': L2[last_save:], 'tau': taus[last_save:], 'stability': stability[last_save:], 'branch': branches[last_save:]})
                df.to_csv(shr.branchfolder + f's.csv', mode='a', header=(last_save == 0), index=False)
                last_save = iter_count

    df = pd.DataFrame({'eta': etas[last_save:], 'v': vs[last_save:], 'L2': L2[last_save:], 'tau': taus[last_save:], 'stability': stability[last_save:], 'branch': branches[last_save:]})
    df.to_csv(shr.branchfolder + f's.csv', mode='a', header=(last_save==0), index=False)

    plt.plot(etas, vs)
    plt.show()

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
            X0, success = newton(X0, shr.cont_rhs, shr.jacobian, test_function=test_func_stability)

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

    params['gamma'] = 0.12
    params['eta'] = -2.0
    shr = SHRaman(branch = 'hss_gamma=0.12', **params)

    X = np.zeros(params['N']) +  shr.getHSS()

    # X = shr.loadX('x0')
    
    plt.plot(X)
    # plt.show()

    X0 = np.append(X, params['eta']) # append eta
    # X0 = X
    t0 = np.zeros_like(X0)
    t0[-1] = 1
    
    #advanceParam(0.2, 0.0001, X, branch='b1', auto_switch=True,  **params)
    with threadpool_limits(limits=1):
        advancePALC(X0, 1e-2, t0=t0, branch='hss_gamma=0.12', motionless=True, **params)
    # etas = getPrange(0, 0.3, 0.02)
    # vs = parameterSweep('eta', etas, X, branch='ptest', **params)



    # plt.plot(etas, vs)
    # plt.show()