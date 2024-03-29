import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
#from pathlib import Path

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATADIR = os.path.join(BASEDIR, 'data', '')

def ptoname(eta, mu, gamma):
    return f'eta={round(eta, 5)}_mu={round(mu, 5)}_gamma={round(gamma, 5)}'

class SHRaman:
    def __init__(self, **params):
        self.p = params
        
        # Defining useful parameters
        self.p['L']= self.p['N'] * self.p['dx']

        # we will consider tau0 = 1 and as such
        # we must rescale tau1 and tau2 accordingly.
        if 'tau0' in self.p:
            self.p['tau1'] = self.p['tau1'] / self.p['tau0']
            self.p['tau2'] = self.p['tau2'] / self.p['tau0']
        
        self.p['a'] = (self.p['tau1'] ** 2 + self.p['tau2'] ** 2) \
                    / (self.p['tau1'] * self.p['tau2'] ** 2)

        self.param_cont = 'eta'

        if 'branch' in params:
            self.branch = params['branch']
            self.branchfolder = os.path.join(DATADIR, self.branch, '')
            if not os.path.exists(self.branchfolder):
                if not os.path.exists(DATADIR):
                    os.mkdir(DATADIR)
                os.mkdir(self.branchfolder)

        # Initialize FFT of Coupling kernel
        N, dx = self.getParams('N dx')
        self.tau = np.linspace(0, N*dx, N, endpoint=False)
        self.kernel_ft = np.fft.fft(self.kernel())

        # For continuation
        self.switch = False
        self.motionless = False

    def set_param_cont(self, param_cont):
        if not param_cont in ['eta', 'mu']:
            raise ValueError("Parameter for Continuation must be either 'eta' or 'mu'")
        self.param_cont = param_cont     
    
    def kernel(self):
        tau1, tau2, a = self.getParams('tau1 tau2 a')
        return a * np.exp(-self.tau / tau2) * np.sin(self.tau / tau1)

    def coupling(self, u_ft):
        return np.fft.ifft(u_ft * self.kernel_ft).real * self.p['dx']

    def get_ik(self):
        return 2j * np.pi / self.p['L'] * np.fft.fftfreq(self.p['N'])\
             * self.p['N'] 

    def spectral_deriv(self, u_ft, order):
        ik = self.get_ik() ** order
        return np.fft.ifft(ik * u_ft ).real

    def L2norm(self, u, ref=2):
        if type(u) == float:
            u = np.array([u])
        return np.sqrt(np.sum((u + ref) ** 2)) / len(u)

    def rhs_shraman(self, t, u):
        eta, mu, alpha, beta, gamma = self.getParams('eta mu alpha beta gamma')

        u_ft = np.fft.fft(u)
        d2u = self.spectral_deriv(u_ft, 2)
        d4u = self.spectral_deriv(u_ft, 4)
        
        return eta + mu * u - u ** 3 + alpha * d2u + beta * d4u + gamma * self.coupling(u_ft)


    def setInitialConditionGaussian(self):
        L = self.p['L']
        self.u0 = np.exp(- (self.tau - L/4)**2 / (L / 50)) - 0.2

    def get_single_HSS(self, mu, eta, gamma, I=1):
        for root in np.roots([-1, 0, mu + gamma * I, eta]):
            if root.imag == 0:
                return root.real

    def getHSS(self, params=None):
        mu, eta, gamma = self.getParams('mu eta gamma')

        I = np.trapz(self.kernel(), x=self.tau)
        print(f'Integral is {I}')

        if params is None:
            return self.get_single_HSS(mu, eta, gamma)
        else:
            if self.param_cont == 'eta':
                return np.array([self.get_single_HSS(mu, eta, gamma, I=I) for eta in params])
            elif self.param_cont == 'mu':
                return np.array([self.get_single_HSS(mu, eta, gamma, I=I) for mu in params])


    def solve(self, T_transient=0, T_f=100):
        Tf = T_f
        dt = 0.1

        if T_transient > 0:
            t_span_trans = (0., T_transient)
            self.u0 = solve_ivp(self.rhs_shraman, t_span_trans, self.u0, t_eval=[T_transient], max_step=0.01).y[:, 0]

        t_span = (T_transient, Tf)
        t_eval = np.linspace(*t_span, int((Tf - T_transient) / dt) + 1)

        self.t = t_eval - T_transient # so that it starts at 0
        self.sol = solve_ivp(self.rhs_shraman, t_span, self.u0, t_eval=t_eval, max_step=0.01).y

    def getState(self, k):
        return self.sol[:, k]

    def saveState(self, k = -1):
        np.save(self.getFilename(), self.getState(k))

    def saveX(self, X, filename=None):
        if filename is None:
            filename = self.getFilename(X='X_')
        else:
            filename = self.branchfolder + filename + '.npy'
        np.save(filename, X)
        return filename

    def loadX(self, filename=None):
        if filename is None:
            filename = self.getFilename(X='X_', ext='.npy')
        else:
            filename = self.branchfolder + filename + '.npy'
        return np.load(filename)

    def loadState(self, filename):
        return np.load(filename)

    def loadInitialCondition(self, filename):
        self.u0 = self.loadState(filename)

    def setInitialCondition(self, u0):
        self.u0 = u0

    def setInitialConditionGaussian(self):
        L = self.p['L']
        #self.u0 = np.exp(- (self.tau - L/2)**2 / (L / 5)) * np.sin(self.tau / 8)
        self.u0 =  np.exp(-((self.tau - L / 2) / 12) ** 2) + np.exp(-((self.tau - 3 * L / 8) / 12) ** 2) - 0.2

    def setInitialConditionPattern(self, wavelength=None):
        if wavelength is None: wavelength = self.p['L'] / 6
        k = 2 * np.pi / wavelength
        self.u0 = np.sin(k * self.tau)

    def center(self, u):
        xpos = self.getXpos(u, index=True)
        return np.roll(u, int(len(u)/2) - xpos)

    def getXpos(self, u, find='max', index=False):
        conv = self.coupling(np.fft.fft(u))

        func = np.argmax if find == 'max' else np.argmin

        i_pos = func(conv)

        if index:
            return i_pos
        return self.tau[i_pos]

    def getVelocity(self):
        self.xs = np.zeros_like(self.t)
        
        jump = 0
        for k in range(len(self.t)):
            self.xs[k] = self.getXpos(self.getState(k))
            if k > 0:
                diff = self.xs[k-1] - jump - self.xs[k]
                if abs(diff) > self.p['L'] / 2:
                    jump += self.p['L'] * diff / abs(diff)
            self.xs[k] += jump

        p, cov = np.polyfit(self.t, self.xs, 1, cov=True)
        v = p[0]
        verr = np.sqrt(cov[0, 0])

        return v, verr

    def getFilename(self, pre=None, ext='', X=''):
        eta, mu, gamma = self.p['eta'], self.p['mu'], self.p['gamma']
        if pre is None: pre = self.branchfolder
        return pre + X + ptoname(eta, mu, gamma) + ext

    def init_cont(self, u0):
        # for pinning condition
        self.du0dx = self.spectral_deriv(np.fft.fft(u0), 1)

        # for jac of spectral deriv
        ik = self.get_ik()

        # jacobian of spatial derivatives
        ik_ift = np.fft.ifft(-ik).real
        self.deriv_du_dx = np.roll(np.append(ik_ift, ik_ift), -1)

        k2 = np.fft.ifft(ik ** 2).real
        self.deriv_d2u_dx2 = np.roll(np.append(k2, k2), -1)

        k4 = np.fft.ifft(ik ** 4).real
        self.deriv_d4u_dx4 = np.roll(np.append(k4, k4), -1)

        # for jac of coupling
        self.deriv_coupling = np.roll(self.kernel()[::-1], 1) * self.getParam('dx')
        
    def setParam(self, pname, pval):
        self.p[pname] = pval

    def change_switch(self, X):
        if self.switch: # now v is a param and eta is X[-1]
            X[-1] = self.switch_off(X[-1])
        else: # now eta is a param and v is X[-1]
            X[-1] = self.switch_on(X[-1])

        return X

    def switch_on(self, v):
        self.switch = True
        self.setParam('v', v)
        return self.getParam('eta')

    def switch_off(self, eta):
        self.switch = False
        self.setParam('eta', eta)
        return self.getParam('v')

    def cont_rhs(self, X):
        if self.motionless:
            return self.rhs_shraman(0, X)

        if self.switch:
            u, eta = X[:-1], X[-1]
            self.setParam('eta', eta)
            v = self.getParam('v')
        else:
            u, v = X[:-1], X[-1]

        dudx = self.spectral_deriv(np.fft.fft(u), 1)
        
        F = self.rhs_shraman(0, u) + v * dudx
        p = np.trapz(u * self.du0dx, dx=self.p['dx']) 

        return np.append(F, p)


    def init_palc(self, ds, t0, X0):
        if self.motionless:
            u0 = X0[:-1]
        else:
            u0 = X0[:-2]
        
        self.setParam(self.param_cont, X0[-1])
        self.init_cont(u0)

        self.ds = ds
        self.tangent = t0
        self.X0 = X0

    def palc_rhs(self, X):
        self.setParam(self.param_cont, X[-1])
        F = self.cont_rhs(X[:-1])
        s = np.dot(self.tangent, (X - self.X0)) - self.ds

        return np.append(F, s)

    def get_tangent(self, X, jacX=None, prev_tangent=None): # the sign may be off
        N = self.getParam('N')
        Np1 = N if self.motionless else N + 1

        if jacX is None:
            F_x = self.jacobian(X[:-1])
        else:
            F_x = jacX

        F_param = np.zeros(Np1)
        if self.param_cont == 'eta':
            F_param[:N] = 1

        elif self.param_cont == 'mu':
            F_param[:N] = X[:N]

        tx = np.linalg.solve(F_x, -F_param)
        t = np.append(tx, 1)

        if prev_tangent is not None:
            sign = np.dot(t, prev_tangent) # must be +
            if sign < 0:
                t = - t
        
        return t / np.linalg.norm(t)


    def jacobian_palc(self, X, jacX=None):
        N = self.getParam('N')
        self.setParam(self.param_cont, X[-1])

        Np1 = N if self.motionless else N+1

        jac = np.zeros((Np1+1, Np1+1))
        
        if jacX is None:
            jac[:Np1, :Np1] = self.jacobian(X[:-1])
        else:
            jac[:Np1, :Np1] = jacX

        # Deriv of F w/r to parm cont
        if self.param_cont == 'eta':
            jac[:N, Np1] = 1
        elif self.param_cont == 'mu':
            jac[:N, Np1] = X[:N]
 
        jac[Np1, :] = self.tangent # deriv of PALC w/r to (X, param)

        return jac

    def getParams(self, params_string):
        return [self.p[param] for param in params_string.split(' ')]

    def getParam(self, pname):
        return self.p[pname]

    def jacobian(self, X):
        if self.motionless:
            u = X
            v = 0
        elif self.switch:
            u, eta = X[:-1], X[-1]
            self.setParam('eta', eta)
            v = self.getParam('v')
        else:
            u, v = X[:-1], X[-1]

        mu, alpha, beta, gamma = self.getParams('mu alpha beta gamma')
        N, dx = self.getParams('N dx')
        if self.motionless:
            jac = np.zeros((N, N))
        else:
            jac = np.zeros((N+1, N+1))

        for i in range(N):
            j0 = N - 1 - i
            jf = j0 + N
            jac[i, :N] = v * self.deriv_du_dx[j0:jf] + alpha * self.deriv_d2u_dx2[j0:jf] \
                        + beta * self.deriv_d4u_dx4[j0:jf] + gamma * np.roll(self.deriv_coupling, i)
            jac[i, i] += mu - 3 * u[i] ** 2
        
        if self.motionless: return jac
        # deriv pinning w/r to u
        trapz_factor = np.ones(N)
        trapz_factor[0] = 0.5
        trapz_factor[-1] = 0.5
        
        jac[N, :-1] = dx * self.du0dx * trapz_factor

        # deriv F w/r to v (or eta if switch)
        if self.switch:
            jac[:-1, N] = 1
        else:
            jac[:-1, N] = self.spectral_deriv(np.fft.fft(u), 1)

        # deriv pinning w/r to v is just zero
        # so no need to set anything

        return jac

params = { 'N': 512, 'dx': 0.5, 
          'tau0': 18, 'tau1': 12, 'tau2': 32,
          'eta': 0, 'mu': -0.1, 'alpha': -1.0, 
          'beta': -1.0, 'gamma': 0.7 }

if __name__ == '__main__':
    #params['eta'] = 0.2
    shr = SHRaman(branch='dns', **params)

    shr.setInitialConditionGaussian()
    shr.solve(T_transient=100, T_f=1000)
    v, _ = shr.getVelocity()

    X = np.append(shr.getState(-1), v)
    shr.saveX(X, filename='x0')
