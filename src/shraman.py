import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import diags
import os
#from pathlib import Path

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATADIR = os.path.join(BASEDIR, 'data', '')

def ptoname(eta, mu, gamma):
    return f'eta={eta}_mu={mu}_gamma={gamma}'

class SHRaman:
    def __init__(self, **params):
        self.p = params
        
        # Defining useful parameters
        self.p['L']= self.p['N'] * self.p['dx']
        self.p['a'] = (self.p['tau1'] ** 2 + self.p['tau2'] ** 2) \
                    / (self.p['tau1'] * self.p['tau2'] ** 2)

        if 'branch' in params:
            self.branch = params['branch']
            self.branchfolder = os.path.join(DATADIR, self.branch, '')
            if not os.path.exists(self.branchfolder):
                os.mkdir(self.branchfolder)

        # Initialize FFT of Coupling kernel
        N, dx = self.p['N'], self.p['dx']
        self.tau = np.linspace(0, N*dx, N, endpoint=False)
        self.kernel_ft = np.fft.fft(self.kernel())
        
    def kernel(self):
        tau1, tau2, a = self.p['tau1'], self.p['tau2'], self.p['a']
        return a * np.exp(-self.tau / tau2) * np.sin(self.tau / tau1)

    def coupling(self, u_ft):
        return np.fft.ifft(u_ft * self.kernel_ft).real

    def spectral_deriv(self, u_ft, order):
        ik = (2j * np.pi / self.p['L'] * np.fft.fftfreq(self.p['N'])\
             * self.p['N']) ** order
        return np.fft.ifft(ik * u_ft ).real

    def rhs_shraman(self, t, u):
        eta, mu, alpha, beta, gamma = self.p['eta'], self.p['mu'], self.p['alpha'], self.p['beta'], self.p['gamma']

        u_ft = np.fft.fft(u)
        d2u = self.spectral_deriv(u_ft, 2)
        d4u = self.spectral_deriv(u_ft, 4)
        
        return eta + mu * u - u ** 3 + alpha * d2u + beta * d4u + gamma * self.coupling(u_ft)

    def setInitialConditionGaussian(self):
        L = self.p['L']
        self.u0 = np.exp(- (self.tau - L/4)**2 / (L / 50)) - 0.2 

    def solve(self, T_transient=0, T_f=100):
        Tf = T_f
        dt = 0.1

        if T_transient > 0:
            t_span_trans = (0., T_transient)
            self.u0 = solve_ivp(self.rhs_shraman, t_span_trans, self.u0, t_eval=[T_transient]).y[:, 0]

        t_span = (T_transient, Tf)
        t_eval = np.linspace(*t_span, int((Tf - T_transient) / dt) + 1)

        self.t = t_eval - T_transient # so that it starts at 0
        self.sol = solve_ivp(self.rhs_shraman, t_span, self.u0, t_eval=t_eval).y

    def getState(self, k):
        return self.sol[:, k]
    
    def save(self):
        pass

    def saveOP(self):
        pass

    def saveState(self, k = -1):
        np.save(self.getFilename(), self.getState(k))

    def loadState(self, filename):
        return np.load(filename)

    def loadInitialCondition(self, filename):
        self.u0 = self.loadState(filename)

    def setInitialCondition(self, u0):
        self.u0 = u0

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
        xs = np.zeros_like(self.t)
        
        jump = 0
        for k in range(len(self.t)):
            xs[k] = self.getXpos(self.getState(k)) + jump
            if k > 0: # assuming it only moves to the right!
                diff = xs[k-1] - xs[k]
                if abs(diff) > self.p['L'] / 2:
                    jump += self.p['L'] * diff / abs(diff)
                    xs[k] += jump

        p, cov = np.polyfit(self.t, xs, 1, cov=True)
        v = p[0]
        verr = np.sqrt(cov[0, 0])

        return v, verr

    def getFilename(self, pre=None, ext=''):
        eta, mu, gamma = self.p['eta'], self.p['mu'], self.p['gamma']
        if pre is None: pre = self.branchfolder
        return pre + ptoname(eta, mu, gamma) + ext

params = {'N': 512, 'dx': 0.5, 'tau1': 3, 'tau2': 10,
          'eta': 0, 'mu': -0.1, 'alpha': -1.0, 'beta': -1.0,
          'gamma': 0.7}

if __name__ == '__main__':
    shr = SHRaman(**params)

    shr.setInitialConditionGaussian()
    shr.solve()
    shr.getVelocity()
