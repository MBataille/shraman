import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.integrate import solve_ivp

# TODO delete this
import numdifftools as nd

import os

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATADIR = os.path.join(BASEDIR, 'data', '')

def ptoname(**kwargs):
    s = ''
    for k, v in sorted(kwargs.items()):
        s += k + '=' + str(v) + '_'
    return s

def isComplex(A:np.ndarray):
    """Checks if the array A is complex"""
    return A.dtype == complex

# TODO add integration with numba lsoda

class LugiatoLefeverEquation():
    def __init__(self, **params):
        self.p = params

        self.p['L'] = self.p['N'] * self.p['dx']

        # we will consider tau0 = 1 and as such
        # we must rescale tau1 and tau2 accordingly.
        if 'tau0' in self.p:
            self.p['tau1'] = self.p['tau1'] / self.p['tau0']
            self.p['tau2'] = self.p['tau2'] / self.p['tau0']
        
        self.p['a'] = (self.p['tau1'] ** 2 + self.p['tau2'] ** 2) \
                    / (self.p['tau1'] * self.p['tau2'] ** 2) 

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
        self.kernel_ft = np.fft.fft(self.kernel_function())

        # For continuation
        self.motionless = False

    def kernel_function(self):
        """Returns the kernel evaluted in tau"""
        tau1, tau2, a = self.getParams('tau1 tau2 a')
        return a * np.exp(-self.tau / tau2) * np.sin(self.tau / tau1)

    def coupling(self, u, real=False, isInFourierSpace=True):
        """Returns the convolution of the coupling kernel
        with the given array u"""
        if isInFourierSpace:
            u_ft = u
        else:
            u_ft = np.fft.fft(u)

        R = np.fft.ifft(u_ft * self.kernel_ft) * self.p['dx']
        
        if real:
            return R.real
        return R

    def get_ik(self):
        """Returns the vector corresponding to ik and shifted"""
        return 2j * np.pi / self.p['L'] * np.fft.fftfreq(self.p['N'])\
             * self.p['N'] 

    def get_k(self):
        return 2 * np.pi / self.p['L'] * np.fft.fftfreq(self.p['N']) * self.p['N']

    def spectral_deriv(self, u_ft, order, real=False):
        """Returns the spectral derivative of the specified order
        of a given array u_ft (already in fourier space)"""
        ik = self.get_ik() ** order
        R = np.fft.ifft(ik * u_ft )

        if real:
            return R.real
        return R

    def rhs(self, t, A):
        """Returns the (complex) right-hand side of the LugiatoLefeverEquation"""
        
        S, Delta, beta, f_R, d4 = self.getParams('S Delta beta f_R d4')

        A_ft = np.fft.fft(A)
        modA2 = A.real * A.real + A.imag * A.imag

        d2A = self.spectral_deriv(A_ft, 2)
        d4A = self.spectral_deriv(A_ft, 4)
        coupling = self.coupling(modA2, isInFourierSpace=False)

        # debug
        # return 1j * beta * d2A # + 1j * f_R * A * coupling

        return S - (1 + 1j * Delta) * A + 1j * beta * d2A + d4 * 1j * d4A \
                + 1j * (1 - f_R) * modA2 * A + 1j * f_R * A * coupling

    def solve_dns(self, T_transient=0, T_f = 100):
        
        dt = 0.1
        
        if T_transient > 0:
            t_span_transient = (0, T_transient)
            self.A0 = solve_ivp(    
                self.rhs, t_span_transient, self.A0, 
                t_eval = [T_transient], max_step=dt
                        ).y[:,0]

        t_span = (T_transient, T_f)
        t_eval = np.linspace(*t_span, int((T_f - T_transient) / dt) + 1)

        self.t = t_eval - T_transient
        self.sol = solve_ivp(self.rhs, t_span, self.A0, t_eval=t_eval, max_step=0.005).y

    def init_cont(self, ds, t0, X0):
        """Sets the parts of the jacobian associted with
        spatial derivatives, pinning conditions and tangent.
        Must call this method before calling Newton's method."""
        N = self.getParam('N')
        U0 = X0[:N]
        V0 = X0[N:2*N]
        
        # for pinning conditions
        self.dU0dx = self.spectral_deriv(np.fft.fft(U0), 1, real=True)
        self.dV0dx = self.spectral_deriv(np.fft.fft(V0), 1, real=True)

        # for jac of spectral_deriv
        ik = self.get_ik()
        k = self.get_k()

        # jacobian of spatial derivatives
        ik_ift = np.fft.ifft(-ik).real
        self.deriv_dA_dx = np.append(ik_ift, ik_ift)

        k2 = np.fft.ifft(-k * k).real
        self.deriv_d2A_dx2 = np.append(k2, k2)

        k4 = np.fft.ifft(k * k * k * k).real
        self.deriv_d4A_dx4 = np.append(k4, k4)

        # for coupling
        chi = self.kernel_function()[::-1]
        self.deriv_coupling = np.append(chi, chi)

        # for PALC
        self.X0 = X0
        self.ds = ds
        self.tangent = self.get_tangent(X0, prev_tangent=t0)
    
    def palc_rhs(self, X):
        """Returns the right-hand side of the continuation equation
        for the pseudo-arclength method. Note: tangent must be already computed"""

        # len(X) = 2*N + 2 = real(N) + im(N) + pinning(1) + parameter(1)
        N, Delta, beta, f_R = self.getParams('N Delta beta f_R')

        U = X[:N]
        V = X[N:2*N]

        U_ft = np.fft.fft(U)
        V_ft = np.fft.fft(V)

        dU = self.spectral_deriv(U_ft, 1, real=True)
        dV = self.spectral_deriv(V_ft, 1, real=True)
        d2U = self.spectral_deriv(U_ft, 2, real=True)
        d2V = self.spectral_deriv(V_ft, 2, real=True)

        modA2 = U * U + V * V
        coupling =  f_R * self.coupling(modA2, real=True, isInFourierSpace=False)

        A = U + 1j * V
        c, S = X[2*N:]
        self.setParam('S', S)

        rhs_LLE_real = c * dU + S - U + Delta * V - beta * d2V \
            - (1 - f_R) * modA2 * V - V * coupling
        rhs_LLE_imag = c * dV - Delta * U - V + beta * d2U \
            + (1 - f_R)  * modA2 * U + U * coupling


        # This could be optimized by computing only once A_ft
        #rhs_LLE = self.rhs(0, A) + c * self.spectral_deriv(np.fft.fft(A), 1)


        dx = self.getParam('dx')
        pinning_condition = np.trapz(U * self.dU0dx - V * self.dV0dx, dx=dx)

        palc = np.dot(self.tangent, X - self.X0) - self.ds

        rhs = np.append(rhs_LLE_real, rhs_LLE_imag)
        rhs = np.append(rhs, np.array([pinning_condition, palc]))

        return rhs

    def jacobian_palc(self, X, jacX = None, forTangent=False):
        N, dx = self.getParams('N dx')
        M = len(X)

        U = X[:N]
        V = X[N:2*N]

        A = U + 1j * V
        dA_dx = self.spectral_deriv(np.fft.fft(A), 1)

        c, S = X[2*N:]

        self.setParam('S', S)

        jac = np.zeros((M, M))

        Delta, f_R, beta, d4 = self.getParams('Delta f_R beta d4')

        one_minus_fr = 1 - f_R

        convolution = self.coupling(U*U + V*V, real=True, isInFourierSpace=False)

        if jacX is None:

            for i in range(N):
                j0 = N - i
                jf = j0 + N

                # dont know why but this works
                coupling_term =  2 * dx * f_R * self.deriv_coupling[j0-1:jf-1]
                # J_11
                jac[i, :N] = c * self.deriv_dA_dx[j0:jf]  - V[i] * U * coupling_term
                jac[i, i] -= 1 + 2 * one_minus_fr * U[i] * V[i]

                # J_12
                jac[i, N:2*N] =  -beta * self.deriv_d2A_dx2[j0:jf]  - d4 * self.deriv_d4A_dx4[j0:jf] - V[i] * V * coupling_term
                jac[i, i + N] += Delta - one_minus_fr * (U[i] * U[i] + 3 * V[i] * V[i]) - f_R * convolution[i]

                # J_21
                jac[i+N, :N] =  beta * self.deriv_d2A_dx2[j0:jf]  + d4 * self.deriv_d4A_dx4[j0:jf] + U[i] * U * coupling_term
                jac[i+N, i] += - Delta + one_minus_fr * (V[i] * V[i] + 3 * U[i] * U[i]) + f_R * convolution[i]

                #J_22
                jac[i+N, N:2*N] = c * self.deriv_dA_dx[j0:jf]  + U[i] * V * coupling_term
                jac[i+N, i+N] += -1 + 2 * one_minus_fr * V[i] * U[i]


            # J_13
            jac[0:N, 2*N] = dA_dx.real
            
            #J_23
            jac[N:2*N, 2*N] = dA_dx.imag

            # J_31
            jac[2*N, 0:N] = self.dU0dx  * dx

            #J_32
            jac[2*N, N:2*N] = - self.dV0dx * dx
        
        else:
            jac[0:M, 0:M] = jacX

        # PALC terms
        if not forTangent:
            jac[2*N + 1, :]= self.tangent # deriv w/r to X

        jac[:N, 2*N + 1] = 1 # deriv w/r to S

        return jac

    def get_tangent(self, X, prev_tangent=None):
        N = self.getParam('N')

        jac = self.jacobian_palc(X, forTangent=True)
        F_x = jac[:-1, :-1]
        F_param = jac[:-1, -1]

        t = np.linalg.solve(F_x, -F_param)
        t = np.append(t, 1)

        if prev_tangent is not None:
            sign = np.dot(t, prev_tangent)
            if sign < 0:
                t = -t

        return t / np.linalg.norm(t)

    def L2norm(self, X):
        N = self.getParam('N')
        U, V = X[:N], X[N:2*N]
        return np.sum(U*U + V*V) / N
    
    def check_param_name(self, param_name):
        if not param_name in self.p:
            raise ValueError(f"Could not find {param_name} in current parameter list {self.p.keys()}")

    def getParam(self, param_name):
        self.check_param_name(param_name)
        return self.p[param_name]

    def getParams(self, params_string):
        return [self.getParam(param_name) for param_name in params_string.split(' ')]
 
    def setParam(self, param_name, param_val):
        self.check_param_name(param_name)
        self.p[param_name] = param_val

    def setParams(self, params_string, params_arr):
        for param_name, param_val in zip(params_string.split(' '), params_arr):
            self.setParam(param_name, param_val)

    def setInitialCondition(self, init_cont):
        self.A0 = init_cont

    def setInitialConditionGaussian(self, x0, sigma, phase, amp):
        gaussian = amp * np.exp(- (self.tau - x0) ** 2 / (2 * sigma * sigma))
        self.setInitialCondition(gaussian * np.exp(1j * gaussian))

    def getState(self, k):
        return self.sol[:, k]

    def getFilename(self, pre=None, ext='', X=''):
        S, Delta = self.getParams('S Delta')
        if pre is None: pre = self.branchfolder
        return pre + X + ptoname(S=S, Delta=Delta) + ext


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

    def center(self, u):
        xpos = self.getXpos(u, index=True)
        return np.roll(u, int(len(u)/2) - xpos)

    def getXpos(self, A, find='max', index=False):
        conv = self.coupling(np.fft.fft(np.abs(A) ** 2))

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

# ---------------- Tests ---------------------
# TODO delete this once it's tested.

    def test_jacobian(self, X):
        # may take a while
        n_jac = nd.Jacobian(self.palc_rhs)(X)
        jac = self.jacobian_palc(X)
        np.save(self.branchfolder + 'n_jac', n_jac)

        #n_jac = np.load(self.branchfolder + 'n_jac.npy')
        diff = np.abs(n_jac - jac)

        print(f'Sum of diff is {diff.sum()}')

        print(f'c = {X[-2]}')

        N = self.getParam('N')

        plt.plot(np.roll(self.deriv_dA_dx, int(N/2)))
        plt.plot(np.roll(self.deriv_d2A_dx2, int(N/2)))
        plt.show()

        ax1 = plt.subplot()

        jac_slice = jac[int(N/2), N:2*N]
        n_jac_slice = n_jac[int(N/2), N:2*N]

        ax1.plot(jac_slice)
        ax1.plot(n_jac_slice)

        ax2 = ax1.twinx()
        # ax1.plot(np.roll(self.deriv_dA_dx, int(N/2)))

        #d2 = np.roll(-self.deriv_d2A_dx2, int(N/2))
        #ax2.plot(np.roll(-self.deriv_d2A_dx2, int(N/2)), color='tab:green')
        #ax2.plot(jac_slice / n_jac_slice)# jac[int(N/2), :])
        ax2.plot(np.abs(jac_slice - n_jac_slice))
        plt.show()

        plt.imshow(diff)
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    params = {
        'Delta': 1.7,
        'S': 1.2,
        'f_R': 0.05,
        'tau0': 1,
        'tau1': 3,
        'tau2': 10,
        'beta': 1.0,
        'dx': 0.25,
        'N' : 512,
        'd4' : 0
    }

    lle = LugiatoLefeverEquation(branch='lle_dns_test', **params)

    X = lle.loadX()

    N = lle.getParam('N')
    A0 = X[:N] + 1j * X[N:2*N]
    c0, S0 = X[2*N:2*N+2]
    t0 = np.zeros_like(X)
    t0[-1] = 1

    lle.init_cont(0.1, t0, X)

    nX = X
    nX = X # + lle.tangent * lle.ds
    Y = lle.palc_rhs(nX)

    print(f'Sum of Y is {Y.sum()}')

    plt.plot(Y)
    plt.show()

    lle.test_jacobian(nX)

    x = lle.tau
    x0 = (x[0] + x[-1]) / 2
    #Are = 0.7 + 1 * np.exp(- (x - x0) ** 2 / 8)
    #Aim = - 0.6 + 1 * np.exp(- (x - x0) ** 2 / 8)
    #Aim = 0

    # A0 = Are + 1j * Aim
    A0 = lle.loadState(lle.getFilename(ext='.npy'))
    # lle.setInitialConditionGaussian(50, 1, 0, 1)
    lle.setInitialCondition(A0)


    plt.plot(x, lle.A0.real)
    plt.plot(x, lle.A0.imag)
    plt.show()

    lle.solve_dns(0, 1000)

    A = lle.getState(-1)

    plt.imshow(np.abs(lle.sol), aspect='auto')
    plt.show()

    plt.plot(x, np.abs(A))
    plt.show()


    lle.saveState(-1)

    v, v_err = lle.getVelocity()

    plt.plot(lle.t, lle.xs)
    plt.plot(lle.t, v * lle.t + lle.xs[0])
    plt.show()

    X = np.append(A.real, A.imag)
    X = np.append(X, [v, lle.getParam('S')])

    lle.saveX(X)

