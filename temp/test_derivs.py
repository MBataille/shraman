import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd

N = 128
dx = 0.5
L = N * dx
ORDER = 4
OMEGA = 2 * np.pi / 16


def get_ik():
    return 2j * np.pi / L * np.fft.fftfreq(N) * N

def get_jac_deriv(order=ORDER):
    ik_ift = np.fft.ifft((-get_ik()) ** order)
    return np.append(ik_ift, ik_ift)

def spectral_deriv(x, order=ORDER, castReal=False):
    ik = get_ik() ** order
    r = np.fft.ifft(ik * np.fft.fft(x))
    if castReal:
        return r.real
    return r

def wrapper_rhs_complex(X):
    return rhs_complex(X[:N] + 1j * X[N:])

def rhs_complex(x):
    dx = spectral_deriv(x)
    return np.append(dx.real, dx.imag)


def rhs_real(X):
    x, y = X[:N], X[N:]
    return np.append(
        spectral_deriv(x, castReal=True), 
        spectral_deriv(y, castReal=True) )

def jacobian(x):
    jac = np.zeros((2*N, 2*N))

    deriv = get_jac_deriv()

    for i in range(N):
        j0 = N - i
        jf = j0 + N

        # First block
        jac[i, :N] = deriv[j0:jf]

        # Second block
        jac[i+N, N:] = deriv[j0:jf]
        
    return jac

def test_jacobian():
    # Lets try with exp
    x = np.linspace(0, L, N, endpoint=False)
    A = np.exp(1j * OMEGA * x)
    U, V = A.real, A.imag
    X = np.append(U, V)

    B = wrapper_rhs_complex(X)
    W = rhs_real(X)

    num_jac_real = nd.Jacobian(rhs_real)(X)
    num_jac_complex = nd.Jacobian(wrapper_rhs_complex)(X)
    jac = jacobian(X)

    diff = np.abs(jac-num_jac_complex)

    print(np.sum(diff))
    plt.imshow(diff)
    plt.colorbar()
    plt.show()

    

if __name__ == '__main__':
    test_jacobian()