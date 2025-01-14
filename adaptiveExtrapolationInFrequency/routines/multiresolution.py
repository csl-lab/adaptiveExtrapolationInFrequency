## ----------------------------------------------------------------------------
## Imports

import numpy as np

#   masking
def masking(h, f):
    def _masking(x):
        return h(x) * f(x)
    return _masking

#   dilation
def dilation(alpha, f):
    def _dilation(x):
        return f(alpha * x)
    return _dilation

#   on torus
def torus(f):
    def _torus(x):
        return f((x + 0.5) % 1.0 - 0.5)
    return _torus

def phi_N(mw, N=64):
    # This defines the refinable function phi_N by the cascade algorithm of the refinement mask mw up to the maxiter's iteration in frequency domain.
    # mw is the refiment mask and must be a callable. maxiter is the amount of iterations of the cascade algorithm to be calculated to approximate the refinable function.
    def _phi_N(x):
        A = dilation(0.5, torus(mw))
        for k in range(N):
            A = masking(dilation(0.5 ** (k + 2), torus(mw)), A)
        return A(x)
    return _phi_N

def Phi_NK(x, mw, N=64, K=128):
    # Phi is the shift invariant function of the refinable function phi_N (receiving mw and maxiter) truncating the shifts up to |j| <= Z in frequency domain.
    # mw is the refiment mask and must be a callable. maxiter is the amount of iterations of the cascade algorithm to be calculated to approximate the refinable function. Z is the upper bound of the shifts.
    return np.sum(np.array([ np.abs(phi_N(mw, N=N)(x + j))**2 for j in np.linspace(-K, K, 2 * K + 1, endpoint=True) ]), axis=0)


def g_NK(x, mw, N=64, K=128):
    # g_NK is the wavelet mask of refiment mask mw in frequency domain.
    # mw is the refiment mask and must be a callable. maxiter is the amount of iterations of the cascade algorithm to be calculated to approximate the refinable function. Z is the upper bound of the shifts.
    return -1j * np.conj(torus(mw)(x + 0.5)) * Phi_NK(x + 0.5, mw, N=N, K=K) * np.exp(-1j*2*np.pi*x)

def psi_NK(mw, psi_0=1.0, N=64, K=128):
    # psi_NK is the wavelet constructed by refinement mask mw in frequency domain.
    # mw is the refiment mask and must be a callable. maxiter is the amount of iterations of the cascade algorithm to be calculated to approximate the refinable function. Z is the upper bound of the shifts. 
    # psi_0 is a free constant to determine the phase.
    if not isinstance(psi_0, (int, float, complex)) or np.abs(np.abs(psi_0) - 1.0) > 1E-7:
        raise ValueError('psi_0 must a scalar satisfying |psi_0|=1.')
    def _psi_NK(x):
        return psi_0 * phi_N(mw, N=N)(0.5*x) * g_NK(0.5*x, mw, N=N, K=K) 
    return _psi_NK
