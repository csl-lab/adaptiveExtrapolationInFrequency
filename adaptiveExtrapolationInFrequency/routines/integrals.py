## ----------------------------------------------------------------------------
## Imports

import numpy as np


def inverse_fourier_transform_1d(f, x, B, M, debug=False, interp=True, w=False, n=False, E=False):
  # evaluation points
  if not w:
    w = (2 * B) * (2 * np.linspace(0, M-1, M) - M) / (2 * M)
  # function values
  f_w = f(w)
  # samples
  if not n:
    n = np.linspace(0, M-1, M)
    n = np.fft.ifftshift(np.where(n >= M // 2, n - M, n))
  # samples
  if not E:
    u_n = (2 * B) * np.exp(-1j * np.pi * n) * np.fft.ifftshift(np.fft.ifft(f_w))
  else:
    u_n = (2 * B) * E * np.fft.ifftshift(np.fft.ifft(f_w))
  # return internal variables when debugging
  if debug:
    return w, f_w, n, u_n, np.exp(-1j * np.pi * n)
  # return interpolation
  if interp:
    return np.sum(np.array([ _u * np.sinc((2 * B) * x - _n) for _u, _n in zip(u_n, n) ], dtype=np.complex128), axis=0)
  raise NotImplementedError('interp = False not implemented.')

def inverse_fourier_transform_2d_squares(f, x, B, M, debug=False, interp=True, w=False, n=False, E=False):
  # evaluation points
  if not w:
    w = np.meshgrid((2 * B[0]) * (2 * np.linspace(0, M[0]-1, M[0]) - M[0]) / (2 * M[0]),
                    (2 * B[1]) * (2 * np.linspace(0, M[1]-1, M[1]) - M[1]) / (2 * M[1]))
  # function values
  f_w = f(np.vstack([ w[0].ravel(), w[1].ravel() ]).T)
  # samples
  if not n:
    n = list(np.meshgrid(np.linspace(0, M[0]-1, M[0]),
                         np.linspace(0, M[1]-1, M[1])))
    n[0] = np.fft.ifftshift(np.where(n[0] >= M[0] // 2, n[0] - M[0], n[0]))
    n[1] = np.fft.ifftshift(np.where(n[1] >= M[1] // 2, n[1] - M[1], n[1]))
  # samples
  if not E:
    u_n = (4 * B[0] * B[1]) * np.exp(-1j * np.pi * (n[0] + n[1])) * np.fft.ifftshift(np.fft.ifft2(f_w.reshape((M[0], M[1]))))
  else:
    u_n = (4 * B[0] * B[1]) * E * np.fft.ifftshift(np.fft.ifft2(f_w))
  # return internal variables when debugging
  if debug:
    return w, f_w, n, u_n, np.exp(-1j * np.pi * (n[0] + n[1]))
  # return interpolation
  if interp:
    n = np.vstack([ n[0].ravel(), n[1].ravel() ]).T
    u_n = u_n.ravel()
    u_x = np.zeros((x.shape[0],), dtype=complex)
    for _u, _n in zip(u_n, n):
      u_x = u_x + _u * np.sinc((2 * B[0]) * x.T[0] - _n[0]) * np.sinc((2 * B[1]) * x.T[1] - _n[1])
    return u_x
  raise NotImplementedError('interp = False not implemented.')
