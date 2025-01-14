## ----------------------------------------------------------------------------
## Imports

import numpy as np
from numpy.typing import NDArray

from collections.abc import Callable

def shifted_exponential_collection(ndim : int, xo : NDArray, h : Callable[[NDArray], complex] | None = None, A : NDArray | float = 1.0) -> Callable[[NDArray], NDArray]:
  if xo.ndim == 1:
    n = 1
    if ndim != xo.size:
      raise ValueError(f'Provided dimension is {ndim} whereas shape of xo is {xo.shape}')
  else:
    n = xo.shape[0]
    if ndim != xo.shape[1]:
      raise ValueError(f'Provided dimension is {ndim} whereas shape of xo is {xo.shape}')
  # reshape translations
  xo = xo.reshape((n, ndim))
  
  # callables
  def eval(w : NDArray) -> NDArray:
    return A @ np.exp(- 1j * 2 * np.pi * (xo @ w.T))
  
  def eval_h(w : NDArray) -> NDArray:
    return A @ np.exp(- 1j * 2 * np.pi * (xo @ w.T)) * h(w)
  
  if h:
    return eval_h
  return eval