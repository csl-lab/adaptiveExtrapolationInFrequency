## ----------------------------------------------------------------------------
## Imports

import numpy as np
from numpy.typing import NDArray
from typing import Callable
from adaptiveExtrapolationInFrequency.objects.domain import Domain

## ----------------------------------------------------------------------------
## Base class
class SMultiplier(object):
  def __init__(self, ndim : int = 1, alpha : float = 2.0, eps : float = 1E-6, domain : Domain | None = None, f_vec : Callable[[NDArray], NDArray] | None = None):
    # list of function handles
    self.f = []
    # number of functions
    self.size = 0
    # optional input: vector-valued function
    if f_vec:
      try:
        f_x = f_vec(np.ones((ndim,), dtype=float))
        f_size = f_x.size
      except:
        raise ValueError(f'Vector-valued function causes an exception when taking a ({ndim},) array as input.')
      try:
        f_x = f_vec(np.ones((2, ndim), dtype=float))
        if f_x.shape[1] != 2:
          raise ValueError(f'Function does not return a ({f_size}, 2) array when evaluated at a (2, {ndim}) array')
      except:
        raise ValueError(f'Function causes an exception when taking a (2, {ndim}) array as input.')
      # vectorized evaluation
      self.vectorized = True
      self.f_vec = f_vec
      self.size = f_size
    else:
      # standar evaluation
      self.vectorized = False
      self.f_vec = None
    # dimension of the domain
    self.ndim = ndim
    # scaling factor
    self.alpha = alpha
    # regularizing factor
    self.eps = 1E-6
    # domain
    self._domain = domain
    # quadrature
    self.quadrule = 'mc'

  @property
  def domain(self) -> Domain:
    return self._domain
  
  @domain.setter
  def domain(self, omega : Domain):
    self._domain = omega

  # appends functions to collection
  def append(self, f : Callable[[NDArray], NDArray], validate : bool = True) -> None:
    if not self.vectorized:
      if validate:
        try:
          f_x = f(np.ones((self.ndim,), dtype=float))
        except:
          raise ValueError(f'Function causes an exception when taking a ({self.ndim},) array as input.')
        try:
          f_x = f(np.ones((2, self.ndim), dtype=float))
          if f_x.size != 2:
            raise ValueError(f'Function does not return a (2,) array when evaluated at a (2, {self.ndim}) array')
        except:
          raise ValueError(f'Function causes an exception when taking a (2, {self.ndim}) array as input.')
      self.f.append(f)
      self.size += 1
  
  # evaluates functions in collection
  def eval_function(self, w : NDArray, index : int | None = None) -> NDArray:
    # vectorized
    if self.vectorized:
      if index:
        return self.f_vec(w)[index]
      return self.f_vec(w)
    # not vectorized
    if index:
      return self.f[index](w)
    return np.array([ f(w) for f in self.f ], dtype=complex).reshape(self.size, w.shape[0])

  # evaluates the S-multiplier at points w
  def eval(self, S : NDArray, w : NDArray) -> NDArray:
    # function evaluation
    f_w = self.eval_function(w)
    # scaled function evaluation
    Da_f_w = self.eval_function(self.alpha * w)
    # auxiliary variable for S inner product
    Sf_w = S @ f_w + self.eps * f_w

    return np.sum(np.conj(Sf_w) * Da_f_w, axis = 0) / np.sum(np.conj(Sf_w) * f_w, axis = 0)
  
  # evaluates the windowed S-multiplier multiplier at points w
  def eval_windowed(self, S : NDArray, w : NDArray) -> NDArray:
    # window
    h_w = self.domain.eval_window(w)
    # points on support
    s_idx = np.where(h_w > 0, True, False)
    ws = w[s_idx]
    # function evaluation
    f_ws = self.eval_function(ws)
    # scaled function evaluation
    Da_f_ws = self.eval_function(self.alpha * ws)
    # auxiliary variable for S inner product
    Sf_ws = S @ f_ws + self.eps * f_ws
    # multiplier
    m_ws = np.sum(np.conj(Sf_ws) * Da_f_ws, axis = 0) / np.sum(np.conj(Sf_ws) * f_ws, axis = 0)
    # full multiplier
    m_w = np.zeros((w.shape[0],), dtype=complex)
    m_w[s_idx] = m_ws

    return m_w * h_w

  # evaluates residual matrix associated to an S-multiplier at points w
  def eval_residual_matrix(self, S : NDArray, w : NDArray) -> NDArray:
    # function evaluation
    f_w = self.eval_function(w)
    # scaled function evaluation
    Da_f_w = self.eval_function(self.alpha * w)
    # auxiliary variable for S inner product
    Sf_w = S @ f_w + self.eps * f_w
    # multiplier evaluation
    mS_w = np.sum(np.conj(Sf_w) * Da_f_w, axis = 0) / np.sum(np.conj(Sf_w) * f_w, axis=0)
    # residual evaluation
    r_w = (Da_f_w - (f_w * mS_w)).T 

    return np.array([ np.outer(r, np.conj(r)) for r in r_w ], dtype=complex)
  
  # evaluates residual matrix associated to an S-multiplier at points w
  def eval_residual_gram(self, S : NDArray, n : int) -> NDArray:
    if self.quadrule == 'mc':
      w = self.domain.sample(n)
    elif self.quadrule == 'quad':
      w, c = self.domain.quadrature(n)
    else:
      raise NotImplemented(f'{self.quadrule} not implemented.')
    # function evaluation
    f_w = self.eval_function(w)
    # scaled function evaluation
    Da_f_w = self.eval_function(self.alpha * w)
    # auxiliary variable for S inner product
    Sf_w = S @ f_w + self.eps * f_w
    # multiplier evaluation
    mS_w = np.sum(np.conj(Sf_w) * Da_f_w, axis = 0) / np.sum(np.conj(Sf_w) * f_w, axis=0)
    # residual evaluation
    r_w = Da_f_w - (f_w * mS_w)

    if self.quadrule == 'mc':
      return r_w @ np.conj(r_w).T / n
    if self.quadrule == 'quad':
      return (r_w * c) @ np.conj(r_w).T

    raise NotImplemented(f'{self.quadrule} not implemented.')
  
