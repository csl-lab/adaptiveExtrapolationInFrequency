## ----------------------------------------------------------------------------
## Imports
import abc
import numpy as np
import scipy.optimize
from numpy.typing import NDArray

## ----------------------------------------------------------------------------
## Functions
def project_l2(x : NDArray) -> NDArray:
  # check if on set
  if np.linalg.norm(x, ord=2) <= 1.0:
    return x
  # project
  return x / np.linalg(x, ord=2)

def project_linf(x : NDArray) -> NDArray:
  # check if on set
  if np.linalg.norm(x, ord=np.inf) <= 1.0:
    return x
  # project
  return np.sign(x) * np.minimum(1.0, np.abs(x))

def project_l1(x : NDArray) -> NDArray:
  # check if on set
  if np.linalg.norm(x, ord=1) <= 1.0:
    return x
  # soft-thresholding
  def sft(t):
    return 1.0 - np.sum(np.maximum(0.0, np.abs(x) -  t))
  # find shrinkage
  tp = scipy.optimize.toms748(sft, 0, np.abs(x).max())
  # project
  return np.maximum(0.0, np.abs(x) -  tp) * np.sign(x)


## ----------------------------------------------------------------------------
## Base class
class Projector(object):
  def __init__(self, n : int):
    # matrix shape
    self.shape = (n, n)
    # numer of entries
    self.size = n ** 2
  
  @abc.abstractmethod
  def eval(self, X : NDArray) -> NDArray:
    pass

## ----------------------------------------------------------------------------
## Schatten lp
class ProjSchattenLp(Projector):
  def __init__(self, n : int, p : float):
    super().__init__(n = n)
    if p not in [ 1.0, 2.0, 1, 2, np.inf ]:
      raise ValueError(f'Exponent {p} not supported')
    # exponent
    self.p = p

  def eval(self, X : NDArray) -> float:
    # eigendecomposition
    D = np.linalg.eigvalsh(X)
    # evaluation of norm
    return np.linalg.norm(D, ord=self.p)

  def proj(self, X : NDArray) -> NDArray:
    # eigendecomposition
    [ D, V ] = np.linalg.eigh(X)
    # safeguard: truncate to 0
    D = np.maximum(0.0, D)
    # projection of eigenvalues
    if self.p == 1:
      D = project_linf(D)
    elif self.p == np.inf:
      D = project_l1(D)
    else:
      D = project_l2(D)
    # matrix projection
    return (V * D) @ np.conj(V).T