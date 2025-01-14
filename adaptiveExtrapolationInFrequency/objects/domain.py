## ----------------------------------------------------------------------------
## Imports

import abc
import numpy as np
from numpy.typing import NDArray

## ----------------------------------------------------------------------------
## Functions

def zeroth_order_window_1d(w : NDArray) -> NDArray:
  return np.where(np.abs(w) <= 1.0, 1.0, 0.0)

def zeroth_order_window_2d(w : NDArray) -> NDArray:
  return np.where(np.abs(w.T[0]) <= 1.0, 1.0, 0.0) * np.where(np.abs(w.T[1]) <= 1.0, 1.0, 0.0)

def first_order_window_1d(w : NDArray, delta : float) -> NDArray:
  return np.minimum(1.0, np.maximum(0.0, 1.0 - np.abs(w)) / delta)

def first_order_window_2d(w : NDArray, delta : float):
  return np.minimum(1.0, np.maximum(0.0, 1.0 - np.linalg.norm(w, axis=1, ord=np.inf)) / delta)

## ----------------------------------------------------------------------------
## Base class

class Domain(object):
  def __init__(self, ndim : int, window : str, delta : float, quadrule : str):
    self.ndim = ndim
    self.window = window
    self.quadrule = quadrule
    self._delta = delta

  @property
  def delta(self) -> float:
    return self._delta
  
  @delta.setter
  def delta(self, s : float) -> None:
    self._delta = s

  @abc.abstractmethod
  def eval_window(self, w : NDArray) -> None:
    pass

  @abc.abstractmethod
  def sample(self, n : int) -> NDArray:
    pass

  @abc.abstractmethod
  def quadrature(self, n : int) -> tuple[NDArray, NDArray]:
    pass

## ----------------------------------------------------------------------------
## 1D Domains

class Square1D(Domain):
  def __init__(self, width : float = 1.0, window : str = 'zeroth_order', delta : float = 1E-2, quadrule : str = 'equispaced'):
    super().__init__(ndim = 1, delta = delta, window = window, quadrule = quadrule)
    self.width = width

  def eval_window(self, w : NDArray) -> NDArray:
    if self.window == 'zeroth_order':
      return zeroth_order_window_1d(2.0 * w / self.width)
    if self.window == 'first_order':
      return first_order_window_1d(2.0 * w / self.width, self.delta)
    return NotImplementedError(f'{self.window} not implemented.')
  
  def sample(self, n : int) -> NDArray:
    return (np.random.rand(n) - 0.5) * self.width
  
  def quadrature(self, n : int) -> tuple[NDArray, NDArray]:
    # --- default
    # nodes
    wq = np.linspace(-0.5 * self.width, +0.5 * self.width, n)
    # weights 
    cq = (self.width / (n-1)) * np.ones((n,), dtype=float)

    return wq, cq 
  
class Annulus1D(Domain):
  def __init__(self, rmin : float = 1.0, rmax : float = 2.0, window : str = 'zeroth_order', delta : float = 1E-2, quadrule : str = 'equispaced'):
    super().__init__(ndim = 1, delta = delta, window = window, quadrule = quadrule)
    self.rmin = rmin
    self.rmax = rmax
    self.width = 2 * rmax
    self.rctr = 0.5 * (rmax + rmin)
    self.rwdt = rmax - rmin

  def eval_window(self, w : NDArray) -> NDArray:
    if self.window == 'zeroth_order':
      return zeroth_order_window_1d(2.0 * (np.abs(w) - self.rctr) / self.rwdt)
    if self.window == 'first_order':
      return first_order_window_1d(2.0 * (np.abs(w) - self.rctr) / self.rwdt, self.delta)
    return NotImplementedError(f'{self.window} not implemented.')
  
  def sample(self, n : int) -> NDArray:
    return (self.rmin + np.random.rand(n) * self.rwdt) * np.sign(np.random.rand(n) - 0.5)
  
  def quadrature(self, n : int) -> tuple[NDArray, NDArray]:
    # --- default
    # distribution of nodes
    n_n = n // 2
    n_p = n - n_n
    # nodes
    wq = np.hstack([ np.linspace(-self.rmax, -self.rmin, n_n), np.linspace(self.rmin, self.rmax, n_p) ])
    # weights
    cq = np.hstack([ (self.rwdt / (n_n - 1)) * np.ones((n_n,), dtype=float), (self.rwdt / (n_p - 1)) * np.ones((n_p,), dtype=float) ])

    return wq, cq 

## ----------------------------------------------------------------------------
## 2D Domains

class Square2D(Domain):
  def __init__(self, r : float = 1.0, window : str = 'zeroth_order', delta : float = 1E-2, quadrule : str = 'equispaced'):
    super().__init__(ndim = 1, delta = delta, window = window, quadrule = quadrule)
    self.r = r
    self.width = np.array([ 2 * r, 2 * r ])

  def eval_window(self, w : NDArray) -> NDArray:
    if self.window == 'zeroth_order':
      return zeroth_order_window_2d(w / self.r)
    if self.window == 'first_order':
      return first_order_window_2d(w / self.r, self.delta)
    return NotImplementedError(f'{self.window} not implemented.')
  
  def sample(self, n : int) -> NDArray:
    return (2.0 * np.random.rand(n, 2) - 1.0) * self.r
  
  def quadrature(self, n : NDArray):
    if isinstance(n, int):
      n = [ n, n ]
    nmsh = np.meshgrid(np.linspace(-self.r, +self.r, n[0]), np.linspace(-self.r, +self.r, n[1]))
    return np.vstack([ nmsh[0].ravel(), nmsh[1].ravel() ]).T, np.ones((n[0] * n[1],), dtype=float)
  
class Circle2D(Domain):
  def __init__(self, r : float = 1.0, window : str = 'zeroth_order', delta : float = 1E-2, quadrule : str = 'equispaced'):
    super().__init__(ndim = 1, delta = delta, window = window, quadrule = quadrule)
    self.r = r
    self.width = np.array([ 2 * r, 2 * r ])

  def eval_window(self, w : NDArray) -> NDArray:
    if self.window == 'zeroth_order':
      return zeroth_order_window_1d(np.linalg.norm(w, axis=1, ord=2) / self.r)
    if self.window == 'first_order':
      return first_order_window_1d(np.linalg.norm(w, axis=1, ord=2) / self.r, self.delta)
    return NotImplementedError(f'{self.window} not implemented.')
  
  def sample(self, n : int) -> NDArray:
    ws = self.r * np.sqrt(np.random.rand(n)) * np.exp(1j * 2 * np.pi * np.random.rand(n))
    return np.vstack([ ws.real, ws.imag ], dtype=float).T
  
  def quadrature(self, n : NDArray):
    if isinstance(n, int):
      n = [ n, n ]
    nmsh = np.meshgrid(np.linspace(0, self.r, n[0]), np.linspace(0.0, 2 * np.pi, n[1], endpoint=False))
    return np.vstack([ nmsh[0].ravel() * np.cos(nmsh[1].ravel()), nmsh[0].ravel() * np.sin(nmsh[1].ravel()) ]).T, np.ones((n[0] * n[1],), dtype=float)
  
class SquareAnnulus2D(Domain):
  def __init__(self, rmin : float = 1.0, rmax : float = 2.0, window : str = 'zeroth_order', delta : float = 1E-2, quadrule : str = 'equispaced'):
    super().__init__(ndim = 1, delta = delta, window = window, quadrule = quadrule)
    self.rmin = rmin
    self.rmax = rmax
    self.width = np.array([ 2 * rmax, 2 * rmax ])
    self.delta_min = 1 - self.rmin / (self.rmin + self.delta * (self.rmax - self.rmin))
    self.delta_max = self.delta * (self.rmax - self.rmin) / self.rmax

  @property
  def delta(self) -> float:
    return self._delta

  @delta.setter
  def delta(self, s : float) -> None:
    self._delta = s
    self.delta_min = 1 - self.rmin / (self.rmin + self.delta * (self.rmax - self.rmin))
    self.delta_max = self.delta * (self.rmax - self.rmin) / self.rmax

  def eval_window(self, w : NDArray) -> NDArray:
    if self.window == 'zeroth_order':
      return zeroth_order_window_2d(w / self.rmax) - zeroth_order_window_2d(w / self.rmin)
    if self.window == 'first_order':
      return first_order_window_2d(w / self.rmax, self.delta_max) - first_order_window_2d(w / self.rmin, self.delta_min)
    return NotImplementedError(f'{self.window} not implemented.')
  
  def sample(self, n : int) -> NDArray:
    # sample radius
    rs = self.rmin + (self.rmax - self.rmin) * np.random.rand(n)
    # sample side
    bs = np.floor(4 * np.random.rand(n))
    # sample on side
    ws = rs * (1j ** bs) + (-self.rmin + (self.rmin + self.rmax) * np.random.rand(n)) * (1j ** (bs + 1)) 

    return np.vstack([ ws.real.ravel(), ws.imag.ravel() ], dtype=float).T
  
  def quadrature(self, n):
    if isinstance(n, int):
      n = [ n, n ]
    rs = np.linspace(self.rmin, self.rmax, n[0])
    ts = np.linspace(0.0, 2 * np.pi, n[1], endpoint=False)
    msh = np.meshgrid(rs, ts)
    ws = np.vstack([ np.cos(msh[1].ravel()), np.sin(msh[1].ravel()) ], dtype=float)

    return (ws * (msh[0].ravel() / np.linalg.norm(ws, axis=0, ord=np.inf))).T, np.ones((n[0] * n[1],), dtype=float)
  
class Annulus2D(Domain):
  def __init__(self, rmin : float = 1.0, rmax : float = 2.0, window : str = 'zeroth_order', delta : float = 1E-2, quadrule : str = 'equispaced'):
    super().__init__(ndim = 1, delta = delta, window = window, quadrule = quadrule)
    self.rmin = rmin
    self.rmax = rmax
    self.width = np.array([ 2 * rmax, 2 * rmax ])
    self.delta_min = 1 - self.rmin / (self.rmin + self.delta * (self.rmax - self.rmin))
    self.delta_max = self.delta * (self.rmax - self.rmin) / self.rmax

  @property
  def delta(self) -> float:
    return self._delta

  @delta.setter
  def delta(self, s : float) -> None:
    self._delta = s
    self.delta_min = 1 - self.rmin / (self.rmin + self.delta * (self.rmax - self.rmin))
    self.delta_max = self.delta * (self.rmax - self.rmin) / self.rmax

  def eval_window(self, w : NDArray) -> NDArray:
    if self.window == 'zeroth_order':
      return zeroth_order_window_1d(np.linalg.norm(w, axis=1, ord=2) / self.rmax) - zeroth_order_window_1d(np.linalg.norm(w, axis=1, ord=2) / self.rmin)
    if self.window == 'first_order':
      return first_order_window_1d(np.linalg.norm(w, axis=1, ord=2) / self.rmax, self.delta_max) - first_order_window_1d(np.linalg.norm(w, axis=1, ord=2) / self.rmin, self.delta_min)
    return NotImplementedError(f'{self.window} not implemented.')
  
  def sample(self, n : int) -> NDArray:
    ws = np.sqrt((self.rmax ** 2 - self.rmin ** 2) * np.random.rand(n) + self.rmin ** 2) * np.exp(1j * 2 * np.pi * np.random.rand(n))
    return np.vstack([ ws.real, ws.imag ], dtype=float).T
  
  def quadrature(self, n):
    if isinstance(n, int):
      n = [ n, n ]
    nmsh = np.meshgrid(np.linspace(self.rmin, self.rmax, n[0]), np.linspace(0.0, 2 * np.pi, n[1], endpoint=False))
    return np.vstack([ nmsh[0].ravel() * np.cos(nmsh[1].ravel()), nmsh[0].ravel() * np.sin(nmsh[1].ravel()) ]).T, np.ones((n[0] * n[1],), dtype=float)
  
class SquareSector2D(Domain):
  def __init__(self, rmin : float = 1.0, rmax : float = 2.0, tmin : float = 0.0, tmax : float = 0.5 * np.pi, window : str = 'zeroth_order', delta : float = 1E-2, quadrule : str = 'equispaced'):
    super().__init__(ndim = 1, delta = delta, window = window, quadrule = quadrule)
    self.rmin = rmin
    self.rmax = rmax
    self.width = np.array([ 2 * rmax, 2 * rmax ])
    self.tmin = tmin
    self.tmax = tmax
    self.tctr = 0.5 * (tmax + tmin)
    self.twdt = (tmax - tmin)
    self.wtctr = np.array([ np.cos(self.tctr), np.sin(self.tctr) ])
    self.delta_min = 1 - self.rmin / (self.rmin + self.delta * (self.rmax - self.rmin))
    self.delta_max = self.delta * (self.rmax - self.rmin) / self.rmax

  @property
  def delta(self) -> float:
    return self._delta

  @delta.setter
  def delta(self, s : float) -> None:
    self._delta = s
    self.delta_min = 1 - self.rmin / (self.rmin + self.delta * (self.rmax - self.rmin))
    self.delta_max = self.delta * (self.rmax - self.rmin) / self.rmax

  def eval_window(self, w : NDArray) -> NDArray:
    tw = np.acos(np.abs(w.T[0] * self.wtctr[0] + w.T[1] * self.wtctr[1]) / (1E-12 + np.linalg.norm(w, axis=1, ord=2)))
    if self.window == 'zeroth_order':
      return (zeroth_order_window_2d(w / self.rmax) - zeroth_order_window_2d(w / self.rmin)) * zeroth_order_window_1d(2.0 * tw / self.twdt)
    if self.window == 'first_order':
      return (first_order_window_2d(w / self.rmax, self.delta_max) - first_order_window_2d(w / self.rmin, self.delta_min)) * first_order_window_1d(2.0 * tw / self.twdt, self.delta)
    return NotImplementedError(f'{self.window} not implemented.')
  
  def sample(self, n : int) -> NDArray:
    rs = self.rmin + (self.rmax - self.rmin) * np.random.rand(n)
    bs = np.floor(4 * np.random.rand(n))
    ws = rs * (1j ** bs) + (-self.rmin + (self.rmin + self.rmax) * np.random.rand(n)) * (1j ** (bs + 1)) 
    ws = np.vstack([ ws.real.ravel(), ws.imag.ravel() ], dtype=float).T
    # rejection sampling
    stop = False
    while not stop:
      tw = np.acos(np.abs(ws.T[0] * self.wtctr[0] + ws.T[1] * self.wtctr[1]) / (1E-12 + np.linalg.norm(ws, axis=1, ord=2)))
      ws = ws[zeroth_order_window_1d(2.0 * tw / self.twdt) > 0]
      print(ws.size)
      if ws.size > n:
        stop = True
        ws = ws[:n]
      else:
        rs = self.rmin + (self.rmax - self.rmin) * np.random.rand(n)
        bs = np.floor(4 * np.random.rand(n))
        _ws = rs * (1j ** bs) + (-self.rmin + (self.rmin + self.rmax) * np.random.rand(n)) * (1j ** (bs + 1)) 
        _ws = np.vstack([ _ws.real.ravel(), _ws.imag.ravel() ], dtype=float).T
        ws = np.vstack([ ws, _ws ], dtype=float)
    return ws
  
  def quadrature(self, n):
    if isinstance(n, int):
      n = [ n, n ]
    rs = np.linspace(self.rmin, self.rmax, n[0])
    ts = np.hstack([ np.linspace(self.tmin, self.tmax, n[1] // 2), np.linspace(self.tmin + np.pi, self.tmax + np.pi, n[1] - n[1] // 2) ])
    msh = np.meshgrid(rs, ts)
    ws = np.vstack([ np.cos(msh[1].ravel()), np.sin(msh[1].ravel()) ], dtype=float)

    return (ws * (msh[0].ravel() / np.linalg.norm(ws, axis=0, ord=np.inf))).T, np.ones((n[0] * n[1],), dtype=float)
  
class Sector2D(Domain):
  def __init__(self, rmin : float = 1.0, rmax : float = 2.0, tmin : float = 0.0, tmax : float = 0.5 * np.pi, window : str = 'zeroth_order', delta : float = 1E-2, quadrule : str = 'equispaced'):
    super().__init__(ndim = 1, delta = delta, window = window, quadrule = quadrule)
    self.rmin = rmin
    self.rmax = rmax
    self.width = np.array([ 2 * rmax, 2 * rmax ])
    self.tmin = tmin
    self.tmax = tmax
    self.tctr = 0.5 * (tmax + tmin)
    self.twdt = (tmax - tmin)
    self.wtctr = np.array([ np.cos(self.tctr), np.sin(self.tctr) ])
    self.delta_min = 1 - self.rmin / (self.rmin + self.delta * (self.rmax - self.rmin))
    self.delta_max = self.delta * (self.rmax - self.rmin) / self.rmax

  @property
  def delta(self) -> float:
    return self._delta

  @delta.setter
  def delta(self, s : float) -> None:
    self._delta = s
    self.delta_min = 1 - self.rmin / (self.rmin + self.delta * (self.rmax - self.rmin))
    self.delta_max = self.delta * (self.rmax - self.rmin) / self.rmax

  def eval_window(self, w : NDArray) -> NDArray:
    tw = np.acos(np.abs(w.T[0] * self.wtctr[0] + w.T[1] * self.wtctr[1]) / (1E-12 + np.linalg.norm(w, axis=1, ord=2)))
    if self.window == 'zeroth_order':
      return (zeroth_order_window_1d(np.linalg.norm(w, axis=1, ord=2) / self.rmax) - zeroth_order_window_1d(np.linalg.norm(w, axis=1, ord=2) / self.rmin)) * zeroth_order_window_1d(2.0 * tw / self.twdt)
    if self.window == 'first_order':
      return (first_order_window_1d(np.linalg.norm(w, axis=1, ord=2) / self.rmax, self.delta_max) - first_order_window_1d(np.linalg.norm(w, axis=1, ord=2) / self.rmin, self.delta_min)) * first_order_window_1d(2.0 * tw / self.twdt, self.delta)
    return NotImplementedError(f'{self.window} not implemented.')
  
  def sample(self, n : int) -> NDArray:
    ws = np.sqrt((self.rmax ** 2 - self.rmin ** 2) * np.random.rand(n) + self.rmin ** 2) * np.exp(1j * (self.tmin + (self.tmax - self.tmin) * np.random.rand(n))) * np.sign(np.random.rand(n) - 0.5)
    return np.vstack([ ws.real, ws.imag ], dtype=float).T
  
  def quadrature(self, n):
    if isinstance(n, int):
      n = [ n, n ]
    nmsh = np.meshgrid(np.linspace(self.rmin, self.rmax, n[0]), np.hstack([ np.linspace(self.tmin, self.tmax, n[1] // 2), np.linspace(self.tmin + np.pi, self.tmax + np.pi, n[1] - n[1] // 2) ]))
    return np.vstack([ nmsh[0].ravel() * np.cos(nmsh[1].ravel()), nmsh[0].ravel() * np.sin(nmsh[1].ravel()) ]).T, np.ones((n[0] * n[1],), dtype=float)