## ----------------------------------------------------------------------------
## Imports

import numpy as np
from numpy.typing import NDArray
from adaptiveExtrapolationInFrequency.objects.multiplier import SMultiplier
from adaptiveExtrapolationInFrequency.objects.projector import Projector

## ----------------------------------------------------------------------------
## Base class

class FixedPointIteration(object):
  def __init__(self):
    # parameters
    self.tG               = 1E-3
    self.tS               = 1E-3
    # options
    # - parameters
    self.theta            = 1.00
    self.tS               = 1E-3
    self.tG               = 1E-4
    # - termination criteria
    self.maxitns          = 100
    self.abs_tol          = 1E-6
    self.rel_tol          = 1E-9
    # - quadrature
    self.num_nodes_min    = 1000
    self.num_nodes_max    = 10000
    self.num_nodes_factor = 1.001
    # - display
    self.verbose          = False
    self.print_every    = 100
    # - log
    self.log = {
      'itns'      : 0,
      'x_nrm'     : 0,
      'dx_nrm'    : 0,
      'n_nodes'   : 0,
      'obj_x'     : 0
    }

  def solve(self, m : SMultiplier, P : Projector, S : NDArray = None) -> NDArray:
    # clear log
    self.log = {
      'itns'      : 0,
      'x_nrm'     : [],
      'dx_nrm'    : [],
      'num_nodes' : [],
      'obj_x'     : []
    }
    # init
    if not S:
      X = np.eye(m.size, dtype=complex)
    else:
      X = 0.5 * (np.conj(S).T + S)
    # iteration
    itn = 0
    itn_disp = 0
    # number of nodes
    num_nodes_real = self.num_nodes_min
    # main loop
    stop = False
    while not stop:
      itn += 1
      # norm of current iterate
      X_nrm = np.linalg.norm(X, ord='fro')
      # number of nodes
      # - this is to avoid overflow
      num_nodes_real = np.minimum(self.num_nodes_max, num_nodes_real * self.num_nodes_factor)
      num_nodes = np.minimum(self.num_nodes_max, np.maximum(self.num_nodes_min, int(num_nodes_real)))
      # evaluate Gram matrix of residuals
      G = m.eval_residual_gram(X, num_nodes)
      # K-M iteration
      Xp = (1 - self.theta) * X + P.proj(self.tS * X + self.tG * G)
      # norm of step
      dX_nrm = np.linalg.norm(Xp - X, ord='fro')
      # update
      X = Xp
      # stopping criteria
      stop = self.maxitns <= itn or dX_nrm <= self.abs_tol or dX_nrm <= X_nrm * self.rel_tol
      # log
      self.log['x_nrm'].append(X_nrm)
      self.log['dx_nrm'].append(dX_nrm)
      self.log['num_nodes'].append(num_nodes)
      self.log['obj_x'].append(P.eval(X))
      # display
      if self.verbose and (stop or itn == 1 or itn % self.print_every == 0):
        itn_disp += 1
        if itn == 1 or itn_disp % 10 == 0:
          print(f'{' itn':10s}|{' obj(m)':12s}|{' |x|':11s}|{' |dx|':11s}|{' nq':8s}')
        print(f'{itn:9d} | {P.eval(X):+.3E} | {X_nrm:.3E} | {dX_nrm:.3E} | {num_nodes:6d}')
    self.log['itns']    = itn
    self.log['x_nrm']   = np.array(self.log['x_nrm'])
    self.log['dx_nrm']  = np.array(self.log['dx_nrm'])
    self.log['num_nodes'] = np.array(self.log['num_nodes'])
    self.log['obj_x']   = np.array(self.log['obj_x'])
    return X
       