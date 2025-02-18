# Adaptive multipliers for extrapolation in frequency

Numerical implementation of a method to find an optimal multiplier to perform extrapolation in the frequency domain. The method assumes that the Fourier transforms of functions in a finite collection $\mathcal{{U}}$ are known on a domain $\Omega_0$. Given an extrapolation factor $\alpha > 1$ the method finds an optimal multiplier, in a suitable sense, that extrapolates the Fourier transform to $\alpha\Omega_0$. 

The optimal multipliers have a canonical structure solely determined by a positive semidefinite Hermitian matrix $\Sigma$. We call these $\Sigma$-multipliers. An inexact Krasnoselskii-Mann fixed point iteration is used to find the matrix $\Sigma^{\star}$ associated to the optimal multiplier.

Version 1.0

Date: 02.02.2025

## References

The motivation behind the method, the algorithms, and the proofs for the theoretical results can be found in:

> D. Castelli Lacunza and C. A. Sing Long, "*Adaptive multipliers for frequency extrapolation*." January 28, 2025. [arXiv:2501.17019](https://doi.org/10.48550/arXiv.2501.17019).

The following manuscripts have built upon these ideas.

> D. Castelli Lacunza and C. A. Sing Long, "*Translation-invariant extrapolation in frequency using adaptive multipliers*." January 31, 2025.

The code reproducing the results of their numerical experiments can be found on ``experiments``.

### Citation

You can use the following code to cite our work.

```bibtex
@misc{lacunza_adaptive_2025,
	title     = {Adaptive multipliers for extrapolation in frequency},
	doi       = {10.48550/arXiv.2501.17019},
	number    = {{arXiv}:2501.17019},
	publisher = {{arXiv}},
	author    = {Castelli Lacunza, Diego and Sing Long, Carlos A.},
	date      = {2025-01-28}}
```

## Repository

### Dependencies

The dependencies for the Python implementation are:
* ``numpy``
* ``scipy``
* ``scikit-learn``
* ``matplotlib``

Some numerical experiments access the [MNIST dataset](https://yann.lecun.com/exdb/mnist/) using ``scikit-learn``. 

### Structure

``adaptiveExtrapolationInFrequency\``
* ``objects\``
    * ``domain.py``: objects defining choices for the low-frequency domain $\Omega_0$
    * ``multiplier.py``: objects defining $\Sigma$-multipliers
    * ``projector.py``: objects implementing orthogonal projectors for the fixed-point iteration
    * ``solver.py``: objects implementing the fixed-point iteration
* ``routines\``
    * ``collections.py``: definition of collections $\mathcal{{U}}$
    * ``integrals.py``: computation of the inverse Fourier transform
    * ``mra.py``: functions required to construct a basis for a multiresolution analysis

``experiments\``
  * ``arXiv.2501.17019\``: experiments in "*Adaptive multipliers for frequency extrapolation*"
    * ``FIGS\``: folder in which the figures will be saved
    * ``01_MNIST.ipynb``: extrapolation in frequency for MNIST image data
    * ``02_MNIST.ipynb``: extrapolation in frequency for MNIST image data with horizontal/vertical multipliers
    * ``03_MRA_1D.ipynb``: recovery of a refinable function using $\Sigma$-multipliers
    * ``04_MRA_1D_SMultiplier.ipynb``: construction of a multiresolution from an optimal multiplier
  * ``conference\``: experiments in "*Translation-invariant extrapolation in frequency using adaptive multipliers*"
    * ``FIGS\``: folder in which the figures will be saved
    * ``01_1D_Translations.ipynb``: translation-invariant MSE optimal multipliers in 1D
 