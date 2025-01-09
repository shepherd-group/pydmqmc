#!/usr/bin/env python

import numba as nb
import numpy as np


@nb.njit
def set_numba_seed(rng: int):
    np.random.seed(rng+10000)


@nb.njit
def propagate(dt, re, im, H, S, nadd, cutoff):
    r''' The Von Neumann equation is:

        dp/dt = -i {H,p}

    Which we can break down into a real and imaginary propagation.
    To do this, we write the density matrix as two separate matrices, one
    for the real, and the other for the imaginary component of the true
    density matrix.

        p = p + p^(i)

    Then combining this with the equation above we can find the evolution
    of the real and imaginary terms in the density matrix.

    Real:

        dp_ij = dt \sum_k (H_ik p^(i)_kj - p^(i)_ik H_kj)

    Imaginary:

        dp_ij^(i) = dt \sum_k (p_ik H_kj - H_ik p_kj)

    Note we assumed that the Hamiltonian is strictly real, if this is not
    true these equations will need to be re-derived to account for the
    additional terms.
    '''
    ndets = re.shape[0]
    dre = np.zeros(re.shape, dtype=nb.float64) + re
    dim = np.zeros(im.shape, dtype=nb.float64) + im

    for i in range(ndets):
        for j in range(ndets):

            dre[i,j] += dt * (H[i,i] * im[i,j] - im[i,j] * H[j,j])
            dim[i,j] += dt * (re[i,j] * H[j,j] - re[i,j] * H[i,i])
            #dre[i,j] += dt * ((H[i,i] - S) * im[i,j] - im[i,j] * (H[j,j] - S))
            #dim[i,j] += dt * (re[i,j] * (H[j,j] - S) - re[i,j] * (H[i,i] - S))

            for k in range(ndets):

                if (k != j):
                    psre = -dt * im[i,k] * H[k,j]

                    if 0.0 < abs(psre) < cutoff:
                        psre /= cutoff
                        psre += np.sign(psre) * np.random.random()
                        psre = np.trunc(psre)
                        psre *= cutoff

                    dre[i,j] += psre

                    psim = dt * re[i,k] * H[k,j]

                    if 0.0 < abs(psim) < cutoff:
                        psim /= cutoff
                        psim += np.sign(psim) * np.random.random()
                        psim = np.trunc(psim)
                        psim *= cutoff

                    dim[i,j] += psim

                if (k != i):
                    psre = dt * H[i,k] * im[k,j]

                    if 0.0 < abs(psre) < cutoff:
                        psre /= cutoff
                        psre += np.sign(psre) * np.random.random()
                        psre = np.trunc(psre)
                        psre *= cutoff

                    dre[i,j] += psre

                    psim = -dt * H[i,k] * re[k,j]

                    if 0.0 < abs(psim) < cutoff:
                        psim /= cutoff
                        psim += np.sign(psim) * np.random.random()
                        psim = np.trunc(psim)
                        psim *= cutoff

                    dim[i,j] += psim

            mod = (dre[i,j]**2.0 + dim[i,j]**2.0)**0.5

            if mod < 1.0:
                if mod + np.random.random() >= 1.0:
                    c = 1.0/mod
                    dre[i,j] *= c
                    dim[i,j] *= c
                else:
                    dre[i,j] = 0.0
                    dim[i,j] = 0.0

    return dre, dim
