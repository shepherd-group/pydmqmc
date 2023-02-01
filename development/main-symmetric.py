#!/usr/bin/env python

import sys
import pickle
import numba as nb
import numpy as np
import pandas as pd
from integrals_readin import integral_system as readin 


@nb.njit
def set_numba_seed(rng):
    np.random.seed(rng+10000)


@nb.njit
def propagate(dt, p, H, S, nadd, cutoff):
    ''' Run `DMQMC` using the asymmetric Bloch equation:

            dp = -dt p H

            dp_ij = -dt \sum_k p_ik H_kj

    Factoring out the diagonal contribution, and including the shift we have:

            dp_ij = p_ij * (1 + dt * (S - H_jj)) - dt \sum_{k != j} p_ik H_kj

    With the initiator approximation we limit those p_ik which spawn to:

            |p_ik| >= n_add (nadd)

    As well, we only store |p_ij| > 1.0, otherwise we round those below this
    threshold in a non-biased manner. To be more correct, while spawning, we
    only accumulate those spawns |dp_ik| > cutoff.
    '''
    dets = p.shape[0]
    dp = np.zeros(p.shape, dtype=nb.float64) + p

    for i in range(dets):
        for j in range(dets):

            dp[i,j] -= p[i,j] * (dt/2) * (H[i,i] - H[0,0] - S[i])
            dp[i,j] -= p[i,j] * (dt/2) * (H[j,j] - H[0,0] - S[i])

            p_ij = abs(p[i,j])

            for k in range(dets):

                if k != j and (abs(p[i,k]) >= nadd or p_ij != 0.0):
                    pr = -(dt/2) * p[i,k] * H[k,j]

                    if abs(pr) < cutoff:
                        pr /= cutoff
                        pr += np.sign(pr) * np.random.random()
                        pr = np.trunc(pr)
                        pr *= cutoff

                    dp[i,j] += pr

                if k != i and (abs(p[k,j]) >= nadd or p_ij != 0.0):
                    pr = -(dt/2) * H[i,k] * p[k,j]

                    if abs(pr) < cutoff:
                        pr /= cutoff
                        pr += np.sign(pr) * np.random.random()
                        pr = np.trunc(pr)
                        pr *= cutoff

                    dp[i,j] += pr

            if abs(dp[i,j]) < 1.0:
                sign = np.sign(dp[i,j])
                pr = dp[i,j] + sign * np.random.random()
                dp[i,j] = np.trunc(pr)

    return dp


def estimates(p, H, S, onw, A, zt, dt, rbr):

    nw = np.abs(p).sum(axis=rbr)
    if rbr is not None:
        for i in range(p.shape[0]):
            if nw[i] != 0.0 and onw[i] != 0.0:
                S[i] -= (zt/(A*dt))*np.log(nw[i]/onw[i])
    else:
        S -= (zt/(A*dt))*np.log(nw/onw)

    tr = p.trace()
    en = (p @ H).trace()
    oc = np.count_nonzero(p)

    return S, tr, en, nw, oc


def store_row_data(p, H, S, df, b):
    tr = np.diag(p)
    en = np.diag(p @ H)
    nw = np.abs(p).sum(axis=1)
    df['beta'].append(b)
    df['shift'].append(np.copy(S))
    df['trace'].append(np.copy(tr))
    df['pH'].append(np.copy(en))
    df['nw'].append(np.copy(nw))
    return df


def initialize(H, initial):
    p = np.random.choice(H.shape[0], size=initial)
    p = np.bincount(p, minlength=H.shape[0]).astype(np.float64)
    return np.diag(p)


def report(b, S, tr, en, nw, oc):
    print(f' {b:>9.3f}  {S.mean():> 18.12E}  {tr:> 18.12E}  '
          f'{en:> 18.12E}  {nw.sum():>18.12E}  {oc:>6}  '
          f'{en/tr:> 18.12f}')


def main(clargs):

    rng, do_rbr, do_save, do_init = [int(k) for k in clargs]
    rng += 100

    np.random.seed(rng)
    set_numba_seed(rng)

    df = {'beta': [], 'shift': [], 'trace': [], 'pH': [], 'nw': []}

    sys = readin(
            int_file = 'STRICT-EIGENVALUES-STO3G-STR-H6.FCIDUMP',
            hamiltonian = True,
        )

    H = np.copy(sys.H)

    nadd = 3.0 if do_init == 1 else 0.0
    spawn_cutoff = 0.01
    tau = 0.001
    ncycles = 10
    final_beta = 25.0
    nreports = int(final_beta/(tau*ncycles))
    zeta = 0.05
    initial_particles = int(float(1E5))
    row_by_row = True if do_rbr == 1 else False
    rbr = 1 if row_by_row else None

    p = initialize(H, initial_particles)
    S = np.zeros(H.shape[0], dtype=np.float64)
    nw = np.sum(p, axis=rbr)

    S, tr, en, nw, oc = estimates(p, H, S, nw, ncycles, zeta, tau, rbr)
    df = store_row_data(p, H, S, df, 0.0)
    report(0, S, tr, en, nw, oc)

    for irep in range(nreports):

        for icyc in range(ncycles):

            p = propagate(tau, p, H, S, nadd, spawn_cutoff)

        beta = (irep+1)*ncycles*tau
        df = store_row_data(p, H, S, df, beta)
        S, tr, en, nw, oc = estimates(p, H, S, nw, ncycles, zeta, tau, rbr)
        report(beta, S, tr, en, nw, oc)

    if do_save == 1:
        srbr = '1' if rbr == 1 else 0
        pklf = f'stretched-H6-sym-row-data-initiator-rbr{srbr}-rng{rng}.pickle'
        with open(pklf, 'wb') as handle:
            pickle.dump(df, handle, protocol=4)


if __name__ == '__main__':
    main(sys.argv[1:])
