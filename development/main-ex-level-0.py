#!/usr/bin/env python

import sys
import pickle
import numba as nb
import numpy as np
import pandas as pd
from integrals_readin import integral_system as readin 


#@nb.njit
def set_numba_seed(rng):
    np.random.seed(rng)


@nb.njit
def propagate(dt, p, H, S, nadd, cutoff, ilvl, flvl):
    ''' Run `DMQMC` using the asymmetric Bloch equation:

            dp = -dt p H

            dp_ij = -dt \sum_k p_ik H_kj

    Factoring out the diagonal contribution, and including the shift we have:

            dp_ij = p_ij * (1 + dt * (S - H_jj)) - dt \sum_{k != j} p_ik H_kj

    With the initiator approximation we limit those p_ik which spawn to:

            |p_ik| >= n_add (nadd)

    As well, we only store |p_ij| > 1.0, otherwise we round those below this
    threshold in a non-biased manner. To be more correct, while spawning, we
    only accumulate those spawns |dp_ij| > cutoff.

    Update 2022-04-23:
        Added ilvl and flvl as parameters.
        ilvl is the initiator_level in the traditional sense, but really
        only turns on initiator level 0.
        flvl is similar to ilvl, but turns on allowed spawning to the
        initiator level 0 regardless of its population (from any site).
    '''
    dets = p.shape[0]
    dp = np.zeros(p.shape, dtype=nb.float64) + p

    for i in range(dets):
        for j in range(dets):

            dp[i,j] -= p[i,j] * dt * (H[j,j] - H[0,0] - S[i])

            p_ij = abs(p[i,j])

            for k in range(dets):

                if k == j:
                    continue

                ichk1 = ilvl and i == k
                ichk2 = flvl and i == j

                if abs(p[i,k]) >= nadd or p_ij != 0.0 or ichk1 or ichk2:
                    pr = -dt * p[i,k] * H[k,j]

                    if abs(pr) < cutoff:
                        pr /= cutoff
                        pr += np.sign(pr) * np.random.random()
                        pr = np.trunc(pr)
                        pr *= cutoff

                    dp[i,j] += pr

            # This adjustment happens to each element
            # after each element has been updated.
            if abs(dp[i,j]) < 1.0:
                sign = np.sign(dp[i,j])
                pr = dp[i,j] + sign * np.random.random()
                dp[i,j] = np.trunc(pr)

    # np.where(np.abs(dp) < 1.0,
    #          np.trunc(dp + np.sign(dp)*np.random.random(dp.shape)),
    #          dp)

    return dp # minomer; is actually updated p


def estimates(p, H, S, onw, A, zt, dt, rbr):
    #estimates(p, H, S, nw, ncycles, zeta, tau, rbr)
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


def initialize(H, initial): # uniform-uniform from functions::initialize_dm
    p = np.random.choice(H.shape[0], size=initial)
    p = np.bincount(p, minlength=H.shape[0]).astype(np.float64)
    return np.diag(p)


def report(b, S, tr, en, nw, oc):
    print(f' {b:>9.3f}  {S.mean():> 18.12E}  {tr:> 18.12E}  '
          f'{en:> 18.12E}  {nw.sum():>18.12E}  {oc:>6}  '
          f'{en/tr:> 18.12f}')


def main(clargs):

    rng, do_rbr, do_save, do_init, do_ilvl, do_flvl = [int(k) for k in clargs]

    print(rng, do_rbr, do_save, do_init, do_ilvl, do_flvl)

    np.random.seed(rng)
    set_numba_seed(rng)

    df = {'beta': [], 'shift': [], 'trace': [], 'pH': [], 'nw': []}

    sys = readin(
            int_file = 'tests/inputs/integrals/STRICT-STO3G-STR-H4.FCIDUMP',
            hamiltonian = True,
        )

    H = np.copy(sys.H)

    nadd = 3.0 if do_init == 1 else 0.0
    spawn_cutoff = 0.01
    tau = 0.001
    ncycles = 1000
    final_beta = 25.0
    nreports = int(final_beta/(tau*ncycles))
    print(nreports)
    zeta = 0.05
    initial_particles = int(float(1E5))
    row_by_row = True if do_rbr == 1 else False
    rbr = 1 if row_by_row else None
    ilevel = True if do_ilvl else False
    flevel = True if do_flvl else False

    p = initialize(H, initial_particles)
    np.save("initial_dm.npy", np.diag(p))
    S = np.zeros(H.shape[0], dtype=np.float64)
    nw = np.sum(p, axis=rbr)

    S, tr, en, nw, oc = estimates(p, H, S, nw, ncycles, zeta, tau, rbr)
    df = store_row_data(p, H, S, df, 0.0)
    print(f" {'Beta':>9}  {'Mean Shift':>18}  {'Trace':>18}  "
          f"{'Energy?':>18}  {'NW? Sum':>18}  {'OC?':>6}  "
          f"{'en/tr':>18}")
    report(0, S, tr, en, nw, oc)

    for irep in range(nreports):

        for icyc in range(ncycles):

            p = propagate(tau, p, H, S, nadd, spawn_cutoff, ilevel, flevel)

        beta = (irep+1)*ncycles*tau
        df = store_row_data(p, H, S, df, beta)
        S, tr, en, nw, oc = estimates(p, H, S, nw, ncycles, zeta, tau, rbr)
        report(beta, S, tr, en, nw, oc)

    print(np.trace(p))

    if do_save == 1:
        srbr = '1' if rbr == 1 else 0
        sini = '1' if nadd == 3.0 else 0
        silvl = '1' if ilevel == True else 0
        sflvl = '1' if flevel == True else 0
        pklf = 'rerandom-stretched-H4-row-data'
        pklf += f'-ilvl{silvl}-flvl{sflvl}-initiator{sini}-rbr{srbr}-rng{rng}.pickle'
        with open(pklf, 'wb') as handle:
            pickle.dump(df, handle, protocol=4)


if __name__ == '__main__':
    main(sys.argv[1:])
