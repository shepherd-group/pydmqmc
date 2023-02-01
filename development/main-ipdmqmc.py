#!/usr/bin/env python

import sys
import pickle
import numba as nb
import numpy as np
import pandas as pd
from integrals_readin import integral_system as readin 
from scipy.optimize import newton


@nb.njit
def set_numba_seed(rng):
    np.random.seed(rng+10000)


@nb.njit
def propagate(dt, p, H, S, nadd, cutoff, ilvl, flvl):
    ''' Run `IP-DMQMC` using the equation:

            dp = -dt [ -H0 p + p H ]

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

            dp[i,j] -= p[i,j] * dt * (-H[i,i] + H[j,j] - S[i])

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


def initialize(H, initial, bars, eig, nel, tb, bitints, spawn_cutoff):
    ''' In IP-DMQMC we initialize to exp(-\\beta_T H^{(0)}), which is done by
    sampling exp(-\\beta_T H`), where H` = \sum_{|D>} \epsilon_i, then we
    re-weight based on the difference of exp(-\\beta_T (H^{(0)} - H`)).
    '''
    def __fermi_function(mu_ff, ei_ff, tb_ff):
        return 1.0/(np.exp(tb_ff*(ei_ff - mu_ff)) + 1.0)

    def __dnav_function(mu_df, ei_df, tb_df, nel_df):
        return nel_df - __fermi_function(mu_df, ei_df, tb_df).sum()

    mu0 = eig[np.argsort(eig)][nel-1:nel+1].sum()/2.0
    mu = newton(__dnav_function, mu0, args=(eig, tb, nel), tol=1E-14)
    fi = __fermi_function(mu, eig, tb)
    print(f' \\beta_T = {tb:>18.12f}')
    print(f' \\mu = {mu:>18.12f}')

    @nb.njit
    def __gci(bars_gci, initial_gci, norb_gci, fi_gci, nalpha_gci, nbeta_gci,
              bitints_gci, hii_gci, ei_gci, tb_gci, eshift_gci, cutoff_gci):
        nspawned = 0
        nel_gci = nalpha_gci + nbeta_gci
        rho0_gci = np.zeros(bars_gci.shape[0], dtype=nb.float64)

        while nspawned < initial_gci:
            ba = np.zeros(norb_gci, dtype=nb.int64)
            nsel, nsela, nselb, bitint = 0, 0, 0, 0

            for iorb in range(norb_gci):
                if np.random.random() < fi_gci[iorb]:
                    occ = 1
                    ba[iorb] = occ
                    nsel += occ
                    nsela += int(occ*(iorb % 2 == 1))
                    nselb += int(occ*(iorb % 2))
                    bitint += int(occ*(2**iorb))

                if nsel > nel_gci or nsela > nalpha_gci or nselb > nbeta_gci:
                    allowed = False
                    bitint = -1
                    break
                else:
                    allowed = nsel == nel_gci

            if allowed and bitint in bitints:
                energy = hii_gci[bitints_gci == bitint][0]
                energy -= ei_gci[ba == 1].sum()
                ps = np.exp(-tb_gci*(energy - eshift_gci))/cutoff_gci
                ps += np.random.random()
                ps = np.trunc(ps)
                ps *= cutoff_gci

                rho0_gci[bitints_gci == bitint] += ps
                nspawned += int(ps)

        return np.trunc(rho0_gci + np.random.random(rho0_gci.shape[0]))

    nalpha = int(nel/2)
    nbeta = int(nel/2)
    norb = bars[0].shape[0]
    hii = np.copy(np.diag(H))
    eshift = hii[0] - eig[np.argsort(eig)][:nel].sum()

    rho0 = __gci(bars, initial, norb, fi, nalpha, nbeta, bitints, hii,
                 eig, tb, eshift, spawn_cutoff)

    return np.diag(rho0)


def report(b, S, tr, en, nw, oc):
    print(f' {b:>9.3f}  {S.mean():> 18.12E}  {tr:> 18.12E}  '
          f'{en:> 18.12E}  {nw.sum():>18.12E}  {oc:>6}  '
          f'{en/tr:> 18.12f}')


def main(clargs):

    rng, betaT, do_rbr, do_save, do_init, do_ilvl, do_flvl = [int(k) for k in clargs]
    rng = int(rng*26) + betaT
    rng += 1000

    np.random.seed(rng)
    set_numba_seed(rng)

    df = {'beta': [], 'shift': [], 'trace': [], 'pH': [], 'nw': []}

    sys = readin(
            int_file = 'STRICT-EIGENVALUES-STO3G-STR-H6.FCIDUMP',
            hamiltonian = True,
        )

    H = np.copy(sys.H)
    bars = np.copy(sys.bitarrays)
    eigs = np.copy(sys.eig)
    nel = sys.nel
    bitints = [np.exp2(sys.orbs[ba==1]).sum() for ba in sys.bitarrays]
    bitints = np.array(bitints).astype(np.int64)

    nadd = 3.0 if do_init == 1 else 0.0
    spawn_cutoff = 0.01
    tau = 0.001
    ncycles = 10
    final_beta = betaT
    nreports = int(final_beta/(tau*ncycles))
    zeta = 0.05
    initial_particles = int(float(1E5))
    row_by_row = True if do_rbr == 1 else False
    rbr = 1 if row_by_row else None
    ilevel = True if do_ilvl else False
    flevel = True if do_flvl else False

    p = initialize(H, initial_particles, bars, eigs, nel, final_beta,
                   bitints, spawn_cutoff)
    S = np.zeros(H.shape[0], dtype=np.float64)
    nw = np.sum(p, axis=rbr)
    nw *= initial_particles/(nw.sum())

    S, tr, en, nw, oc = estimates(p, H, S, nw, ncycles, zeta, tau, rbr)
    df = store_row_data(p, H, S, df, 0.0)
    report(0, S, tr, en, nw, oc)

    for irep in range(nreports):

        for icyc in range(ncycles):

            p = propagate(tau, p, H, S, nadd, spawn_cutoff, ilevel, flevel)

        beta = (irep+1)*ncycles*tau
        df = store_row_data(p, H, S, df, beta)
        S, tr, en, nw, oc = estimates(p, H, S, nw, ncycles, zeta, tau, rbr)
        report(beta, S, tr, en, nw, oc)

    if do_save == 1:
        srbr = '1' if rbr == 1 else 0
        sini = '1' if nadd == 3.0 else 0
        silvl = '1' if ilevel == True else 0
        sflvl = '1' if flevel == True else 0
        pklf = f'stretched-H6-row-data-ipdmqmc-betaT{final_beta}-'
        pklf += f'ilvl{silvl}-flvl{sflvl}-initiator{sini}-rbr{srbr}-rng{rng}.pickle'
        with open(pklf, 'wb') as handle:
            pickle.dump(df, handle, protocol=4)


if __name__ == '__main__':
    main(sys.argv[1:])
