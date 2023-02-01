#!/usr/bin/env python

import sys
import pickle
import numpy as np
import pandas as pd
from integrals_readin import integral_system as readin 


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
    df['shift'].append(S)
    df['trace'].append(tr)
    df['pH'].append(en)
    df['nw'].append(nw)
    return df


def report(b, S, tr, en, nw, oc):
    print(f' {b:>9.3f}  {S.mean():> 18.12E}  {tr:> 18.12E}  '
          f'{en:> 18.12E}  {nw.sum():>18.12E}  {oc:>6}  '
          f'{en/tr:> 18.12f}')


def main(clargs):

    rng, do_rbr, do_save = [int(k) for k in clargs]

    df = {'beta': [], 'shift': [], 'trace': [], 'pH': [], 'nw': []}

    sys = readin(
            int_file = 'STRICT-EIGENVALUES-STO3G-STR-H6.FCIDUMP',
            hamiltonian = True,
        )

    H = np.copy(sys.H)

    tau = 0.001
    ncycles = 10
    final_beta = 25.0
    nreports = int(final_beta/(tau*ncycles))
    zeta = 0.05
    initial_particles = int(float(1E5))
    row_by_row = True if do_rbr == 1 else False
    rbr = 1 if row_by_row else None

    p = np.eye(H.shape[0], dtype=np.float64) * (initial_particles / H.shape[0])
    S = np.zeros(H.shape[0], dtype=np.float64)
    nw = np.sum(p, axis=rbr)

    S, tr, en, nw, oc = estimates(p, H, S, nw, ncycles, zeta, tau, rbr)
    df = store_row_data(p, H, S, df, 0.0)
    report(0, S, tr, en, nw, oc)

    for irep in range(nreports):

        for icyc in range(ncycles):

            p -= tau * p @ (H - np.eye(H.shape[0])*H[0,0] - np.diag(S))

        beta = (irep+1)*ncycles*tau
        df = store_row_data(p, H, S, df, beta)
        S, tr, en, nw, oc = estimates(p, H, S, nw, ncycles, zeta, tau, rbr)
        report(beta, S, tr, en, nw, oc)

    if do_save == 1:
        srbr = '1' if rbr == 1 else 0
        pklf = f'stretched-H6-row-data-power-method-rbr{srbr}.pickle'
        with open(pklf, 'wb') as handle:
            pickle.dump(df, handle, protocol=4)


if __name__ == '__main__':
    main(sys.argv[1:])
