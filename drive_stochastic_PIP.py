#!/usr/bin/env python

import numpy as np
import pandas as pd
import functions as fn

# The system
hamil_file = '../../STRETCHED-H6-STO3G.hamil'
H, Heval, HS, H0 = fn.system_initialize(hamil_file, 0, ip=True)

# "QMC" paramters
shift = 0
cycles = 10
pip_target = 1.0
target = 10.0
tau = 0.0001
zeta = 0.05
reports = int(target/(tau*cycles))
beta_loops = 50
Nattempts = int(1E4)
init = 'thermal-uniform'
propagator = 'IPDMQMC'
stoch_rounding = True
cutoff = 0.01
variable_shift = True
exact_diagonals = False

# Data Saving
data = []
csvname = 'STR-H6-PIP-Analytical-Tau1En4'
seed = 27
np.random.seed(seed)

for betaloop in range(1,beta_loops+1):
    propagator = 'IPDMQMC'
    print(' # Beta Loop:', betaloop)

    N_iter, report, shift = 0, 0, 0
    d, occrows, df = fn.initialize_dm(init, Nattempts, pip_target, Heval, HS)

    if exact_diagonals:
        occrows, occ_ind = fn.get_occupied_rows(d)
        w = np.exp(-pip_target*np.diag(H0))
        for ind in occ_ind:
            d[ind,ind] = w[ind]
        d *= (Nattempts/d.trace())

    df = fn.write_report(N_iter, tau, shift, d, Heval, df=df, stdout=True)

    for report in range(reports):

        for cycle in range(cycles):
            N_iter += 1

            dd = -tau*(d @ H)

            if propagator == 'IPDMQMC':
                dd += tau*(H0 @ d)
            elif  propagator == 'aDMQMC':
                pass

            if stoch_rounding:
                if exact_diagonals:
                    ddiagonal = np.diag(np.diag(dd))
                    dd -= ddiagonal
                else:
                    ddiagonal = 0

                dd = fn.stochastic_round(dd, cutoff) + ddiagonal


            d += dd

        df = fn.write_report(N_iter, tau, shift, d, Heval, df=df, stdout=True)

        if variable_shift:
            nw_old, nw = df['Nw'][-2:] 
            H, shift = fn.update_shift(shift, nw, nw_old, zeta, tau, cycles, H, HS)

        if df['Beta'][-1] == pip_target:
            H = H + np.eye(HS)*shift
            df['Shift'][-1] = shift = 0.0
            print(' # Changing propagators from IP-DMQMC to Asymmetric DMQMC!')
            propagator = 'aDMQMC'

    data = fn.store_data(data, df, betaloop, beta_loops, csvname, '')

