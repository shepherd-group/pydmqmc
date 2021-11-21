#!/usr/bin/env python

import numpy as np
import pandas as pd
import functions as fn

# "QMC" & System parameters
seed = 7
np.random.seed(seed)
init_shift = 0.0
cycles = 10
target = 25
tau = 0.001
reports = int(target/(tau*cycles))
beta_loops = 25
Nattempts = int(1E4)
init = 'uniform-uniform'
Ntarget = 1E4
variable_shift = True
n_add = 3.0
zeta = 0.05
cutoff = 0.1

# Data Saving
data = []
csvname = 'INITIATOR-STR-H6'

for betaloop in range(1,beta_loops+1):
    print(' Beta Loop:', betaloop)

    iteration, report, shift = 0, 0, init_shift*1.0
    H, Heval, HS = fn.system_initialize('STRETCHED-H6-STO3G.hamil', shift)
    d, occrows, df = fn.initialize_dm(init, Nattempts, target, Heval, HS)
    df = fn.write_report(iteration, tau, shift, d, Heval, df=df, stdout=True)

    for report in range(reports):
    
        for cycle in range(cycles):
            iteration += 1
            '''
                Initiator approximation:
                    1) Some threshold "n_add" that we will allow to spawn
                       regardless unaffected. IE if |rho_ij| >= n_add.
                    2) let the elements |rho_ij| < n_add spawn when their
                        n_spawn -> rho_ij != 0.0
                    2) let the elements rho_ij sp
                    2) Count the non-initiator spaces failed spawns.
            '''
            # Spawn initiator first
            absd = np.abs(d)
            dinit = np.copy(d)
            dinit[absd<n_add] = 0.0
            init_spawn = -(tau/2)*(H @ dinit + dinit @ H)
            init_spawn = fn.stochastic_round(init_spawn, threshold=cutoff)

            # Then do the non-initiators
            dnoninit = np.copy(d)
            dnoninit[absd>=n_add] = 0.0
            noninit_spawn = -(tau/2)*(H @ dnoninit + dnoninit @ H)
            noninit_spawn = fn.stochastic_round(noninit_spawn, threshold=cutoff)
            noninit_spawn[d==0.0] = 0.0
            d += (init_spawn + noninit_spawn)

        df = fn.write_report(iteration, tau, shift, d, Heval, df=df, stdout=True)

        nw = df['Nw'][-1]
        nw_old = df['Nw'][-2]
        if variable_shift or nw > Ntarget:
            H, shift = fn.update_shift(shift, nw, nw_old, zeta, tau, cycles, H, HS)
            variable_shift = True

    data = fn.store_data(data, df, betaloop, beta_loops, csvname, '')

