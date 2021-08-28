#!/usr/bin/env python

import numpy as np
import pandas as pd
import functions as fn

# "QMC" & System parameters
seed = 12
np.random.seed(seed)
shift = 0.0
cycles = 1
#target = 10
target = 10
tau = 0.1
reports = int(target/(tau*cycles))
beta_loops = 1
#Nattempts = int(1E4)
Nattempts = int(3E2)
init = 'uniform-uniform'
#init = 'deterministic-uniform'
Ntarget = 1E8
variable_shift = True
#variable_shift = False
n_add = 3.0
zeta = 0.05
cutoff = 0.1
#cutoff = 1.0
new_intitiator = False

#H, Heval, HS = fn.system_initialize('STRETCHED-H6-STO3G.hamil', shift)

# Data Saving
data = []
csvname = 'STR-H6-STO3G-INITIATOR-FAIL-3E2-ILVn1-'

for betaloop in range(1,beta_loops+1):
    print(' Beta Loop:', betaloop)

    iteration, report, shift = 0, 0, 0.0
    H, Heval, HS = fn.system_initialize('STRETCHED-H6-STO3G.hamil', shift)
    d, occrows, df = fn.initialize_dm(init, Nattempts, target, Heval, HS)
    df = fn.write_report(iteration, tau, shift, d, Heval, df=df, stdout=True)
    nfail = np.zeros((HS,HS))

    for report in range(reports):
    
        for cycle in range(cycles):
            iteration += 1
            '''
                Initiator approximation:
                    1) Some threshold that we will allow to spawn regardless.
                    2a) let the elements rho_ij < n_add spawn when their
                        n_spawn > n_add
                    2b) let the elements rho_ij sp
                    2) Count the non-initiator spaces failed spawns.
            '''
            dd = np.zeros((HS,HS))
            occupied = np.transpose(np.nonzero(d))
            for (i, j) in occupied:
                    if not(new_intitiator):
                        nfail[i,j] = 0
                    #spawn = i == j
                    spawn = False
                    zero_mat = np.zeros((HS,HS))
                    zero_mat[i,j] = d[i,j]
                    tau_factor = 1.0

                    scaled_population = np.abs(d[i,j])*(nfail[i,j] + 1.0)
                    if scaled_population >= n_add:
                        spawn = True
                        tau_factor = nfail[i,j] + 1.0
                        nfail[i,j] = 0.0

                    if spawn:
                        tmp_dd = -(tau/2.0)*(H @ zero_mat + zero_mat @ H)
                        tmp_dd[d == 0.0] *= tau_factor
                        dd += fn.stochastic_round(tmp_dd, threshold=cutoff)
                    else:
                        tmp_dd = -(tau/2)*(H @ zero_mat + zero_mat @ H)
                        spawned_count = np.count_nonzero(tmp_dd)
                        zerod_count = np.count_nonzero(d[tmp_dd != 0.0] == 0.0)
                        nfail[i,j] += zerod_count/spawned_count
                        tmp_dd[d == 0.0] = 0.0
                        dd += fn.stochastic_round(tmp_dd, threshold=cutoff)

                    dd = fn.stochastic_round(dd, threshold=cutoff)
            d += dd

        df = fn.write_report(iteration, tau, shift, d, Heval, df=df, stdout=True)

        nw = df['Nw'][-1]
        nw_old = df['Nw'][-2]
        if variable_shift or nw > Ntarget:
            H, shift = fn.update_shift(shift, nw, nw_old, zeta, tau, cycles, H, HS)
            variable_shift = True

    data = fn.store_data(data, df, betaloop, beta_loops, csvname, '')

