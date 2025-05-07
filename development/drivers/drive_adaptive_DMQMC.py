#!/usr/bin/env python

import numpy as np
import pandas as pd
import functions as fn

# "QMC" & System parameters
#seed = np.random.randint(2**32 - 1)
seed = 343
np.random.seed(seed)
shift = 0.0
cycles = 20
target = 25
tau = 0.005
zeta = 0.1
reports = int(target/(tau*cycles))
beta_loops = 50
Nattempts = int(5E3)
init = 'thermal-uniform'

H, Heval, HS = fn.system_initialize('EQUILIBRIUM-H6-STO3G.hamil', shift)
H0 = fn.non_interacting(H)

# Data Saving
data = []
csvname = 'adaptive-asym-ipdmqmc-asym-dmqmc-stochastic-rounding'
            
for betaloop in range(1,beta_loops+1):
    print(' # Beta Loop:', betaloop)

    iteration, report = 0, 0
    d, occrows, df = fn.initialize_dm(init, Nattempts, 3.0, Heval, HS)
    fn.write_report(iteration, tau, shift, d, Heval, df=df, stdout=True)

    for report in range(reports):
    
        for cycle in range(cycles):
            iteration += 1

            if round(iteration*tau,4) > 3.0:
                '''
                    "DMQMC" non-symmetric propagator:
                    dd/dtau = -d @ H
                '''
                deltad = -tau * (d @ H)
                deltad = fn.stochastic_round(deltad)
                d += deltad
            else:
                '''
                    IP-DMQMC non-symmetric propagator:
                    dd/dtau = H0 @ d - d @ H
                '''
                deltad = tau*(H0 @ d - d @ H) # eqn 9 in IP-DMQMC paper; d = f
                deltad = fn.stochastic_round(deltad)
                d += deltad

        fn.write_report(iteration, tau, shift, d, Heval, df=df, stdout=True)
        if round(iteration*tau,4) == 3.0:
            print(' # Switching to asymmetric DMQMC propagation.')
        H, shift = fn.update_shift(H, HS, cycles, tau, df, zeta)

    fn.store_data(data, df, betaloop, beta_loops, csvname, '')

