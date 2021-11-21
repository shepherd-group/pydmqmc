#!/usr/bin/env python

import numpy as np
import pandas as pd
import functions as fn

# "QMC" & System parameters
shift = 0.0
cycles = 10
target = 25
tau = 0.001
reports = int(target/(tau*cycles))
beta_loops = 1
Nattempts = 1
init = 'deterministic-uniform'
Ntarget = 1
variable_shift = True
zeta = 0.05

H, Heval, HS = fn.system_initialize('STRETCHED-H6-STO3G.hamil', shift)

# Data Saving
data = []
csvname = 'OG-STR-H6-STO3G-THROUGH-PLAT'

for betaloop in range(1,beta_loops+1):
    print(' Beta Loop:', betaloop)

    iteration, report = 0, 0
    d, occrows, df = fn.initialize_dm(init, Nattempts, target, Heval, HS)
    df = fn.write_report(iteration, tau, shift, d, Heval, df=df, stdout=True)

    for report in range(reports):
    
        for cycle in range(cycles):
            '''
                DMQMC symmetric propagator:
                dd/dtau = -0.5*( H @ d + d @ H)
            '''
            iteration += 1
            d -= (tau/2)*(H @ d + d @ H)

        df = fn.write_report(iteration, tau, shift, d, Heval, df=df, stdout=True)

        nw = df['Nw'][-1]
        nw_old = df['Nw'][-2]
        if variable_shift:
            H, shift = fn.update_shift(shift, nw, nw_old, zeta, tau, cycles, H, HS)

    data = fn.store_data(data, df, betaloop, beta_loops, csvname, '')

