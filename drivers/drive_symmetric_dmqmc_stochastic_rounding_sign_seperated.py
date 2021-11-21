#!/usr/bin/env python

import numpy as np
import pandas as pd
import functions as fn

# "QMC" & System parameters
#seed = np.random.randint(2**32 - 1)
seed = 17
np.random.seed(seed)
shift = 0.05
cycles = 10
target = 40
tau = 0.001
zeta = 0.1
reports = int(target/(tau*cycles))
beta_loops = 1
Nattempts = int(1E2)
init = 'uniform-uniform'

H, Heval, HS = fn.system_initialize('EQUILIBRIUM-H6-STO3G.hamil', shift)
Hpos, Hneg = fn.seperate_signs(H, diagonals=False)

# Data Saving
data = []
csvname = 'V2-sym-dmqmc-stochastic-rounding'
            
for betaloop in range(1,beta_loops+1):
    print(' # Beta Loop:', betaloop)

    iteration, report = 0, 0
    d, occrows, df = fn.initialize_dm(init, Nattempts, 3.0, Heval, HS)
    df = fn.write_report(iteration, tau, shift, d, Heval, df=df, stdout=True)

    for report in range(reports):
    
        for cycle in range(cycles):
            iteration += 1

            '''
                "DMQMC" non-symmetric propagator:
                dd/dtau = -(H @ d + d @ H)
                        = -((Hpos - Hneg) @ (dpos - dneg) + (dpos - dneg) @ (Hpos - Hneg))
                        = -(Hpos @ dpos - Hpos @ dneg - Hneg @ dpos + Hneg @ dneg
                            dpos @ Hpos - dpos @ Hneg - dneg @ Hpos + dneg @ Hneg)
                        Pos:            Neg:
                        Hpos @ dpos     Hpos @ dneg
                        Hneg @ dneg     Hneg @ dpos
                        dpos @ Hpos     dpos @ Hneg
                        dneg @ Hneg     dneg @ Hpos
            '''
            dpos, dneg = fn.seperate_signs(d, diagonals=False)

            deltadpos = Hpos @ dpos + Hneg @ dneg + dpos @ Hpos + dneg @ Hneg
            deltadpos = fn.stochastic_round(tau*deltadpos)

            deltadneg = Hpos @ dneg + Hneg @ dpos + dpos @ Hneg + dneg @ Hpos
            deltadneg = fn.stochastic_round(tau*deltadneg)

            deltad    = -(deltadpos - deltadneg)
            d += deltad

        df = fn.write_report(iteration, tau, shift, d, Heval, df=df, stdout=True)
        #H, shift = fn.update_shift(H, HS, cycles, tau, df, zeta)

    data = fn.store_data(data, df, betaloop, beta_loops, csvname, '')

