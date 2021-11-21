#!/usr/bin/env python

import numpy as np
import pandas as pd
from functions import *

# "QMC" & System parameters
#seed = np.random.randint(2**32 - 1)
#np.random.seed(seed)
shift = 0.0
cycles = 10
target = 25
tau = 0.001
zeta = 0.3
reports = int(target/(tau*cycles))
beta_loops = 1
Nattempts = 1
init = 'deterministic-uniform'

H, Heval, HS = system_initialize('STRETCHED-H6-STO3G.hamil', shift)

for row in range(HS):
    # Data Saving
    data = []
    path  = ''
    csvname = str(row+1).zfill(5)+'ROW-nonsym-dmqmc'

    for betaloop in range(1,beta_loops+1):
        print(' Beta Loop:', betaloop)

        iteration = 0
        report = 0
        d, occrows, df = initialize_dm(init, Nattempts, target, Heval, HS)
        d *= 0.0
        d[row,row] = 1
        write_report(iteration, tau, shift, d, Heval, df=df, stdout=True)

        for report in range(reports):
        
            for cycle in range(cycles):
                iteration += 1
        
                '''
                    "DMQMC" non-symmetric propagator:

                    dd/dtau = -H @ d
                '''

                deltad = -tau * (d @ H)
                
                d += deltad

            write_report(iteration, tau, shift, d, Heval, df=df, stdout=False)
            #H, shift = update_shift(H, HS, cycles, tau, df, zeta)

        data = store_data(data, df, betaloop, beta_loops, csvname, path)

