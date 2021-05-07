#!/usr/bin/env python

import numpy as np
import pandas as pd
from functions import *

# "QMC" & System parameters
#seed = np.random.randint(2**32 - 1)
#np.random.seed(seed)
shift = 0.0
cycles = 20
target = 10
tau = 0.005
zeta = 0.3
reports = int(target/(tau*cycles))
beta_loops = 1
Nattempts = 200
init = 'deterministic-uniform'

H, Heval, HS = system_initialize('STRETCHED-H6-STO3G.hamil', shift)
H0 = np.diag(np.diag(H))

# Data Saving
data = []
path  = './outputs/dmqmc-variable-shift/'
csvname = 'variableshift-analytical-nonsym-dmqmc'
            
for betaloop in range(1,beta_loops+1):
    print(' Beta Loop:', betaloop)

    iteration = 0
    report = 0
    d, occrows, df = initialize_dm(init, Nattempts, target, Heval, HS)
    write_report(iteration, tau, shift, d, Heval, df=df, stdout=True)

    for report in range(reports):
    
        for cycle in range(cycles):
            iteration += 1
    
            '''
                "DMQMC" non-symmetric propagator:

                dd/dtau = -H @ d
            '''

            deltad = -tau * (H @ d)
            
            d += deltad

        write_report(iteration, tau, shift, d, Heval, df=df, stdout=True)
        H, shift = update_shift(H, HS, cycles, tau, df, zeta)

    #data = store_data(data, df, betaloop, beta_loops, csvname, path)

