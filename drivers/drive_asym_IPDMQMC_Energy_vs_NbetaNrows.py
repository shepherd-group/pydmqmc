#!/usr/bin/env python

import numpy as np
import pandas as pd
import functions as fn

# "QMC" & System parameters
#seed = np.random.randint(2**32 - 1)
seed = 23
np.random.seed(seed)
shift = 0.0
cycles = 20
#target = 7
tau = 0.005
#reports = int(target/(tau*cycles))
beta_loops = 1887
Nattempts = 1
init = 'thermal-uniform'

H, Heval, HS = fn.system_initialize('STRETCHED-H6-STO3G.hamil')
H0 = np.diag(np.diag(H))

seed = 350
for target in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    reports = int(target/(tau*cycles))
    seed += 1
    np.random.seed(seed)
    for Natt in [1, 4, 40, 100, 300, 3000]:
        data = []
        csvname  = str(Natt)+'Natt-'
        csvname += str(target)+'Tbeta-'
        csvname += str(seed)+'Seed-ASYMMETRIC-IPDMQMC-STR-H6'

        for betaloop in range(1,beta_loops+1):
            #print(' Beta Loop:', betaloop)

            iteration, report = 0, 0
            f, occ, df = fn.initialize_dm(init, Natt, target, Heval, HS)
            df = fn.write_report(iteration, tau, shift, f, Heval, df=df, stdout=True)

            for report in range(reports):
                for cycle in range(cycles):
                    iteration += 1
            
                    '''
                        "IP-DMQMC" non-symmetric propagator:
        
                        dd/dtau = (H0 @ f - f @ H)
                    '''
                    deltaf = tau * ((H0 @ f) - (f @ H))
                    f += deltaf
        
                df = fn.write_report(iteration, tau, shift, f, Heval, df=df)
        
            data = fn.store_data(data, df, betaloop, beta_loops, csvname, '')

