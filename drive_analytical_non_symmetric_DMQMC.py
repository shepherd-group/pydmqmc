#!/usr/bin/env python

import numpy as np
import pandas as pd
from functions import *

# "QMC" & System parameters
#seed = np.random.randint(2**32 - 1)
seeds = [11,12,13,14]
shift = 0.0
cycles = 20
targets = [10]
tau = 0.005
#reports = int(target/(tau*cycles))
beta_loops = 50
attempts = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
init_dm = ['thermal-thermal', 'thermal-uniform', 'uniform-thermal', 'uniform-uniform']
simulations = zip(seeds, [attempts, attempts, attempts, attempts], init_dm)

H, Heval, HS = system_initialize('STRETCHED-H6-STO3G.hamil', shift)
H0 = np.diag(np.diag(H))

for seed, attempts, init in simulations:
    np.random.seed(seed)
    print('\n', ' Initalization:', init)

    for target in targets:
        print(' Target:', target)
        reports = int(target/(tau*cycles))

        for Nattempts in attempts:
            print(' Nattempts:', Nattempts)
            # Data Saving
            data = []
            outname  = './outputs/dmqmc-beta-scan/'
            outname += init + '-analytical-nonsym-dmqmc'
            outname += '-Nbeta' + str(beta_loops)
            outname += '-Natt' + str(Nattempts)
            outname += '-seed' + str(seed)
            outname += '-tbeta' + str(target)
            
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
                
                        write_report(iteration, tau, shift, d, Heval, df=df)
            
                data.append(pd.DataFrame(df))
            
            data = pd.concat(data, ignore_index=True)
            data.to_csv(outname + '.csv', index=False)
    
