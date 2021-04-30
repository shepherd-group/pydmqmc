#!/usr/bin/env python

import numpy as np
import pandas as pd
from functions import *

# "QMC" & System parameters
#seed = np.random.randint(2**32 - 1)
seeds = [7,8,9,10]
shift = 0.0
cycles = 1
targets = np.arange(1,11)
tau = 0.1
#reports = int(target/(tau*cycles))
beta_loops = 50
attempts = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
init_dm = ['thermal-thermal', 'thermal-uniform', 'uniform-thermal', 'uniform-uniform']
simulations = zip(seeds, [attempts, attempts, attempts, attempts], init_dm)

H, Heval, HS = system_initialize('STRETCHED-H6-STO3G.hamil', shift)
H0 = np.diag(np.diag(H))

for seed, attempts, init in simulations:
    np.random.seed(seed)

    for target in targets:
        reports = int(target/(tau*cycles))

        for Nattempts in attempts:
            # Data Saving
            data = []
            outname  = './outputs/ip-dmqmc-beta-scan/'
            outname += init + '-analytical-nonsym-ipdmqmc'
            outname += '-Nbeta' + str(beta_loops)
            outname += '-Natt' + str(Nattempts)
            outname += '-seed' + str(seed)
            outname += '-tbeta' + str(target)
            
            for betaloop in range(1,beta_loops+1):
                print(' Beta Loop:', betaloop)
            
                iteration = 0
                report = 0
                f, occrows, df = initialize_dm(init, Nattempts, target, Heval, HS)
                write_report(iteration, tau, shift, f, Heval, df=df, stdout=True)
            
                for report in range(reports):
                
                    for cycle in range(cycles):
                        iteration += 1
                
                        '''
                            "IP-DMQMC"
                
                            df/dtau = H0 @ f - f @ H
                        '''
                
                        deltaf = tau * ( (H0 @ f) - (f @ H) )
                        
                        f += deltaf
                
                        write_report(iteration, tau, shift, f, Heval, df=df)
            
                data.append(pd.DataFrame(df))
            
            #data = pd.concat(data, ignore_index=True)
            #data.to_csv(outname + '.csv', index=False)
    
